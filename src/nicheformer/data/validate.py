import logging
from datetime import datetime
from typing import Literal

from anndata import AnnData
from cellxgene_schema.ontology import GeneChecker, SupportedOrganisms
from cellxgene_schema.utils import enforce_canonical_format
from cellxgene_schema.validate import Validator as CellxgeneValidator
from cellxgene_schema.validate import logger
from cellxgene_schema.write_labels import AnnDataLabelAppender as CellxgeneAnnDataLabelAppender

from nicheformer.data.constants import AssayOntologyTermId, SexOntologyTermId, SuspensionTypeId, TissueOntologyTermId

METADATA_REMOVE = [
    "cell_type_ontology_term_id",
    "disease_ontology_term_id",
    # "tissue_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "development_stage_ontology_term_id",
    "is_primary_data",
]


def validate(
    adata: AnnData,
    organism: Literal["mouse", "human"] = "mouse",
    ignore_labels: bool = False,
    verbose: bool = False,
) -> tuple[AnnData, bool, list, bool]:
    """
    Entry point for validation.

    :param AnnData to validate.

    :return (True, [], <bool>) if successful validation, (False, [list_of_errors], <bool>) otherwise; last bool is for
    seurat convertibility
    :rtype tuple
    """
    # Perform validation
    start = datetime.now()
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # check genes and filter out invalid ones
    if organism == "mouse":
        gene_checker = GeneChecker(SupportedOrganisms.MUS_MUSCULUS)
    elif organism == "human":
        gene_checker = GeneChecker(SupportedOrganisms.HOMO_SAPIENS)
    else:
        raise ValueError("Only supported organisms are `mouse` and `human`.")

    valid_genes = []
    for gene in adata.var_names:
        if gene_checker.is_valid_id(gene):
            valid_genes.append(gene)
    if len(valid_genes) == 0:
        raise AssertionError(
            f"No valid {organism} genes are found! Make sure the variables are indexed by Ensembl Gene IDs!"
        )
    if len(valid_genes) != len(adata.var_names):
        logger.warning(f"WARNING: Found {len(adata.var_names) - len(valid_genes)} invalid genes, subsetting anndata")
        # subset anndata to only valid genes, since sparse should be fine
        adata = adata[:, valid_genes].copy()

    validator = CurrentFormat(ignore_labels=ignore_labels)
    validator.validate_adata(adata)
    validator.validate_nicheformer(adata)
    logger.info(f"Validation complete in {datetime.now() - start} with status is_valid={validator.is_valid}")

    # Stop if validation was unsuccessful
    if not validator.is_valid:
        _print_valid_ontologies(validator.errors)
        return None, False, validator.errors, validator.is_seurat_convertible

    label_start = datetime.now()
    writer = AnnDataLabelAppender(validator)
    adata = writer.write_labels()

    logger.info(f"H5AD label writing complete in {datetime.now() - label_start}")

    if len(writer.errors):
        return None, False, writer.errors, validator.is_seurat_convertible

    return adata, True, validator.errors + writer.errors, validator.is_seurat_convertible


def _print_valid_ontologies(errors: list[str]) -> None:
    enum_map = {
        "assay_ontology_term_id": AssayOntologyTermId,
        "organism_ontology_term_id": SupportedOrganisms,
        "sex_ontology_term_id": SexOntologyTermId,
        "suspension_type": SuspensionTypeId,
        "tissue_ontology_term_id": TissueOntologyTermId,
    }
    for error in errors:
        for key, enum_class in enum_map.items():
            if key in error:
                logger.info(f"These are valid `{key}`:")
                logger.info("FOR:\tUSE")
                for enum in enum_class:
                    logger.info(f"{enum.name}:\t{enum.value}")


class Validator(CellxgeneValidator):
    """Validator patched from cellxgene."""

    def __init__(self, ignore_labels: bool = False):
        super().__init__(ignore_labels=ignore_labels)

    def _get_component_def(self, component: str) -> dict:
        """
        Gets the definition of an individual component in the schema (e.g. obs)

        :param component: the component name

        :rtype dict
        """
        if self.schema_def:
            comp = self.schema_def["components"][component]
            if component == "obs":
                for c in METADATA_REMOVE:
                    comp["columns"].pop(c, None)
            return comp
        else:
            raise RuntimeError("Schema has not been set in this instance class")

    def validate_adata(self, adata: AnnData) -> bool:
        """
        Validates adata

        :params Union[str, bytes, os.PathLike] h5ad_path: path to h5ad to validate, if None it will try to validate
        from self.adata

        :return True if successful validation, False otherwise
        :rtype bool
        """
        logger.info("Starting validation...")
        # Re-start errors in case a new h5ad is being validated
        self.errors = []

        if adata:
            logger.debug("Reading the h5ad file...")
            self.adata = adata

            # TODO: do something about this below
            # https://github.com/chanzuckerberg/single-cell-curation/issues/677
            # don't validate encoding as we support higher than 0.8.0
            # self._validate_encoding_version()

            logger.debug("Successfully read the h5ad file")

        # Fetches schema def from anndata if schema version is not found in AnnData, this fails
        self._set_schema_def()

        # removing emdedding requirement from schema
        del self.schema_def["components"]["obsm"]

        if not self.errors:
            self._deep_check()

        # Print warnings if any
        if self.warnings:
            self.warnings = ["WARNING: " + i for i in self.warnings]
            for w in self.warnings:
                logger.warning(w)

        # Print errors if any
        if self.errors:
            self.errors = ["ERROR: " + i for i in self.errors]
            for e in self.errors:
                logger.error(e)
            self.is_valid = False
        else:
            self.is_valid = True

        return self.is_valid


class Format001(Validator):
    """Validator patched from cellxgene."""

    def __init__(self, ignore_labels: bool = False):
        super().__init__(ignore_labels=ignore_labels)

    def validate_nicheformer(self, adata: AnnData) -> bool:
        """Validate nicheformer specific fields."""
        if not self.errors:
            self.errors = []

        self.nicheformer_errors = []
        if "condition_id" not in adata.obs:
            self.nicheformer_errors.append("Dataframe 'obs' is missing column 'condition_id'.")
        if "tissue_type" not in adata.obs:
            self.nicheformer_errors.append(
                "Dataframe 'obs' is missing column 'tissue_type'. Supported tissue types are `tissue` and `organoid`."
            )

        # Print errors if any
        if self.nicheformer_errors:
            self.nicheformer_errors = ["ERROR: " + i for i in self.nicheformer_errors]
            for e in self.nicheformer_errors:
                logger.error(e)
            self.is_valid = self.is_valid and False
        else:
            self.is_valid = self.is_valid and True

        self.errors += self.nicheformer_errors
        return self.is_valid

    @property
    def nicheformer_version(self) -> str:  # noqa: D102
        return "0.0.1"


CurrentFormat = Format001


class AnnDataLabelAppender(CellxgeneAnnDataLabelAppender):
    """Validator patched from cellxgene."""

    def __init__(self, validator: Validator):
        super().__init__(validator=validator)

    def write_labels(self) -> AnnData:
        """
        Convert ids to ontologies when valid.

        From a valid (per cellxgene's schema) h5ad, this function writes a new h5ad file with ontology/gene labels added
        to adata.obs  and adata.var respectively

        :param str add_labels_file: Path to new h5ad file with ontology/gene labels added

        :rtype None
        """
        logger.info("Writing labels")
        # Add labels in obs
        self._add_labels()

        # Remove unused categories
        self._remove_categories_with_zero_values()

        # Update version
        self.adata.uns["schema_version"] = self.validator.schema_version
        self.adata.uns["nicheformer_version"] = self.validator.nicheformer_version

        enforce_canonical_format(self.adata)

        # Print errors if any
        if self.errors:
            for e, tb in self.errors:
                logger.error(e, extra={"exec_info": tb})
            self.was_writing_successful = False
        else:
            self.was_writing_successful = True

        return self.adata
