from enum import Enum


class AssayOntologyTermId(Enum):
    """Assay ontology term ids."""

    TENX = "EFO:0030080"  # if version not known
    TENX_3 = "EFO:0030003"
    TENX_3V2 = "EFO:0009899"
    TENX_3V3 = "EFO:0009922"
    TENX_5 = "EFO:0030004"
    TENX_5V1 = "EFO:0011025"
    TENX_5V2 = "EFO:0009900"
    SMART_SEQ2 = "EFO:0008931"
    MERFISH_SPATIAL = "EFO:0008992"
    # this is the ontology id for GeoMx, since CosMx data has not yet it's own ontology id and general in-situ hybridization are not allowed
    COSMX_SPATIAL = "EFO:0030029"
    # VISIUM_SPATIAL_GENE_EXPRESSION = "EFO:0010961"


class SuspensionTypeId(Enum):
    """Suspension type id."""

    TENX_CELL = "cell"
    TENX_NUCLEUS = "nucleus"
    SMART_SEQ_CELL = "cell"
    SMART_SEQ_NUCLEUS = "nucleus"
    SPATIAL = "na"


class SexOntologyTermId(Enum):
    """There are many in, here just collecting 4"""

    FEMALE = "PATO:0000383"
    MALE = "PATO:0000384"
    HERMAPHRODITE = "PATO:0001340"
    UNKNOWN = "unknown"


class OrganismOntologyTermId(Enum):
    """Mouse and human"""

    MOUSE = "NCBITaxon:10090"
    HUMAN = "NCBITaxon:9606"


class TissueOntologyTermId(Enum):
    """There are many in, here just collecting 7"""

    BLOOD = "UBERON:0000178"
    BONE_MARROW = "UBERON:0002371"
    BRAIN = "UBERON:0000955"
    LIVER = "UBERON:0002107"
    LUNG = "UBERON:0002048"
    HEART = "UBERON:0000948"
    INTESTINE = "UBERON:0000160"
    KIDNEY = "UBERON:0002113"
    COLON = "UBERON:0001155"
    SKIN = "UBERON:0002097"
    PANCREAS = "UBERON:0001264"
    BREAST = "UBERON:0000310"


class GeneExpressionOmnibus:
    """Constants associated with any dataset deposited at the Gene Expression Omnibus (GEO)."""

    DOWNLOAD_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc="
    FORMAT = "&format=file"


class DefaultPaths:
    """Default paths used for data storage."""

    DISSOCIATED = "/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/dissociated"
    SPATIAL = "/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/spatial"


class ObsConstants:
    """Constants associated with obs in AnnData objects. These should primarily be used for loading scripts."""

    ASSAY_ONTOLOGY_TERM_ID = "assay_ontology_term_id"
    SEX_ONTOLOGY_TERM_ID = "sex_ontology_term_id"
    ORGANISM_ONTOLOGY_TERM_ID = "organism_ontology_term_id"
    TISSUE_ONTOLOGY_TERM_ID = "tissue_ontology_term_id"
    SUSPENSION_TYPE = "suspension_type"
    DEVELOPMENT_STAGE_ONTOLOGY_TERM_ID = "development_stage_ontology_term_id"
    DISEASE_ONTOLOGY_TERM_ID = "disease_ontology_term_id"
    SELF_REPORTED_ETHNICITY_ONTOLOGY_TERM_ID = "self_reported_ethnicity_ontology_term_id"
    CELL_TYPE_ONTOLOGY_TERM_ID = "cell_type_ontology_term_id"

    TISSUE_TYPE = "tissue_type"
    CONDITION_ID = "condition_id"
    SAMPLE_ID = "sample_id"
    DONOR_ID = "donor_id"

    # obs spatial schema
    AUTHOR_CELL_TYPE = "author_cell_type"
    LIBRARY_KEY = "library_key"
    NICHE = "niche"
    REGION = "region"
    SPLIT = "nicheformer_split"
    SPATIAL_X = "x"
    SPATIAL_Y = "y"
    DATASET = "dataset"

class UnsConstants:
    """Constants associated with uns in AnnData objects. These should primarily be used for loading scripts."""

    TITLE = "title"
    NICHE = "niche"


class ObsmConstants:
    """Constants associated with obsm in AnnData objects. These should primarily be used for loading scripts."""

    SPATIAL = "spatial"
    NICHE = "X_niche"


class VarConstants:
    """Constants associated with var in AnnData objects. These should primarily be used for loading scripts."""

    FEATURE_IS_FILTERED = "feature_is_filtered"
