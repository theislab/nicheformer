import anndata
import os 
import sys
import pandas as pd
import scanpy as sc
from tqdm import tqdm

from nicheformer.data.constants import DefaultPaths, ObsConstants, UnsConstants, VarConstants, AssayOntologyTermId, SexOntologyTermId, OrganismOntologyTermId, TissueOntologyTermId, SuspensionTypeId
from nicheformer.data.tools import qc_filter
from nicheformer.data.validate import validate

if len(sys.argv)==1:
    path = DefaultPaths.DISSOCIATED
else:
    path = sys.argv[1]

raw_path = f"{path}/raw"
preprocessed_path = f"{path}/preprocessed"

if not os.path.exists(f"{preprocessed_path}"):
    os.mkdir(f"{preprocessed_path}")

# In most cases only this section needs to be updates for the dataset
geo_id = 'GSE172127'
doi = "10.1038/s41421-021-00266-1"

## manual entries that are equal across the dataset
assay = str(AssayOntologyTermId.TENX_3V2.value)
sex = str(SexOntologyTermId.FEMALE.value)
organism = str(OrganismOntologyTermId.MOUSE.value)
organism_validator = "mouse"
tissue = str(TissueOntologyTermId.LIVER.value)
suspension_type = str(SuspensionTypeId.SMART_SEQ_CELL.value)
tissue_type = "tissue" # or alternatively "organoid"
condition_id = "wild type"

sample_ids = ["GSM5242402_E14.5FL", "GSM5242403_E14.5FL_HSC"]
adatas = []
with tqdm(total=len(sample_ids), desc =geo_id) as pbar:
    for sample in sample_ids:
        # reading of raw files
        adata = sc.read_mtx(f"{raw_path}/{geo_id}/{sample}_matrix.mtx.gz").T
        features = pd.read_table(f"{raw_path}/{geo_id}/{sample}_features.tsv.gz", index_col=0, header=None)
        features.columns = ["gene_name", "feature_types"]
        features.index.name = 'gene_ids'
        barcodes = pd.read_table(f"{raw_path}/{geo_id}/{sample}_barcodes.tsv.gz", index_col=0, header=None)
        barcodes.index.name = None
        adata.var = features
        adata.obs = barcodes

        # Ontology terms defined for AnnData object
        adata.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical([assay for i in range(len(adata))])
        adata.obs[ObsConstants.SEX_ONTOLOGY_TERM_ID] = pd.Categorical([sex for i in range(len(adata))])
        adata.obs[ObsConstants.ORGANISM_ONTOLOGY_TERM_ID] = pd.Categorical([organism for i in range(len(adata))])
        adata.obs[ObsConstants.TISSUE_ONTOLOGY_TERM_ID] = pd.Categorical([tissue for i in range(len(adata))])
        adata.obs[ObsConstants.SUSPENSION_TYPE] = pd.Categorical([suspension_type for i in range(len(adata))])

        # NicheFormer data schema
        adata.obs[ObsConstants.CONDITION_ID] = pd.Categorical([condition_id for i in range(len(adata))])
        adata.obs[ObsConstants.DONOR_ID] = pd.Categorical([sample for i in range(len(adata))])
        adata.obs[ObsConstants.TISSUE_TYPE] = pd.Categorical([tissue_type for i in range(len(adata))])

        adatas.append(adata)
        pbar.update(1)

adata = anndata.concat(adatas, index_unique='_')
adata.uns[UnsConstants.TITLE] = doi
adata.var[VarConstants.FEATURE_IS_FILTERED] = False

# after concatenation these are dtype=object, but need to be category
adata.obs[ObsConstants.CONDITION_ID] = adata.obs[ObsConstants.CONDITION_ID].astype('category')
adata.obs[ObsConstants.DONOR_ID] = adata.obs[ObsConstants.DONOR_ID].astype('category')

# run basic filtering with default values
print(f"\nPerforming basic quality control for {geo_id}.")
adata = qc_filter(adata=adata)

# run validator
print(f"\nValidating {geo_id}.")
adata_output, valid, errors, is_seurat_convertible = validate(adata, organism=organism_validator)

if valid:
    print(f"DONE: Successfully preprocessed {geo_id}, validation completed with status is_valid={valid}.")
    print(f"\nWRITING PREPROCESSED FILE TO: {geo_id}.h5ad")
    adata_output.write(f"{preprocessed_path}/{geo_id}.h5ad")
else:
    print(f"ERROR: Preprocessing of {geo_id} failed, validation completed with status is_valid={valid}.")
