import scanpy as sc
import sys

import os 
import sys
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from nicheformer.data.constants import DefaultPaths, ObsConstants, UnsConstants, VarConstants, AssayOntologyTermId, SexOntologyTermId, OrganismOntologyTermId, TissueOntologyTermId, SuspensionTypeId
from nicheformer.data.validate import validate

if len(sys.argv)==1:
    path = DefaultPaths.SPATIAL
else:
    path = sys.argv[1]

raw_path = f"{path}/raw"
preprocessed_path = f"{path}/preprocessed"

if not os.path.exists(f"{preprocessed_path}"):
    os.mkdir(f"{preprocessed_path}")

adata = sc.read_10x_h5(f"{raw_path}/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/cell_feature_matrix.h5")

adata.obs = pd.read_csv(
    f"{raw_path}/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/cells.csv.gz", index_col=0
)

adata.var = adata.var.reset_index().rename(columns={'index': 'gene_name'}).set_index('gene_ids')
adata.var.index.name = None

# setting MERFISH for now until an official ontology term is released
assay = str(AssayOntologyTermId.MERFISH_SPATIAL.value)
sex = str(SexOntologyTermId.UNKNOWN.value)
organism = str(OrganismOntologyTermId.HUMAN.value)
organism_validator = "human"
tissue = str(TissueOntologyTermId.LUNG.value)
suspension_type = str(SuspensionTypeId.SPATIAL.value)
tissue_type = "tissue"

adata.X = csr_matrix(adata.X)
adata.obs[ObsConstants.SPATIAL_X] = adata.obs['x_centroid']
adata.obs[ObsConstants.SPATIAL_Y] = adata.obs['y_centroid']
adata.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical([assay for i in range(len(adata))])
adata.obs[ObsConstants.SEX_ONTOLOGY_TERM_ID] = pd.Categorical([sex for i in range(len(adata))])
adata.obs[ObsConstants.ORGANISM_ONTOLOGY_TERM_ID] = pd.Categorical([organism for i in range(len(adata))])
adata.obs[ObsConstants.TISSUE_ONTOLOGY_TERM_ID] = pd.Categorical([tissue for i in range(len(adata))])
adata.obs[ObsConstants.SUSPENSION_TYPE] = pd.Categorical([suspension_type for i in range(len(adata))])

adata.obs[ObsConstants.DONOR_ID] = pd.Categorical(['Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs' for i in range(len(adata))])
adata.obs[ObsConstants.CONDITION_ID] = pd.Categorical(['non diseased' for i in range(len(adata))])
adata.obs[ObsConstants.TISSUE_TYPE] = pd.Categorical([tissue_type for i in range(len(adata))])

adata.uns[UnsConstants.TITLE] = "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs"
adata.var[VarConstants.FEATURE_IS_FILTERED] = False

adata.obs[ObsConstants.LIBRARY_KEY] = pd.Categorical(['section' for i in range(len(adata))])

adata_output, valid, errors, is_seurat_convertible = validate(adata, organism=organism_validator)

adata_output.obs['assay'] = pd.Categorical(['Xenium' for i in range(len(adata_output))])
adata_output.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical(['no yet defined' for i in range(len(adata_output))])

adata_output.obs[ObsConstants.DATASET] = adata_output.uns['title']
adata_output.obs[ObsConstants.SPLIT] = 'train'
adata_output.obs[ObsConstants.NICHE] = 'nan'
adata_output.obs[ObsConstants.REGION] = 'nan'

adata_output.write(f"{preprocessed_path}/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.h5ad")
