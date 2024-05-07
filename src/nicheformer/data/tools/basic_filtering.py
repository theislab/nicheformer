import scanpy as sc
from anndata import AnnData
from typing import Optional


def qc_filter(
    adata: AnnData,
    min_counts: Optional[int]=200,
    min_cells: Optional[int]=5,
    
) -> AnnData:
    """Permissive quality control filtering for removing low quality cells in processing script.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_cells
        Minimum number of cells expressed required for a gene to pass filtering.

    Returns
    -------
    Filtered AnnData object.
    """
    print(f"AnnData object before filtering has {adata.n_obs} cells and {adata.n_vars} genes.")
    # Filter cell outliers based on counts and numbers of genes expressed.
    sc.pp.filter_cells(
        data=adata, 
        min_counts=min_counts, 
    )
    print(f"AnnData object after cell filtering: {adata.n_obs} cells, {adata.n_vars} genes.")
    # Filter genes based on number of cells or counts.
    sc.pp.filter_genes(
        data=adata, 
        min_cells=min_cells,
    )
    print(f"AnnData object after gene filtering: {adata.n_obs} cells, {adata.n_vars} genes.")
    return adata