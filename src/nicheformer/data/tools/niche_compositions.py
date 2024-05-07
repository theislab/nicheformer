from anndata import AnnData
from nicheformer.data.constants import ObsConstants, ObsmConstants, UnsConstants
from pandas import get_dummies
from scipy.sparse import csr_matrix
from squidpy.gr import spatial_neighbors
import numpy as np


def niche_compositions(
    adata: AnnData,
    radii: list[float]
) -> AnnData:
    f"""
    Calculates .obsm['{ObsConstants.NICHE}'] for a cell's niche (cells within a specific radius neighborhood).

    Parameters
    ----------
    adata
        Anndata object with information about .obs['{ObsConstants.AUTHOR_CELL_TYPE}'], .obsm['spatial'] and .obs['{ObsConstants.LIBRARY_KEY}']
    radii
        (multiple) radius for squidpy graph

    Returns
    -------
    Anndata object with the new key (or keys if multiple radius):
        :attr:`anndata.AnnData.obsm` ``['{ObsmConstants.NICHE}_n']`` - n_cells x n_cell_types matrix with n being index of radius. 
    """
    print("Calculate X_niche for...")
    one_hot_ct = get_dummies(adata.obs[ObsConstants.AUTHOR_CELL_TYPE], dtype=int, sparse=True)
    uns_niche = {}
    uns_niche['columns'] = list(one_hot_ct.columns)
    for n, radius in enumerate(radii):
        print(f"radius = {radius}.")
        connectivities, _ = spatial_neighbors(
            adata, 
            spatial_key=ObsmConstants.SPATIAL,
            library_key=ObsConstants.LIBRARY_KEY,
            coord_type='generic',
            radius=radius,
            copy=True
        )
        niche = connectivities @ csr_matrix(one_hot_ct)
        adata.obsm[f"{ObsmConstants.NICHE}_{n}"] = niche
        uns_niche[f"{UnsConstants.NICHE}_{n}"] = {
            'radius': radius,
            'coord_type': 'generic',
            'node_degree': np.mean(niche.sum(axis=1)),
            'library_key': ObsConstants.LIBRARY_KEY
        }
    adata.uns[UnsConstants.NICHE] = uns_niche
    return adata