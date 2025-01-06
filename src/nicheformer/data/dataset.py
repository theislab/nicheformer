import gc
import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import issparse
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.utils import sparsefuncs
import numba


def sf_normalize(X):
    """Normalize the input matrix to a scale of 10000."""
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x: np.array, max_seq_len: int = -1, aux_tokens: int = 30):
    """Tokenize the input gene vector"""
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    for i, cell in enumerate(x):
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len] 
        sorted_indices = sorted_indices + aux_tokens # we reserve some tokens for padding etc (just in case)
        if max_seq_len:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros_like(cell, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)

        scores_final[i, :] = scores

    return scores_final


def tokenize_data(x: np.array, median_counts_per_gene: np.array, max_seq_len: int = None):
    """Tokenize the input gene vector to a vector of 32-bit integers."""
    x = np.nan_to_num(x) # is NaN values, fill with 0s
    x = sf_normalize(x)
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))

    scores_final = _sub_tokenize_data(out, 4096, 30)

    return scores_final.astype('i4')


class NicheformerDataset(Dataset):
    """Dataset for Nicheformer"""

    def __init__(self, adata, technology_mean, split='train', max_seq_len=4096, aux_tokens=30, chunk_size=1000,
                 metadata_fields=None):
        """
        Initialize the dataset

        Args:
            adata (AnnData): Annotated data matrix
            technology_mean (np.array): technology mean
            split (str): 'train', 'test', or 'val'
            max_seq_len (int): Maximum sequence length for tokenization
            aux_tokens (int): Number of reserved tokens
            chunk_size (int): Number of cells to process at once
            metadata_fields (dict): Dictionary specifying which metadata fields to include.
                                  Format: {
                                      'obs': ['field1', 'field2'],  # fields from adata.obs
                                      'obsm': ['field3', 'field4']  # fields from adata.obsm
                                  }
        """
        self.adata = adata[adata.obs.nicheformer_split == split].copy()
        self.technology_mean = technology_mean
        self.max_seq_len = max_seq_len
        self.aux_tokens = aux_tokens
        self.chunk_size = chunk_size
        self.metadata_fields = metadata_fields or {'obs': [], 'obsm': []}

        # Initialize storage for tokenized data
        self.n_cells = len(self.adata)
        self.tokens = None

        # Process data in chunks
        self._process_chunks()

        # Store metadata
        self._prepare_metadata()

    def _process_chunks(self):
        n_chunks = (self.n_cells + self.chunk_size - 1) // self.chunk_size
        tokens_list = []

        for chunk_idx in tqdm(range(n_chunks)):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, self.n_cells)

            # Process chunk
            chunk_tokens = self._process_chunk(start_idx, end_idx)
            tokens_list.append(chunk_tokens)

            torch.cuda.empty_cache()
            gc.collect()  # force garbage collection

        # Concatenate all chunks
        self.tokens = np.concatenate(tokens_list, axis=0)

    def _process_chunk(self, start_idx, end_idx):
        # Get chunk of data
        chunk_adata = self.adata[start_idx:end_idx]

        # Convert sparse to dense for this chunk only
        if issparse(chunk_adata.X):
            x = chunk_adata.X.toarray()
        else:
            x = chunk_adata.X

        # Process chunk
        x = np.nan_to_num(x)
        x = sf_normalize(x)

        tech_mean = self.technology_mean
        tech_mean += tech_mean == 0
        x = x / tech_mean.reshape((1, -1))

        # Tokenize
        tokens = _sub_tokenize_data(x, self.max_seq_len, self.aux_tokens).astype(np.int32)

        return tokens

    def _prepare_metadata(self):
        self.metadata = {}

        # Process obs fields - direct assignment, no special processing needed
        for field in self.metadata_fields.get('obs', []):
            self.metadata[field] = self.adata.obs[field].values

        # Process obsm fields - chunk processing only if sparse
        for field in self.metadata_fields.get('obsm', []):
            if issparse(self.adata.obsm[field]):
                vectors = []
                for chunk_idx in range(0, self.n_cells, self.chunk_size):
                    end_idx = min(chunk_idx + self.chunk_size, self.n_cells)
                    chunk = self.adata.obsm[field][chunk_idx:end_idx].toarray()
                    vectors.append(chunk)
                self.metadata[field] = np.concatenate(vectors, axis=0)
            else:
                self.metadata[field] = self.adata.obsm[field]

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        item = {
            'X': torch.tensor(self.tokens[idx])
        }

        # Add all metadata fields to the item
        for key, value in self.metadata.items():
            item[key] = torch.tensor(value[idx])

        return item
