import os
from math import ceil
from os.path import join
from typing import Dict, List
from torch.distributed import get_rank

import pytorch_lightning as pl
import merlin.io
from merlin.dataloader.torch import Loader
from merlin.dtypes import boolean
from merlin.dtypes import float32, int64, int32, string
from merlin.schema import ColumnSchema, Schema


PARQUET_SCHEMA = {
    'X': int32,
    'X_niche_0': int32,
    'X_niche_1': int32,
    'X_niche_2': int32,
    'X_niche_3': int32,
    'X_niche_4': int32,
    'density_0': float32,
    'density_1': float32,
    'density_2': float32,
    'density_3': float32,
    'density_4': float32,
    'niche': int64,
    'author_cell_type': int64,
    'region': int64,
    'soma_joinid': int64,
    'is_primary_data': boolean,
    'dataset_id': int64,
    'donor_id': int64,
    'assay': int64,
    'cell_type': int64,
    'development_stage': int64,
    'disease': int64,
    'tissue': int64,
    'tissue_general': int64,
    'tech_sample': int64,
    'idx': int64,
    'specie': int64,
    'modality': int64,
    'organism': int64,
    'measured_genes': int32,
}


def merlin_dataset_factory(path: str, columns: List[str], dataset_kwargs: Dict[str, any]):
    return merlin.io.Dataset(
        path,
        engine='parquet',
        schema=Schema(
            [
                ColumnSchema(
                    'X', dtype=PARQUET_SCHEMA['X'],
                    is_list=True, is_ragged=False,
                    properties={'value_count': {'max': 2048}}
                )
            ] +
            [ColumnSchema(col, dtype=PARQUET_SCHEMA[col]) for col in columns]
        ),
        **dataset_kwargs
    )


def set_default_kwargs_dataloader(kwargs: Dict[str, any] = None, training: bool = True):
    assert isinstance(training, bool)
    if kwargs is None:
        kwargs = {}
    if 'parts_per_chunk' not in kwargs:
        kwargs['parts_per_chunk'] = 8 if training else 1
    # if 'drop_last' not in kwargs:
    #     kwargs['drop_last'] = training
    if'shuffle' not in kwargs:
        kwargs['shuffle'] = training

    return kwargs


def set_default_kwargs_dataset(kwargs: Dict[str, any] = None, training: bool = True):
    if kwargs is None:
        kwargs = {}
    if all(['part_size' not in kwargs, 'part_mem_fraction' not in kwargs]):
        kwargs['part_size'] = '50MB' if training else '80MB'

    return kwargs


def _get_data_files_distributed(base_path: str, split: str, world_size: int, sub_sample_frac: float = 1):
    files_devices = []
        
    for device in range(world_size):
        files = [file for file in os.listdir(join(base_path, split)) if (file.endswith('.parquet')) and ((int(file.split('.')[0].split('-')[1]) % world_size)==device)]
        files = [join(base_path, split, file) for file in sorted(files, key=lambda x: int(x.split('.')[0].split('-')[1]))]
        files.sort(reverse=True)
        files_devices.append(files[:ceil(sub_sample_frac * len(files))])
        
    return files_devices


def _create_single_distributed_dataset(files_devices: List[str], columns: List[str], path: str, world_size: int, dataset_kwargs_train: Dict[str, any] = None, dataset_kwargs_inference: Dict[str, any] = None):

    datasets = []

    for device in range(world_size):
        dataset = merlin_dataset_factory(
            files_devices[device],
            columns,
            set_default_kwargs_dataset(dataset_kwargs_train, training=True)
            )
        
        datasets.append(dataset)
        
    return datasets
    
    
class MerlinDataModuleDistributed(pl.LightningDataModule):

    def __init__(
            self,
            path: str,
            columns: List[str],
            batch_size: int,
            world_size: int,
            sub_sample_frac: float = 1.,
            splits: bool = True,
            dataloader_kwargs_train: Dict[str, any] = None,
            dataloader_kwargs_inference: Dict[str, any] = None,
            dataset_kwargs_train: Dict[str, any] = None,
            dataset_kwargs_inference: Dict[str, any] = None,
    ):
        super().__init__()
        for col in columns:
            assert col in PARQUET_SCHEMA
        
        key = 'train'
        if splits:
            key = 'test'
        
        files_devices_train = _get_data_files_distributed(path, 'train', world_size=world_size, sub_sample_frac=sub_sample_frac)
        files_devices_val = _get_data_files_distributed(path, key, world_size=world_size, sub_sample_frac=sub_sample_frac)
        files_devices_test = _get_data_files_distributed(path, key, world_size=world_size, sub_sample_frac=sub_sample_frac)
                    
        self.dataloader_kwargs_train = set_default_kwargs_dataloader(dataloader_kwargs_train, training=True)
        self.dataloader_kwargs_inference = set_default_kwargs_dataloader(dataloader_kwargs_inference, training=False)
        
        self.train_datasets = _create_single_distributed_dataset(files_devices=files_devices_train,
                                                                columns=columns,
                                                                path=path,
                                                                world_size=world_size,
                                                                dataset_kwargs_train=dataset_kwargs_train,
                                                                dataset_kwargs_inference=dataset_kwargs_inference)
        
        self.val_datasets = _create_single_distributed_dataset(files_devices=files_devices_val,
                                                                columns=columns,
                                                                path=path,
                                                                world_size=world_size,
                                                                dataset_kwargs_train=dataset_kwargs_train,
                                                                dataset_kwargs_inference=dataset_kwargs_inference)
        
        self.test_datasets = _create_single_distributed_dataset(files_devices=files_devices_test,
                                                                columns=columns,
                                                                path=path,
                                                                world_size=world_size,
                                                                dataset_kwargs_train=dataset_kwargs_train,
                                                                dataset_kwargs_inference=dataset_kwargs_inference)
        
        self.batch_size = batch_size
        
        self.prepare_data_per_node = True
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False
        
    def train_dataloader(self):
        return Loader(self.train_datasets[get_rank()], batch_size=self.batch_size, **self.dataloader_kwargs_train)

    def val_dataloader(self):
        return Loader(self.val_datasets[get_rank()], batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def test_dataloader(self):
        return Loader(self.test_datasets[get_rank()], batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def predict_dataloader(self):
        return Loader(self.test_datasets, batch_size=self.batch_size, **self.dataloader_kwargs_inference)
