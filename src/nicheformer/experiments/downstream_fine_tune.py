import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import anndata as ad
from typing import Optional, Dict, Any

import argparse
import yaml

from ..models.nicheformer import Nicheformer
from ..models._nicheformer_fine_tune import NicheformerFineTune
from ..data.dataset import NicheformerDataset

def fine_tune_predictions(config: Optional[dict[str, Any]] = None) -> None:
    """
    Fine-tune a pre-trained Nicheformer model in a downsttream task and store predictions in AnnData.

    Args:
        config (dict): Configuration dictionary containing all necessary parameters
    """
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Load data
    adata = ad.read_h5ad(config['data_path'])
    technology_mean = np.load(config['technology_mean_path'])

    # Create datasets
    train_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=config.get('max_seq_len', 4096),
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000)
    )

    val_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='val',
        max_seq_len=config.get('max_seq_len', 4096),
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000)
    )

    test_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='test',
        max_seq_len=config.get('max_seq_len', 4096),
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    # Load pre-trained model
    model = Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path'])

    # Create fine-tuning model
    fine_tune_model = NicheformerFineTune(
        backbone=model,
        supervised_task=config['supervised_task'],
        extract_layers=config['extract_layers'],
        function_layers=config['function_layers'],
        lr=config['lr'],
        warmup=config['warmup'],
        max_epochs=config['max_epochs'],
        dim_prediction=config['dim_prediction'],
        n_classes=config['n_classes'],
        baseline=config['baseline'],
        freeze=config['freeze'],
        reinit_layers=config['reinit_layers'],
        extractor=config['extractor'],
        regress_distribution=config['regress_distribution'],
        pool=config['pool'],
        predict_density=config['predict_density'],
        ignore_zeros=config['ignore_zeros'],
        organ=config.get('organ', 'unknown'),
        label=config['label'],
        without_context=True
    )

    # Set up model checkpointing if needed
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['output_dir'], 'checkpoints'),
        filename='nicheformer-{epoch:02d}-{val/accuracy:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val/accuracy',
        mode='max'
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        #callbacks=[checkpoint_callback],
        default_root_dir=config['output_dir'],
        precision=config.get('precision', 32),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 10),
    )

    # Train the model
    print("Training the model...")
    trainer.fit(
        model=fine_tune_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Test the model
    print("Testing the model...")
    test_results = trainer.test(
        model=fine_tune_model,
        dataloaders=test_loader
    )

    # Test the model and get predictions
    print("Getting predictions...")
    predictions = trainer.predict(fine_tune_model, dataloaders=test_loader)
    predictions = [torch.cat([p[0] for p in predictions]).cpu().numpy(),
                  torch.cat([p[1] for p in predictions]).cpu().numpy()]
    if 'regression' in config['supervised_task']:
        predictions = predictions[0]  # For regression both values are the same

    # Store predictions in AnnData object
    prediction_key = f"predictions_{config.get('label', 'X_niche_1')}"
    test_mask = adata.obs.nicheformer_split == 'test'

    if 'classification' in config['supervised_task']:
        # For classification tasks
        adata.obs.loc[test_mask, f"{prediction_key}_class"] = predictions
    else:
        # For regression tasks
        adata.obs.loc[test_mask, prediction_key] = predictions

    # Store test metrics
    for metric_name, value in test_results[0].items():
        adata.uns[f"{prediction_key}_metrics_{metric_name}"] = value

    # Save updated AnnData
    adata.write_h5ad(config['output_path'])

    print(f"Results saved to {config['output_path']}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained Nicheformer model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fine_tune_predictions(config=config)
