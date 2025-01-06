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
from ..data.dataset import NicheformerDataset

def fine_tune_pretraining(config: Optional[dict[str, Any]] = None) -> None:
    """
    Fine-tune a pre-trained Nicheformer model and save the checkpoint.

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

    # Load pre-trained model and modify for fine-tuning
    model = Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path'])
    
    # Update model parameters for fine-tuning
    model.lr = config['lr']
    model.warmup = config['warmup']
    model.max_epochs = config['max_epochs']

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['output_dir'], 'checkpoints'),
        filename='nicheformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],  # Enable checkpoint callback
        default_root_dir=config['output_dir'],
        precision=config.get('precision', 32),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 10),
    )

    # Train the model
    print("Training the model...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained Nicheformer model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fine_tune_pretraining(config=config)
