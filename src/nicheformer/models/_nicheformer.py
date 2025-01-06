import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List, Dict, Optional, Union
from torch import optim
import numpy as np
import math

from nicheformer.models._utils import complete_masking

MASK_TOKEN = 0
CLS_TOKEN = 2

class Nicheformer(pl.LightningModule):
    def __init__(self,
                 dim_model: int,
                 nheads: int,
                 dim_feedforward: int,
                 nlayers: int,
                 dropout: float,
                 batch_first: bool,
                 masking_p: float,
                 n_tokens: int,
                 context_length: int,
                 lr: float,
                 warmup: int,
                 batch_size: int,
                 max_epochs: int,
                 cls_classes: int = 164,
                 supervised_task: Optional[str] = None,
                 learnable_pe: bool = True,
                 specie: bool = False,
                 assay: bool = False,
                 modality: bool = False,
                 contrastive: bool = False):
        """Initialize NicheformerBase.

        Args:
            dim_model: Dimensionality of the model
            nheads: Number of attention heads
            dim_feedforward: Dimensionality of MLPs in attention blocks
            nlayers: Number of transformer layers
            dropout: Dropout probability
            batch_first: Whether batch dimension is first
            masking_p: Probability of masking tokens
            n_tokens: Total number of tokens (excluding auxiliary)
            context_length: Length of the context window
            lr: Learning rate
            warmup: Number of warmup steps
            batch_size: Batch size
            max_epochs: Maximum number of epochs
            cls_classes: Number of classification classes
            supervised_task: Type of supervised task
            learnable_pe: Whether to use learnable positional embeddings
            specie: Whether to add specie token
            assay: Whether to add assay token
            modality: Whether to add modality token
            contrastive: Whether to use contrastive loss
        """
        super().__init__()
        self.save_hyperparameters()

        # Core transformer components
        self._init_transformer(dim_model, nheads, dim_feedforward, nlayers, dropout, batch_first)

        # Embedding layers
        self._init_embeddings(n_tokens, dim_model, context_length, learnable_pe, dropout)

        # Task-specific heads
        self._init_heads(dim_model, n_tokens)

        # Loss functions
        self.loss = nn.CrossEntropyLoss()
        if supervised_task is not None:
            self.cls_loss = nn.CrossEntropyLoss()

        # Training metrics
        self.gc_freq = 5
        self.batch_train_losses = []

        self.initialize_weights()

    def _init_transformer(self, dim_model: int, nheads: int, dim_feedforward: int, 
                         nlayers: int, dropout: float, batch_first: bool) -> None:
        """Initialize transformer components."""
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
            dropout=dropout,
            layer_norm_eps=1e-12
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=nlayers,
            enable_nested_tensor=False
        )

    def _init_embeddings(self, n_tokens: int, dim_model: int, context_length: int,
                        learnable_pe: bool, dropout: float) -> None:
        """Initialize embedding layers."""
        self.embeddings = nn.Embedding(
            num_embeddings=n_tokens+5,
            embedding_dim=dim_model,
            padding_idx=1
        )

        if learnable_pe:
            self.positional_embedding = nn.Embedding(
                num_embeddings=context_length,
                embedding_dim=dim_model
            )
            self.dropout = nn.Dropout(p=dropout)
            self.register_buffer('pos', torch.arange(0, context_length, dtype=torch.long))
        else:
            self.positional_embedding = PositionalEncoding(
                d_model=dim_model,
                max_seq_len=context_length
            )

    def _init_heads(self, dim_model: int, n_tokens: int) -> None:
        """Initialize model heads."""
        self.classifier_head = nn.Linear(dim_model, n_tokens, bias=False)
        self.classifier_head.bias = nn.Parameter(torch.zeros(n_tokens))

        self.pooler_head = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()
        self.cls_head = nn.Linear(dim_model, 164)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the model."""
        token_embedding = self.embeddings(x)

        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device))
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        transformer_output = self.encoder(
            embeddings,
            is_causal=self.hparams.autoregressive,
            src_key_padding_mask=attention_mask
        )

        prediction = self.classifier_head(transformer_output)

        return {
            'mlm_prediction': prediction,
            'transformer_output': transformer_output
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        with torch.no_grad():
            batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)

        masked_indices = batch['masked_indices']
        mask = batch['mask']
        real_indices = batch['X']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        mlm_predictions = predictions['mlm_prediction']

        real_indices = self._prepare_target_indices(mask, real_indices)
        loss = self._compute_loss(mlm_predictions, real_indices)

        self.log('train_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute validation step."""
        with torch.no_grad():
            batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)

        masked_indices = batch['masked_indices']
        mask = batch['mask']
        real_indices = batch['X']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        mlm_predictions = predictions['mlm_prediction']

        real_indices = self._prepare_target_indices(mask, real_indices)
        loss = self._compute_loss(mlm_predictions, real_indices)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        return loss

    def _apply_masking(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply masking to input batch."""
        return complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)

    def _prepare_target_indices(self, mask: torch.Tensor, real_indices: torch.Tensor) -> torch.Tensor:
        """Prepare target indices for loss computation. Only masked indices are taken into account."""
        return torch.where(
            mask == MASK_TOKEN,
            real_indices,
            torch.tensor(-100, dtype=torch.long, device=real_indices.device)
        ).type(torch.int64)

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for training/validation."""
        predictions = predictions.view(-1, self.hparams.n_tokens)
        targets = targets.view(-1)

        if self.hparams.masking_p == 0.0:
            return torch.tensor(0.0, device=predictions.device)
        return self.loss(predictions, targets)

    def _apply_positional_encoding(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to token embeddings."""
        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device))
            return self.dropout(token_embedding + pos_embedding)
        return self.positional_embedding(token_embedding)


    def on_before_batch_transfer(self, batch: tuple, dataloader_idx: int) -> dict[str, torch.Tensor]:
        """Process batch before device transfer."""
        if isinstance(batch, tuple):
            batch, _ = batch
        return batch

    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], 
                              dataloader_idx: int) -> dict[str, torch.Tensor]:
        """Process batch after device transfer."""
        data_key = 'X'
        x = batch[data_key]

        if self.hparams.modality and 'modality' in batch:
            x = torch.cat((batch['modality'].reshape(-1, 1), x), dim=1)

        if self.hparams.assay and 'assay' in batch:
            x = torch.cat((batch['assay'].reshape(-1, 1), x), dim=1)

        if self.hparams.specie and 'specie' in batch:
            x = torch.cat((batch['specie'].reshape(-1, 1), x), dim=1)

        if self.hparams.supervised_task:
            if 'cell_type' in batch:
                batch['label'] = batch['cell_type']
            if 'X_niche' in batch:
                batch['label'] = batch['X_niche']

        batch[data_key] = x[:, :self.hparams.context_length]
        return batch

    def configure_optimizers(self) -> tuple:
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams.warmup,
            max_epochs=self.hparams.max_epochs
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def get_embeddings(self, batch: dict[str, torch.Tensor], layer: int = -1, with_context: bool = False) -> torch.Tensor:
        """Get embeddings from the model."""
        batch = complete_masking(batch, 0.0, self.hparams.n_tokens + 5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']

        # Get token embeddings and positional encodings
        token_embedding = self.embeddings(masked_indices)

        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device))
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        # Process through transformer layers up to desired layer
        if layer < 0:
            layer = len(self.encoder.layers) + layer # -1 means last layer

        for i in range(layer + 1):
            embeddings = self.encoder.layers[i](
                embeddings,
                src_key_padding_mask=attention_mask,
                is_causal=False
            )

        if not with_context:
            embeddings = embeddings[:, 3:, :]

        embeddings = embeddings.mean(dim=1)

        return embeddings


class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)

        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.encoding[:, :x.size(1)]


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with cosine annealing and warmup."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup: int, max_epochs: int):
        self.warmup = warmup
        self.max_num_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get learning rates for all parameter groups."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(1e-5, base_lr * lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """Calculate learning rate factor based on epoch."""
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
