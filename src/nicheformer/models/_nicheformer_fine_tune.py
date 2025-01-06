from typing import List, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from torch import Tensor
from torch import optim

import gc
import numpy as np

from ._utils import complete_masking
from ._nicheformer import CosineWarmupScheduler

CLS_TOKEN = 2

class NicheformerFineTune(pl.LightningModule):
    def __init__(
        self,
        backbone: pl.LightningModule,
        supervised_task: str,
        extract_layers: list[int],
        function_layers: str,
        lr: float,
        warmup: int,
        max_epochs: int,
        dim_prediction: int = 33,
        n_classes: int = 60,
        freeze: bool = False,
        reinit_layers: Optional[list[int]] = None,
        extractor: bool = False,
        regress_distribution: bool = False,
        pool: str = "mean",
        predict_density: bool = False,
        ignore_zeros: bool = False,
        organ: str = "brain",
        label: str = "X_niche_0",
        without_context: bool = True,
    ):
        """Fine-tuning model for various downstream tasks.

        Args:
            backbone: Pre-trained transformer model
            supervised_task: Type of task ('niche_regression', 'density_regression', etc.)
            extract_layers: Indices of layers to extract features from
            function_layers: How to combine extracted layers ("mean", "sum", "concat")
            lr: Learning rate
            warmup: Number of warmup steps
            max_epochs: Total training epochs
            dim_prediction: Output dimension for predictions
            n_classes: Number of classes for classification tasks
            freeze: If True, freeze backbone parameters
            reinit_layers: Layers to reinitialize
            extractor: If True, use additional MLP before final layer
            regress_distribution: If True, predict probability distribution
            pool: Pooling strategy ("mean", "cls", None)
            predict_density: If True, predict scalar cell density
            ignore_zeros: If True, ignore zero labels in classification
            organ: Tissue type (for logging)
            label: Target column name
            without_context: If True, exclude context tokens
        """
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        # Initialize backbone
        self.backbone = backbone
        self.backbone.hparams.masking_p = 0.0 # not MLM loss anymore
        if freeze:
            self.backbone.freeze()

        if reinit_layers:
            for layer in self.backbone.encoder.layers:
                print(f"Reinit layer {layer}")
                self.backbone.initialize_weights(layer)

        # Setup task-specific components
        self.task = supervised_task.split('_')[-1]
        self._setup_task_components(supervised_task)

        # Optional MLP extractor
        if extractor:
            self.pooler_head = nn.Linear(
                self.backbone.hparams.dim_model, 
                self.backbone.hparams.dim_model
            )
            self.activation = nn.Tanh() # as in HuggingFace

        self.gc_freq = 10
        self.batch_train_losses = []

    def _setup_task_components(self, supervised_task: str) -> None:
        """Initialize task-specific layers and loss functions."""
        dim_model = self.backbone.hparams.dim_model

        if supervised_task == 'niche_regression':
            self.linear_head = nn.Linear(dim_model, self.hparams.dim_prediction, bias=False)
            self.softmax = nn.Softmax(dim=1)
            self.cls_loss = nn.MSELoss()
        elif supervised_task == 'density_regression':
            self.linear_head = nn.Linear(dim_model, 1, bias=False)
            self.cls_loss = nn.MSELoss()
        elif supervised_task == 'niche_classification':
            self.linear_head = nn.Linear(dim_model, self.hparams.n_classes, bias=False)
            self.cls_loss = nn.CrossEntropyLoss()
        elif supervised_task == 'niche_binary_classification':
            self.linear_head = nn.Linear(dim_model, self.hparams.dim_prediction, bias=False)
            self.cls_loss = nn.CrossEntropyLoss()
        elif supervised_task == 'niche_multiclass_classification':
            self.linear_head = nn.Linear(
                dim_model,
                self.hparams.dim_prediction * self.hparams.n_classes, 
                bias=False
            )
            self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """Forward pass through the model."""
        # Get embeddings
        token_embedding = self.backbone.embeddings(x)

        # Add positional embeddings
        if self.backbone.hparams.learnable_pe:
            pos_embedding = self.backbone.positional_embedding(
                self.backbone.pos.to(token_embedding.device)
            )
            embeddings = self.backbone.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.backbone.positional_embedding(token_embedding)

        # Process through transformer layers
        hidden_repr = []
        for i, layer in enumerate(self.backbone.encoder.layers):
            embeddings = layer(
                embeddings, 
                is_causal=self.backbone.autoregressive,
                src_key_padding_mask=attention_mask
            )
            if i in self.hparams.extract_layers:
                hidden_repr.append(embeddings)

        # Combine hidden states
        if self.hparams.function_layers == "mean":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.mean(combined_tensor, dim=-1)
        elif self.hparams.function_layers == "sum":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.sum(combined_tensor, dim=-1)
        elif self.hparams.function_layers == "concat":
            transformer_output = torch.cat(hidden_repr, dim=2)

        # Get predictions
        if self.hparams.extractor:
            if self.hparams.without_context:
                cls_prediction = self.pooler_head(transformer_output[:, 3:, :].mean(1))
            else:
                cls_prediction = self.pooler_head(transformer_output.mean(1))
            cls_prediction = self.activation(cls_prediction)
            cls_prediction = self.linear_head(cls_prediction)
        else:
            if self.hparams.without_context:
                cls_prediction = self.linear_head(transformer_output[:, 3:, :].mean(1))
            else:
                cls_prediction = self.linear_head(transformer_output.mean(1))

        if self.hparams.regress_distribution:
            cls_prediction = self.softmax(cls_prediction)

        return {
            'cls_prediction': cls_prediction, # CLS is actually mean, there is no difference
            'representation': transformer_output[:, 3:, :].mean(1)
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute training step."""
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens + 5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        label = batch['label']
        cls_prediction = predictions['cls_prediction']

        if self.task == 'regression':
            if self.hparams.regress_distribution:
                label = self.softmax(label)

            if self.hparams.predict_density:
                label = batch['label'].sum(1)
                label = label.view(-1, 1)

            loss = self.cls_loss(cls_prediction, label)
            self.log('train/regression_loss', loss, sync_dist=True, prog_bar=True)

        if self.task == 'classification':
            cls_prediction = cls_prediction.view(-1, self.hparams.n_classes, self.hparams.dim_prediction)

            if self.hparams.supervised_task == 'niche_binary_classification':
                label = (label != 0).int()

            if self.hparams.ignore_zeros:
                label = torch.where(label == 0, torch.tensor(-100, dtype=torch.long), label)

            if self.hparams.supervised_task == 'niche_classification':
                label = label.view(-1, 1)

            loss = self.cls_loss(cls_prediction, label.long())
            self.log('train/classification_loss', loss, sync_dist=True)

            accuracy_pred = torch.argmax(cls_prediction, dim=1)
            acc = (accuracy_pred == label).float().mean()
            self.log('train/accuracy', acc, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute validation step."""
        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens + 5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        cls_prediction = predictions['cls_prediction']
        label = batch['label']

        if self.task == 'regression':

            if self.hparams.regress_distribution:
                label = self.softmax(label) # normalize to sum 1

            if self.hparams.predict_density:
                label = batch['label'].sum(1)
                label = label.view(-1, 1)

            loss = self.cls_loss(cls_prediction, label)
            self.log('val/regression_loss', loss, sync_dist=True)

        if self.task == 'classification':
            cls_prediction = cls_prediction.view(-1, self.hparams.n_classes, self.hparams.dim_prediction)

            if self.hparams.supervised_task == 'niche_binary_classification':
                label = (label != 0).int()

            if self.hparams.ignore_zeros:
                label = torch.where(label == 0, torch.tensor(-100, dtype=torch.long), label)

            if self.hparams.supervised_task == 'niche_classification':
                label = label.view(-1, 1)

            loss = self.cls_loss(cls_prediction, label.long())
            self.log('val/classification_loss', loss, sync_dist=True)

            accuracy_pred = torch.argmax(cls_prediction, dim=1)
            acc = (accuracy_pred == label).float().mean()
            self.log('val/accuracy', acc, sync_dist=True)

        return loss

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute test step."""
        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens + 5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        cls_prediction = predictions['cls_prediction']
        label = batch['label']

        if self.task == 'regression':
            if self.hparams.regress_distribution:
                label = self.softmax(label)

            if self.hparams.predict_density:
                label = batch['label'].sum(1)
                label = label.view(-1, 1)

            loss = self.cls_loss(cls_prediction, label)
            self.log('test/regression_loss', loss, sync_dist=True)

        if self.task == 'classification':
            cls_prediction = cls_prediction.view(-1, self.hparams.n_classes, self.hparams.dim_prediction)

            if self.hparams.supervised_task == 'niche_binary_classification':
                label = (label != 0).int()

            if self.hparams.ignore_zeros:
                label = torch.where(label == 0, torch.tensor(-100, dtype=torch.long), label)

            if self.hparams.supervised_task == 'niche_classification':
                label = label.view(-1, 1)

            loss = self.cls_loss(cls_prediction, label.long())
            self.log('test/classification_loss', loss, sync_dist=True)

            accuracy_pred = torch.argmax(cls_prediction, dim=1)
            acc = (accuracy_pred == label).float().mean()
            self.log('test/accuracy', acc, sync_dist=True)

        return loss

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute prediction step."""
        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens + 5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        cls_prediction = predictions['cls_prediction']

        if self.task == 'classification':
            cls_prediction = cls_prediction.view(-1, self.hparams.n_classes, self.hparams.dim_prediction)
            predictions = torch.argmax(cls_prediction, dim=1)
        else:
            predictions = cls_prediction

        return predictions, cls_prediction

    def get_embeddings(self, batch: dict[str, Tensor], layer: int = -1) -> Tensor:
        """Get embeddings from the model."""
        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens + 5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']

        # Get token embeddings and positional encodings
        token_embedding = self.backbone.embeddings(masked_indices)

        if self.backbone.hparams.learnable_pe:
            pos_embedding = self.backbone.positional_embedding(self.backbone.pos.to(token_embedding.device))
            embeddings = self.backbone.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.backbone.positional_embedding(token_embedding)

        # Process through transformer layers up to desired layer
        if layer < 0:
            layer = len(self.backbone.encoder.layers) + layer # -1 means last layer

        for i in range(layer + 1):
            embeddings = self.backbone.encoder.layers[i](
                embeddings,
                src_key_padding_mask=attention_mask,
                is_causal=self.backbone.autoregressive
            )

        return embeddings

    def on_after_batch_transfer(self, batch: tuple[dict[str, Tensor], any], dataloader_idx: int) -> dict[str, Tensor]:
        """Process batch after transfer to device."""
        if isinstance(batch, tuple):
            batch, _ = batch

        data_key = 'X'

        # Add auxiliary tokens
        if self.backbone.hparams.modality:
            x = batch[data_key]
            modality = batch['modality']
            x = torch.cat((modality.reshape(-1, 1), x), dim=1)
            batch[data_key] = x

        if self.backbone.hparams.assay:
            x = batch[data_key]
            assay = batch['assay']
            x = torch.cat((assay.reshape(-1, 1), x), dim=1)
            batch[data_key] = x

        if self.backbone.hparams.specie:
            x = batch[data_key]
            specie = batch['specie']
            x = torch.cat((specie.reshape(-1, 1), x), dim=1)
            batch[data_key] = x

        # Add label to predict to the btch
        if self.hparams.label in batch.keys():
            batch['label'] = batch[self.hparams.label].to(torch.float32)
        else:
            raise KeyError(f"Label '{self.hparams.label}' not found in batch keys")

        if self.hparams.pool == 'cls':
            x = batch[data_key]
            cls = torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device) * CLS_TOKEN
            x = torch.cat((cls, x), dim=1)
            batch[data_key] = x

        batch['X'] = batch['X'][:, :self.backbone.hparams.context_length]
        return batch

    def configure_optimizers(self) -> tuple[list[optim.AdamW], list[dict]]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.001
        )

        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams.warmup,
            max_epochs=self.hparams.max_epochs
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
