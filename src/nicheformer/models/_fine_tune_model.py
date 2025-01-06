import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List
from torch import optim
from ._utils import complete_masking
from ._nicheformer import CosineWarmupScheduler
from pyro.distributions import DirichletMultinomial
import gc
import numpy as np

CLS_TOKEN = 2

class FineTuningModel(pl.LightningModule):
    
    def __init__(self, 
                 backbone: pl.LightningModule,
                 baseline: bool, 
                 freeze: bool, 
                 extract_layers: List[int],
                 reinit_layers: List[int],
                 function_layers: str,
                 extractor: bool,
                 lr: float, 
                 warmup: int, 
                 max_epochs: int,
                 supervised_task: str = 'regression',
                 regress_distribution: bool = False,
                 pool: int = 'mean',
                 dim_prediction: int = 33,
                 n_classes: int = 60,
                 predict_density: bool = False,
                 ignore_zeros: bool = False,
                 organ: str = 'brain',
                 label: str = 'X_niche_0',
                 predict: bool = False,
                 without_context: bool = True,
                 ):
        """
        Args:
            backbone (pl.LightningModule): pretrained model
            baseline (bool): just for wandb logger to know it's baseline; baseline here means non-trained Transformer
            freeze (bool): to freeze backbone or not to freeze backbone (that is the question)
            extract_layers (int): which hidden representations use as input for the linear layer
            reinit_layers (int): which layers reinitialize
            function_layers (str): which function use to combine the hidden representations used
            extractor(bool): use extractor before linear layer or not
            lr (float): learning rate
            warmup (int): number of steps that the warmup takes
            max_epochs (int): number of steps until the learning rate reaches 0
            supervised_task (int): None, 'classification' or 'regression'
            regress_distribution (bool): if True, regression over the density distribution. Not used in classification
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token that gathers info of the sequence, mean just averages all tokens
            dim_prediction (int): dimensionality of the vector to predict
            n_classes (int): number of classes to classify
            predict_density (bool): if True, we predict the number of cells surrounding the target cell, just an scalar
            ignore_zeros (bool): if True, ignore the zeros in the classification
            organ (str): just for wandb logger
            label (str): column to predict
        """
        super().__init__()
        self.backbone = backbone
        self.backbone.hparams.masking_p = 0.0
        if freeze:
            self.backbone.freeze() # Freeze backbone
            
        if reinit_layers:
            for layer in self.backbone.encoder.layers:
                print(f"Reinit layer {layer}")
                self.backbone.initialize_weights(layer)
             
            
        # Identify task:
        task_strings = supervised_task.split('_')
        self.task = task_strings[-1] # just makes easier selection 
        
        if supervised_task == 'niche_regression': # we always regress the niches (possibly change in the future)
            self.linear_head = nn.Linear(self.backbone.hparams.dim_model, dim_prediction, bias=False) # we predict a value per cell type
            self.softmax = nn.Softmax(dim=1)
            self.cls_loss = nn.MSELoss()
        elif supervised_task == 'density_regression':
            self.linear_head = nn.Linear(self.backbone.hparams.dim_model, 1, bias=False) # predict single value
            self.cls_loss = nn.MSELoss()
        elif supervised_task == 'niche_classification':
            self.linear_head = nn.Linear(self.backbone.hparams.dim_model, n_classes, bias=False) # predict logit per each niche type
            self.cls_loss = nn.CrossEntropyLoss()
        elif supervised_task == 'niche_binary_classification':
            self.linear_head = nn.Linear(self.backbone.hparams.dim_model, dim_prediction, bias=False) # predict logit per each cell type
            self.cls_loss = nn.CrossEntropyLoss()
        elif supervised_task == 'niche_multiclass_classification':
            self.linear_head = nn.Linear(self.backbone.hparams.dim_model, dim_prediction*n_classes, bias=False) # predict max_n_neigh per cell type
            self.cls_loss = nn.CrossEntropyLoss()
        # This is how it's done in HuggingFace
        if extractor:
            self.pooler_head = nn.Linear(self.backbone.hparams.dim_model, self.backbone.hparams.dim_model)
            self.activation = nn.Tanh()

        self.save_hyperparameters(ignore=['backbone'])
        self.gc_freq = 10
        
        self.batch_train_losses = []
        
        print("Backbone")
        print(self.linear_head)
        
    def forward(self, x, attention_mask):
        
        # x -> size: batch x (context_length) x 1

        token_embedding = self.backbone.embeddings(x)

        if self.backbone.hparams.learnable_pe:
            pos_embedding = self.backbone.positional_embedding(self.backbone.pos.to(token_embedding.device))
            embeddings = self.backbone.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.backbone.positional_embedding(token_embedding)

        hidden_repr = []

        for i in range(len(self.backbone.encoder.layers)):
            layer = self.backbone.encoder.layers[i]
            embeddings = layer(embeddings, is_causal=self.backbone.autoregressive, src_key_padding_mask=attention_mask) # bs x seq_len x dim
            if i in self.hparams.extract_layers:
                hidden_repr.append(embeddings)

        if self.hparams.function_layers == "mean":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.mean(combined_tensor, dim=-1)  # bs x seq_len x dim
        if self.hparams.function_layers == "sum":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.sum(combined_tensor, dim=-1)  # bs x seq_len x dim
            
        if self.hparams.function_layers == "concat":
            transformer_output = torch.cat(hidden_repr, dim=2)
                        
        # backbone_output = self.backbone(x, attention_mask)
        # transformer_output = backbone_output['transformer_output']
        
        if self.hparams.extractor:
            # Pooler MLP
            if self.hparams.without_context:
                cls_prediction = self.pooler_head(transformer_output[:, 3:, :].mean(1))
            else:
                cls_prediction = self.pooler_head(transformer_output.mean(1))
            cls_prediction = self.activation(cls_prediction)
        
            # Linear layer on top
            cls_prediction = self.linear_head(cls_prediction)
        
        else:
            if self.hparams.without_context:
                cls_prediction = self.linear_head(transformer_output[:, 3:, :].mean(1))
            else:
                cls_prediction = self.linear_head(transformer_output.mean(1))

        if self.hparams.regress_distribution:
            cls_prediction = self.softmax(cls_prediction)
                
            return {'cls_prediction': cls_prediction,
                    'representation': transformer_output[:, 3:, :].mean(1)}
            
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        
        if batch_idx % self.gc_freq == 0:
            gc.collect()
            
        # masking is 0.0 cause we don't use SSL again
        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens+5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        
        label = batch['label']
        cls_prediction = predictions['cls_prediction']

        if self.task == 'regression':
            
            if self.hparams.regress_distribution:
                label = self.softmax(label) # if regress distribution, the label should be a distribution
                
            if self.hparams.predict_density:
                label = batch['label'].sum(1)
                label = label.view(-1,1)
                
            loss = self.cls_loss(cls_prediction, label) 
            self.log('fine_tuning_regression_train', loss, sync_dist=True, prog_bar=True)

            # r2_scores = r2_score(label.detach().cpu().numpy(), cls_prediction.detach().cpu().numpy())
            # self.log('r2_score_training', r2_scores, sync_dist=True)

        if self.task == 'classification':
            
            cls_prediction = cls_prediction.view(-1, self.hparams.n_classes, self.hparams.dim_prediction) # reshape prediction to (bs x n_classes x n_classifications)
            
            if self.hparams.supervised_task == 'niche_binary_classification':
                label = (label != 0).int() # if binary, transform label to just 2 values, 0 and 1
                
            if self.hparams.ignore_zeros:
                label = torch.where(label == 0, torch.tensor(-100, dtype=torch.long), label) 
                
            if self.hparams.supervised_task == 'niche_classification':
                label = label.view(-1, 1) # reshape needed
                
            loss = self.cls_loss(cls_prediction, label.long())
            self.log('fine_tuning_classification_train', loss, sync_dist=True)
            
            accuracy_pred = torch.argmax(cls_prediction, dim=1)
            acc = (accuracy_pred == label).float().mean()
            self.log('accuracy_train', acc, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        
        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens+5)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']
        
        predictions = self.forward(masked_indices, attention_mask)
        
        cls_prediction = predictions['cls_prediction'] 
        label = batch['label']
        
        if self.task == 'regression':
            if self.hparams.predict:
                return 1 
            if self.hparams.regress_distribution:
                label = self.softmax(label)
                
            if self.hparams.predict_density:
                label = batch['label'].sum(1)
                label = label.view(-1,1)
            
            loss = self.cls_loss(cls_prediction, label)   
            self.log('fine_tuning_regression_validation', loss, sync_dist=True)

        if self.task == 'classification':

            cls_prediction = cls_prediction.view(-1, self.hparams.n_classes, self.hparams.dim_prediction) # N x C x K
            
            if self.hparams.supervised_task == 'niche_binary_classification':
                label = (label != 0).int()
                
            if self.hparams.ignore_zeros:
                label = torch.where(label == 0, torch.tensor(-100, dtype=torch.long), label)
                
            if self.hparams.supervised_task == 'niche_classification':
                label = label.view(-1, 1) # N x K

            loss = self.cls_loss(cls_prediction, label.long())
            self.log('fine_tuning_classification_validation', loss, sync_dist=True)
            accuracy_pred = torch.argmax(cls_prediction, dim=1)
            acc = (accuracy_pred == label).float().mean()
            self.log('accuracy_validation', acc, sync_dist=True)
         
        return loss
    
    def getting_prediction(self, batch, batch_idx, transformer_output=False,*args, **kwargs):
        """We get simply the predictions, which are the own logits"""

        batch = complete_masking(batch, 0.0, self.backbone.hparams.n_tokens+5)
        masked_indices = batch['masked_indices'].to(self.backbone.device)
        attention_mask = batch['attention_mask'].to(self.backbone.device)

        predictions = self.forward(masked_indices, attention_mask)
        cls_prediction = predictions['cls_prediction']

        if transformer_output:
            return predictions['representation']

        if self.task == 'classification':
            cls_prediction = cls_prediction

        if self.task == 'regression':
            cls_prediction = cls_prediction

        return cls_prediction

    def on_after_batch_transfer(self, batch, dataloader_idx: int):

        batch, _ = batch

        data_key = 'X'

       # Add auxiliar tokens
        if self.backbone.hparams.modality:
            x = batch[data_key]
            modality = batch['modality']
            x = torch.cat((modality.reshape(-1, 1), x), dim=1) # add modality token
            batch[data_key] = x

        if self.backbone.hparams.assay:
            x = batch[data_key]
            assay = batch['assay']
            x = torch.cat((assay.reshape(-1, 1), x), dim=1) # add assay token
            batch[data_key] = x

        if self.backbone.hparams.specie:
            x = batch[data_key]
            specie = batch['specie']
            x = torch.cat((specie.reshape(-1, 1), x), dim=1) # add organism token
            batch[data_key] = x

        # Add label to predict
        if self.hparams.label in batch.keys():
            batch['label'] = batch[self.hparams.label].to(torch.float32)
        elif not self.hparams.predict:
            raise NotImplementedError("Label specified not existent in parquet or model.")
        else:
            batch['label'] = batch['specie'] # whatever to label, it's not used

        if self.hparams.pool == 'cls': # Add cls token at the beginning of the set
            x = batch[data_key]
            cls = torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device)*CLS_TOKEN # CLS token is index 2
            x = torch.cat((cls, x), dim=1) # add CLS
            batch[data_key] = x

        batch['X'] = batch['X'][:, :self.backbone.hparams.context_length]

        return batch

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.001)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_epochs=self.hparams.max_epochs)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def initialize_weights(self):

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0.0, std=0.02)
