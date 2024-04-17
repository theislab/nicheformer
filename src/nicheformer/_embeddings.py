from models._nicheformer import Nicheformer
from models._fine_tune_model import FineTuningModel
from data.datamodules import MerlinDataModuleDistributed
import pytorch_lightning as pl
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
    
def get_embeddings_model(config=None):
    
    pl.seed_everything(42)
    
    without_context = True
    folder = "embeddings"
    
    model = FineTuningModel.load_from_checkpoint(checkpoint_path=config['fine_tuned_checkpoint_path'], backbone=Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path']), predict=True, without_context=without_context)
    
    trainer = pl.Trainer(
                        accelerator='gpu',
                        max_steps=0,
                        devices=1,
                        strategy="ddp_find_unused_parameters_true",
                        precision='bf16-mixed',)
    
    if config['organ'] == 'dissociated_merfish':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/dissociated_brain/scaled_merfish'
        label_key = ['assay', 'specie', 'modality', 'X', 'idx']
        splits=False 
    if config['organ'] == 'healthy_liver':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/cosmx_healthy_liver'
        label_key = ['assay', 'specie', 'modality', 'niche', 'X', 'X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
        splits=False
    if config['organ'] == 'dissociated_scaled':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/dissociated_brain/scaled_dissociated'
        label_key = ['assay', 'specie', 'modality', 'X', 'idx']
        splits=False 
    if config['organ'] == 'full_dissociated_merfish':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/dissociated_brain/full_scaled_merfish'
        label_key = ['assay', 'specie', 'modality', 'X', 'idx']
        splits=False 
    if config['organ'] == 'full_dissociated_scaled':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/dissociated_brain/full_scaled_dissociated'
        label_key = ['assay', 'specie', 'modality', 'X', 'idx']
        splits=False 
    if config['organ'] == 'brain':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/merfish_brain'
        label_key = ['assay', 'specie', 'modality', 'author_cell_type', 'region', 'niche', 'idx', 'X', 'X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
        splits=False 
    if config['organ'] == 'liver':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/cosmx_liver'
        label_key = ['assay', 'specie', 'modality', 'author_cell_type', 'region', 'niche', 'idx', 'X', 'X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
        splits=False 
    if config['organ'] == 'lung':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/cosmx_lung'
        label_key = ['assay', 'specie', 'modality', 'author_cell_type', 'region', 'niche', 'idx', 'X', 'X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
        splits=False 
    if config['organ'] == 'dissociated_brain':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_organ/dissociated_brain'
        label_key = ['assay', 'specie', 'modality', 'cell_type', 'X', 'idx']
        splits=False 
    if config['organ'] == 'dissociated_census':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_organ/dissociated_census'
        label_key = ['assay', 'specie', 'modality', 'cell_type', 'X', 'idx']
        splits=False 
    if config['organ'] == 'dissociated_mouse_brain':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_organ/dissociated_mouse_brain'
        label_key = ['assay', 'specie', 'modality', 'cell_type', 'X', 'idx']
        splits=False 
    if config['organ'] == 'everything':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_tokens'
        splits = False
        label_key = ['X', 'specie', 'assay', 'modality', 'idx']
    
    print(f"Using path {path_organ}")
    
    module = MerlinDataModuleDistributed(path=path_organ, 
                        columns=label_key,
                        batch_size=24,
                        world_size=trainer.world_size,
                        splits=splits)
        
    trainer.fit(model=model, datamodule=module)
        
    dataloader_test = module.val_dataloader()
    dataloader_test.drop_last = False
    dataloader_test.shuffle = False
    
    print("Getting predictions")
    model = model.to('cuda')
    model.eval()
    
    assay_l = np.empty(shape=(0,1))
    specie_l = np.empty(shape=(0,1))
    modality_l = np.empty(shape=(0,1))
    idx_final = np.empty(shape=(0,1))
    cell_type_l = np.empty(shape=(0,1))
    region_l = np.empty(shape=(0,1)) 
    niche_l = np.empty(shape=(0,1)) 
    representation_final = torch.empty(size=(0, 512))
    idx_final = np.empty(shape=(0,1))
    
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            batch = model.on_after_batch_transfer(batch, 2)
            representation = model.getting_prediction(batch, 2, transformer_output=True)
            representation_final = torch.cat((representation_final, representation.detach().cpu()))
                        
            if 'idx' in batch:
                idx_final = np.concatenate((idx_final, batch['idx'].detach().cpu().unsqueeze(-1).numpy()))
            if 'author_cell_type' in batch:
                cell_type_l = np.concatenate((cell_type_l, batch['author_cell_type'].detach().cpu().unsqueeze(-1).numpy()))
            if 'region' in batch:
                region_l = np.concatenate((region_l, batch['region'].detach().cpu().unsqueeze(-1).numpy()))
            if 'niche' in batch:
                niche_l = np.concatenate((niche_l, batch['niche'].detach().cpu().unsqueeze(-1).numpy()))
            if 'assay' in batch:
                assay_l = np.concatenate((assay_l, batch['assay'].detach().cpu().unsqueeze(-1).numpy()))
            if 'specie' in batch:
                specie_l = np.concatenate((specie_l, batch['specie'].detach().cpu().unsqueeze(-1).numpy()))
            if 'modality' in batch:
                modality_l = np.concatenate((modality_l, batch['modality'].detach().cpu().unsqueeze(-1).numpy()))
                
        metadata = {
                    #'development_stage': developmental_l.squeeze(),
                    'region': region_l.squeeze() if 'region' in batch else -1,
                    'niche': niche_l.squeeze() if 'niche' in batch else -1,
                    'cell_type': cell_type_l.squeeze() if 'author_cell_type' in batch else -1,
                    'assay': assay_l.squeeze(),
                    'specie_l': specie_l.squeeze(),
                    'modality': modality_l.squeeze(),
                    'index': idx_final.squeeze() if 'idx' in batch else -1
                }

        # Create a DataFrame from the dictionary
        metadata = pd.DataFrame(metadata)

        np.save(f"/{folder}/embeddings.npy", representation_final)
        metadata.to_csv(f"/{folder}/metadata_embeddings.csv")

        print(f"Saved in /{folder}")

