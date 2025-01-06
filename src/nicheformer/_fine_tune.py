from models._nicheformer import Nicheformer
from models._fine_tune_model import FineTuningModel
from data.datamodules import MerlinDataModuleDistributed
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import numpy as np
from tqdm import tqdm
import os
    
def fine_tune_predictions(config=None):
    
    pl.seed_everything(42)
         
    without_context = True
    folder = "predictions"
    
    model = Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path'])

    fine_tune_model = FineTuningModel(backbone=model,
                                    freeze=config['freeze'],
                                    extract_layers=config['extract_layers'],
                                    function_layers=config['function_layers'],
                                    reinit_layers=config['reinit_layers'],
                                    extractor=config['extractor'],
                                    baseline=config['baseline'],
                                    warmup=config['warmup'],
                                    lr=config['lr'],
                                    max_epochs=config['max_epochs'],
                                    supervised_task=config['supervised_task'],
                                    regress_distribution=config['regress_distribution'],
                                    pool=config['pool'],
                                    dim_prediction=config['dim_prediction'],
                                    n_classes=config['n_classes'],
                                    predict_density=config['predict_density'],
                                    ignore_zeros=config['ignore_zeros'],
                                    organ=config['organ'],
                                    label=config['label'],
                                    without_context=without_context)
    
    run_name = config['supervised_task']
        
    run_name += f"_{config['checkpoint_path'].split('=')[2].split('.')[0]}_steps"
    
    if config['baseline']:
        run_name += "_baseline"
    
    wandb_logger = WandbLogger(project=f"Get_predictions_{config['organ']}", entity="nicheformer", name=run_name)
    
    checkpoint_callback = ModelCheckpoint(dirpath=f"/fine_tuned_models", every_n_train_steps=5000, monitor='fine_tuning_regression_train', save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
                        logger=wandb_logger,
                        accelerator='gpu',
                        max_epochs=1,
                        devices=-1,
                        check_val_every_n_epoch=None,
                        strategy="ddp_find_unused_parameters_true" if config['freeze']==False else "ddp",
                        callbacks=[checkpoint_callback, lr_monitor],
                        precision='bf16-mixed',
                        gradient_clip_val=1,
                        accumulate_grad_batches=10)
    
    path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/spatial_brain_tokens/'
    label_key = 'X_niche'
    
    if config['organ'] == 'healthy_liver':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/cosmx_healthy_liver'
        label_key = ['assay', 'specie', 'modality', 'niche', 'X', 'X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
        splits=True 
    if config['organ'] == 'liver':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream_capped/cosmx_liver'
        label_key = ['assay', 'specie', 'modality', 'niche', 'X', 'X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
        splits=True 
    elif config['organ'] == 'lung':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream_capped/cosmx_lung'
        label_key = ['assay', 'specie', 'modality', 'niche', 'X', 'X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
        splits=True 
    elif config['organ'] == 'brain':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_downstream/merfish_brain'
        label_key = ['assay', 'specie', 'modality',  'author_cell_type', 'niche', 'region', 'X', 'X_niche_1']#, 'density_0', 'density_1', 'density_2', 'density_3', 'density_4']
        splits=True 
    if config['organ'] == 'entire_mouse':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/dissociated_spatial_mouse_tokens_predictions'
        splits=True    
        
    
    print(f"Using path {path_organ}")
    
    module = MerlinDataModuleDistributed(path=path_organ, 
                        columns=label_key,
                        batch_size=config['batch_size'],
                        world_size=trainer.world_size,
                        splits=splits)
        
    trainer.fit(model=fine_tune_model, datamodule=module)
    
    dataloader_test = module.val_dataloader()
    dataloader_test.drop_last = False
    dataloader_test.shuffle = False
    
    #dataloader_test.batch_size = 1024

    print("Getting predictions")
    fine_tune_model = fine_tune_model.to('cuda')
    model.eval()
    
    predictions_final = torch.empty(size=(0, config['dim_prediction'] if config['supervised_task'] != 'niche_classification' else config['n_classes']))
    labels_final = torch.empty(size=(0, config['dim_prediction'] if config['supervised_task'] != 'density_regression' else 1))
    idx_final = np.empty(shape=(0,1))
    
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            batch = fine_tune_model.on_after_batch_transfer(batch, 2)
            prediction = fine_tune_model.getting_prediction(batch, 2)
            predictions_final = torch.cat((predictions_final, prediction.detach().cpu()))
            labels_final = torch.cat((labels_final, batch["label"].detach().cpu().unsqueeze(1) if batch["label"].detach().cpu().ndim == 1 else batch["label"].detach().cpu()))
            idx_final = np.concatenate((idx_final, batch['idx'].detach().cpu().unsqueeze(-1).numpy()))
                                        
    print("saving predictions")                        
    if not os.path.exists(f"/{folder}/"):
        # If the directory doesn't exist, create it
        os.makedirs(f"/{folder}/")

    np.save(f"/{folder}/{config['label']}_linear_{config['freeze']}.npy", predictions_final)
    np.save(f"/{folder}/label_{config['label']}_linear_{config['freeze']}.npy", labels_final)
    np.save(f"/{folder}/index_{config['label']}_linear_{config['freeze']}.npy", idx_final)
