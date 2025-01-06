from models._nicheformer import Nicheformer
from data.datamodules import MerlinDataModuleDistributed
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    
def manual_train_fm(config=None):
    
    pl.seed_everything(42)
    
    model = Nicheformer(dim_model=config['dim_model'], 
                        nheads=config['nheads'], 
                        dim_feedforward=config['dim_feedforward'], 
                        nlayers=config['nlayers'],
                        dropout=config['dropout'],
                        batch_first=config['batch_first'], 
                        masking_p=config['masking_p'], 
                        n_tokens=config['n_tokens'],
                        context_length=config['context_length'],
                        warmup=config['warmup'],
                        lr=config['lr'],
                        batch_size=config['batch_size'],
                        max_epochs=config['max_epochs'],
                        autoregressive=config['autoregressive'],
                        pool=config['pool'],
                        supervised_task=config['supervised_task'],
                        learnable_pe=config['learnable_pe'],
                        specie=config['specie'],
                        assay=config['assay'],
                        modality=config['modality'],
                        contrastive=config['contrastive'])
        
    if config['pretrained_path'] is not None:
        print("Loading pretrained model")
        model = Nicheformer.load_from_checkpoint(checkpoint_path=config['pretrained_path'])
    
    wandb_logger = WandbLogger(project=f'FM-{config["organ"]}', entity='nicheformer')
    
    checkpoint_callback = ModelCheckpoint(dirpath=f'/pretrained', every_n_train_steps=1500, monitor='train_loss', save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
                        logger=wandb_logger,
                        accelerator='gpu',
                        max_epochs=1000,
                        devices=-1,
                        log_every_n_steps=100,
                        check_val_every_n_epoch=50,
                        strategy="ddp_find_unused_parameters_true",
                        default_root_dir=f'/home/icb/alejandro.tejada/spatial-transformer/trained_model_heads_{config["nheads"]}_blocks_{config["nlayers"]}/',
                        callbacks=[checkpoint_callback, lr_monitor],
                        precision='bf16-mixed',
                        gradient_clip_val=1,
                        accumulate_grad_batches=10)
    
    
    path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/cellxgene_census_tokenized'
    key_organ = 'cell_type'
    splits=True
  
    if config['organ'] == 'everything':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_tokens'
        key_organ = ['X', 'specie', 'assay', 'modality']
        splits = True
    
    print(f"Using path {path_organ}")
    
    module = MerlinDataModuleDistributed(path=path_organ, 
                        columns=key_organ,
                        batch_size=config['batch_size'],
                        world_size=trainer.world_size,
                        splits=splits)
    
    if config['pretrained_path'] is not None and config['retake_training']:
        print(f"Training model in {config['organ']} from checkpoint!")
        trainer.fit(model=model, datamodule=module, ckpt_path=config['pretrained_path'])
        
    print(f"Training model in {config['organ']} from scratch")
    trainer.fit(model=model, datamodule=module)
