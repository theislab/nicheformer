sweep_config = {
    'checkpoint_path': "/lustre/groups/ml01/projects/2023_nicheformer/pretrained_models/nicheformmer.ckpt",
    'freeze': False,
    'reinit_layers': None,
    'extractor': False,
    'batch_size': 12,
    'lr': 1e-4,
    'warmup': 1,
    'max_epochs': 50000,
    'pool': 'mean',
    'n_classes': 17,
    'dim_prediction': 1, 
    'supervised_task': 'niche_classification',
    'regress_distribution': False,
    'predict_density': False,
    'ignore_zeros': False,
    'baseline': False,
    'organ': 'brain',
    'label': 'region',
    }
