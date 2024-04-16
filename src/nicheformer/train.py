from _train import manual_train_fm
from models._utils import set_seed
#import _config as config
import config_files._config_train as config
import torch

if __name__ == "__main__":
        
    set_seed(42)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("No GPUs available.") 
    
    manual_train_fm(config=config.sweep_config)


    
    


    