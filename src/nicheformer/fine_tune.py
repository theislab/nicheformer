import pprint
import wandb
from _fine_tune import fine_tune_predictions
#import _config as config
import config_files._config_fine_tune as config

if __name__ == "__main__":
        
    fine_tune_predictions(config=config.sweep_config)
    


    
    


    