import pprint
import wandb
from _embeddings import get_embeddings_organ
#import _config as config
#import config_files._config_embeddings_organs as config
import config_files._config_embeddings as config

if __name__ == "__main__":
        
    get_embeddings_organ(config=config.sweep_config)
    


    
    


    