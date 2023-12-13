import time
import numpy as np
import random
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size
# Specify CUDA_VISIBLE_DEVICES in the command, 
# e.g., CUDA_VISIBLE_DEVICES=0,1 nohup bash exp_on_b7server_0.sh
# ---------------------- only for debug -----------------------
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# -------------------------------------------------------------

from utils.logger import init_experiment
from utils.dataset import get_dataset
from utils.config import get_params
from utils.factory import get_model

def main_cl(params):

    # ------------------------------------------------------------------------------------------------------------------------=====
    
    # Initialize Accelerator
    accelerator = Accelerator(log_with="wandb")
    
    # Using Fixed Random Seed
    if params.seed:
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True

    # Initialize Experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info(params.__dict__)

    # Initialize wandb
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    # retry request (handles connection errors, timeouts)
    try_cnt = 0
    while try_cnt<5:
        try:
            accelerator.init_trackers(
                project_name=params.wandb_project,
                init_kwargs={
                    "wandb":{
                        'entity':params.wandb_entity, 
                        'name':params.wandb_name,
                        'config':vars(params), 
                        'settings':wandb.Settings(start_method="fork"),
                        # Disable wandb when debug
                        'mode': 'disabled' if 'default' in params.exp_prefix else 'online' if params.is_wandb else 'offline'
                    }
                }
            )
            # params.__setattr__('wandb_url', wandb.run.get_url() if params.is_wandb else '')
            break
        except Exception as e:
            print(str(e))
            print("Retrying Connecting wandb...")
            try_cnt+=1
            time.sleep(120)

    # Dataset
    CL_dataset = get_dataset(params)

    @find_executable_batch_size(starting_batch_size=params.batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator # Ensure they can be used in our context
        accelerator.free_memory() # Free all lingering references

        params.__setattr__('batch_size',batch_size)
        model = get_model(params, CL_dataset, accelerator)     
        model.incremental_training()
        model.finish_training()
        
    inner_training_loop()
        
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    params = get_params()
    main_cl(params)
