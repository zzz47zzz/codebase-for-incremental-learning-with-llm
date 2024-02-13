import os
import yaml
import argparse
import importlib

from models import METHOD_LIST

METHOD_IMPORT_LIST = {}
for _method_name in METHOD_LIST:
    try:
        METHOD_IMPORT_LIST[_method_name] = importlib.import_module('models.%s'%_method_name, package=_method_name)
    except Exception as e:
        print(e)
        print('Fail to import %s!'%(_method_name))

# ===========================================================================
# LOAD CONFIGURATIONS
def get_params():
    parser = argparse.ArgumentParser(description="Continual Learning for Pretrained Language Models")

    # =========================================================================================
    # Experimental Settings
    # =========================================================================================
    # Wandb
    parser.add_argument("--is_wandb", default=False, type=bool, help="if using wandb")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")

    # Config
    parser.add_argument("--cfg", default="./config/default_config.yaml", help="Hyper-parameters") # debug: "./config/default_config.yaml"

    # Path
    parser.add_argument("--wandb_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--exp_prefix", type=str, default="default", help="Prefix for experiment name")
    parser.add_argument("--logger_filename", type=str, default="train.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--save_ckpt", type=bool, default=False, help="If saving checkpoints")
    parser.add_argument("--save_probing_classifiers", type=bool, default=False, help="If saving probing classifiers")
    parser.add_argument("--save_features_before_after_IL", type=bool, default=False, help="If saving training features before and after IL")

    parser.add_argument("--seed", type=int, default=None, help="Random Seed")

    # Backbone
    parser.add_argument("--backbone", type=str, default="EleutherAI/pythia-70m-deduped", help="backbone name")
    parser.add_argument("--backbone_type", type=str, default="auto", choices=["auto","generative","discriminative"],
                         help="What is the type of the PLMs.\
                        For example, GPT is generative and BERT is discriminative. \
                        If 'auto' is selected, the backbone_type is set according to the backbone name which is pre-defined in backbone.py")
    parser.add_argument("--backbone_extract_token", type=str, default="last_token", choices=["cls_token","last_token"],help="Using last_token or cls_token as the feature.")
    parser.add_argument("--backbone_revision", type=str, default="", help="the revision of pretrained model")
    parser.add_argument("--backbone_cache_path", type=str, default="..", help="the path of storing pretrained model")
    parser.add_argument("--backbone_max_new_token", type=int, default=10, help="The max new tokens for generation")
    parser.add_argument("--backbone_random_init", default=False, type=bool, help="if randomly initialized model parameters")

    # Data
    parser.add_argument("--dataset", type=str, default="clinc150_task15", help="dataset name")
    parser.add_argument("--classification_type", type=str, default="auto", choices=["auto","sentence-level","word-level"], 
                        help="What is the classification type of the task? \
                        For example, Named Entity Recognition is word-level classification and Intent Classification is sentence-level classification. \
                        If 'auto' is selected, the classification_type is set according to the dataset name which is pre-defined in dataset.py")
    parser.add_argument("--prompt_type", type=str, default="auto", choices=["auto","default","none","relation_classification_qa","relation_classification_state","relation_classification_state_no_pos"], help="the prompt format")

    # =========================================================================================
    # Training Settings
    # =========================================================================================
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size") 
    parser.add_argument("--max_seq_length", type=int, default=-1, help="Max length for each sentence (default=-1 means that max_seq_length will be decided according to the dataset automatically)") 

    parser.add_argument("--is_probing", default=False, type=bool, help="If calculate probing performance for each evaluation")
    parser.add_argument("--probing_n_feature", default=1, type=int, help="Using the feature of next N tokens for linear probing")
    
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate") 
    parser.add_argument("--classifier_lr", type=float, default=1e-3, help="Initial learning rate") 
    parser.add_argument("--training_epochs", type=int, default=10, help="The number training epochs. NOTE: SEQ_warmup_epoch_before_fix_encoder and IPEFT_causallm_epoch are not included in training epochs.")

    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    parser.add_argument("--info_per_epochs", type=int, default=1, help="Print information every how many epochs")
    parser.add_argument("--info_per_steps", type=int, default=25, help="Print information every how many steps")
    parser.add_argument("--evaluate_interval", type=int, default=-1, help="Evaluation interval (default=-1 means that only evaluate at the end of each task)")
    parser.add_argument("--early_stop", type=int, default=-1, help="No improvement after several epoch, we stop training (defualt=-1 means no using early stopping)")

    # =========================================================================================
    # General Settings for Incremental Learning
    # =========================================================================================
    
    parser.add_argument("--il_mode", default="CIL", choices=["CIL","TIL","IIL"], help="the mode of incremental learning ( Class / Task / Instance Incremental Learning )")
    parser.add_argument("--method", type=str, default='SEQ', choices=METHOD_LIST, help="the selected baseline method to run")
    parser.add_argument("--classifier", type=str, default="None", choices=['None','CosineLinear','Linear'], help="Classifier")

    parser.add_argument("--is_replay", default=False, type=bool, help="if using experience replay")
    parser.add_argument("--Replay_buffer_size", type=int, default=100, help="The number of samples for data replay")
    parser.add_argument("--Replay_batch_level", type=bool, default=True, 
                        help="If Replay_batch_level = True, replay in each batch (we empirically find that it is effective);\
                              Otherwise, combine replay data in the current training set.")
    parser.add_argument("--Replay_fix_budge_each_class", type=bool, default=False, 
                        help="If storing the same number of samples for each class. \
                        For example, if we learn 10 tasks and each task contains 10 classes, and the buffer size is 100. \
                        When Replay_fix_budge_each_class=True, we only save 10 samples from the first task. \
                        When Replay_fix_budge_each_class=False (default), we save 100 samples from the first task \
                        and reduce the number of samples in the following tasks using Replay_sampling_algorithm.")
    parser.add_argument("--Replay_sampling_algorithm", type=str, default='random', choices=['random','herding'], help="The sampling algorithm for selecting old samples")


    params = parser.parse_args()

    # Add model-specific parameters
    method = params.method
    if params.cfg is not None and params.cfg!='None':
        with open(params.cfg) as f:
            config = yaml.safe_load(f)
            if 'method' in config.keys():
                method = config['method']
    assert method in METHOD_LIST, 'Not implemented for method %s'%(method)
    getattr(METHOD_IMPORT_LIST[method],'get_%s_params'%(method))(parser)
    params = parser.parse_args()

    # Set the pre-defined parameters in the yaml configuration
    # NOTE: the hyper-parameters in yaml files will replace those in the command line!
    params.__setattr__('method',method)
    if params.cfg is not None and params.cfg!='None':
        with open(params.cfg) as f:
            # Set wandb name automatically according to the filename of yaml configuration.
            params.__setattr__('wandb_name',params.exp_prefix+'-'+os.path.basename(params.cfg).split('.')[0])         
            config = yaml.safe_load(f)
            for k, v in config.items():
                params.__setattr__(k,v)

    return params
