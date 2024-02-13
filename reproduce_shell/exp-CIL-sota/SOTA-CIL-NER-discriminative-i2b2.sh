config_setting='--config_file your-accelerate-config-file.yaml'

exp_prefix='SOTACILNER'


for i in $(seq 1 3);
do
    echo 'Run '$i
    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-NER --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/ExtendNER.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SelfTrain.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-NER --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-NER --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/ExtendNER.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SelfTrain.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-NER --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/i2b2_task5_base8_inc2/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




done