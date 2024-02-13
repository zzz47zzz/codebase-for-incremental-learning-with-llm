config_setting='--config_file your-accelerate-config-file.yaml'

exp_prefix='SOTACILIntent'

replay_setting='--is_replay True --Replay_buffer_size 77'

for i in $(seq 1 3);
do
    echo 'Run '$i


    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_full.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/DERpp.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/CLSER.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_full.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/DERpp.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/CLSER.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1



    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/banking77_task7/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




done