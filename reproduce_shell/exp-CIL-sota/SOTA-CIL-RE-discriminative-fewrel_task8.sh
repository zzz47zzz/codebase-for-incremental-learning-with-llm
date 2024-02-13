config_setting='--config_file your-accelerate-config-file.yaml'

exp_prefix='SOTACILRE'

replay_setting='--is_replay True --Replay_buffer_size 80'

for i in $(seq 1 3);
do
    echo 'Run '$i


    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_full.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/DERpp.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/CLSER.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_full.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/DERpp.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/CLSER.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/fewrel_task8/SEQ_pre_warm_fix.yaml' $replay_setting --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




done