config_setting='--config_file your-accelerate-config-file.yaml'

exp_prefix='ExpProbCILIntent'


for i in $(seq 1 3);
do
    echo 'Run '$i
    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_full.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True



    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_full.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_full.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True



    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_full.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='bert-base-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_full.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='Linear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_full.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True




    # ========================================================================================================================
    backbone='bert-large-cased'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-IntentClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_full.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/discriminative_backbones/clinc150_task15/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1 --is_probing True

done