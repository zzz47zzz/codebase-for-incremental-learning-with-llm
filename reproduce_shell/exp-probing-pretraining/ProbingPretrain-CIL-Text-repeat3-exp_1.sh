config_setting='--config_file your-accelerate-config-file.yaml'

exp_prefix='ExpProbPretrainCILText'


for i in $(seq 1 3);
do
    echo 'Run '$i

    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    backbone_random_init='True'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_random_init $backbone_random_init --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    backbone_random_init='True'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_random_init $backbone_random_init --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    backbone_random_init='True'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_random_init $backbone_random_init --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1




    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    backbone_revision='step16'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    backbone_revision='step16'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    backbone_revision='step16'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1




    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    backbone_revision='step128'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    backbone_revision='step128'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    backbone_revision='step128'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1




    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    backbone_revision='step1000'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    backbone_revision='step1000'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    backbone_revision='step1000'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1




    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    backbone_revision='step10000'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    backbone_revision='step10000'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    backbone_revision='step10000'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-TextClassification --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    # accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/topic3datasets_task5/SEQ_full.yaml' --backbone $backbone --backbone_revision $backbone_revision --classifier $classifier --training_epochs 3 --evaluate_interval -1 --is_probing True --probing_n_metrics 1



done