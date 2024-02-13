config_setting='--config_file your-accelerate-config-file.yaml'

exp_prefix='SOTA'
dataset=concept_1k_task10
batch_size=32


wandb_setting='--is_wandb True --wandb_project Concept-1K --wandb_entity your-name'

Replay_buffer_size=0

for i in $(seq 1 3);
do
    echo 'Run '$i

    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    # ========================================================================================================================
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task10/LAMOL_g.yaml' --backbone $backbone --training_epochs 10  --batch_size $batch_size 
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task10/LAMOL_t.yaml' --backbone $backbone --training_epochs 10  --batch_size $batch_size 
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task10/LAMOL_KD.yaml' --backbone $backbone --training_epochs 10  --batch_size $batch_size 
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task10/L2KD.yaml' --backbone $backbone --training_epochs 10  --batch_size $batch_size 
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task10/PCLL.yaml' --backbone $backbone --training_epochs 10  --batch_size $batch_size 
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task10/LFPT5.yaml' --backbone $backbone --training_epochs 10  --batch_size $batch_size 
    
done
