config_setting='--config_file /home/zjh/.cache/huggingface/accelerate/1gpus_idx0.yaml'

exp_prefix='ICL'
dataset=concept_1k_task1
batch_size=32

wandb_setting='--is_wandb True --wandb_project Concept-1K --wandb_entity your-name'

for i in $(seq 1 3);
do
    echo 'Run '$i



    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    # ========================================================================================================================
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_same_instance.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_same_concept.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_rand.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_same_instance.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_same_concept.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_rand.yaml' --backbone $backbone --batch_size $batch_size
    



    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    # ========================================================================================================================
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_same_instance.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_same_concept.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_rand.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_same_instance.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_same_concept.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_rand.yaml' --backbone $backbone --batch_size $batch_size
    



    # ========================================================================================================================
    backbone='EleutherAI/pythia-70m-deduped'
    # ========================================================================================================================
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_same_instance.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_same_concept.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_1shot_rand.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_same_instance.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_same_concept.yaml' --backbone $backbone --batch_size $batch_size
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone --cfg './config/IIL/concept_1k_task1/ICL_5shot_rand.yaml' --backbone $backbone --batch_size $batch_size
    

    
done
