import os
import torch
from adapters import AutoAdapterModel
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

from utils.prompt import get_auto_prompt_tuning_init_text

BACKBONE2TYPE = {
    'gpt2':'generative',
    'gpt2-large':'generative',
    'EleutherAI/pythia-70m-deduped':'generative',
    'EleutherAI/pythia-160m-deduped':'generative',
    'EleutherAI/pythia-410m-deduped':'generative',
    'EleutherAI/pythia-1b-deduped':'generative',
    'EleutherAI/pythia-1.4b-deduped':'generative',
    'EleutherAI/pythia-2.8b-deduped':'generative',
    'EleutherAI/pythia-2.8b-deduped':'generative',
    'decapoda-research/llama-7b-hf':'generative',
    'lmsys/vicuna-7b-v1.1':'generative',
    'llama2-13b-orca-8k-3319':'generative',
    'roberta-base':'discriminative',
    'roberta-large':'discriminative',
    'bert-base-cased':'discriminative',
    'bert-base-uncased':'discriminative',
    'bert-large-cased':'discriminative',
    'bert-large-uncased':'discriminative',
}

def get_backbone(params, num_task: int=1):
    '''
        Build model
    '''

    # Set backbone type
    if params.backbone_type == 'auto':
        assert params.backbone in BACKBONE2TYPE.keys(), 'Not Implemented for backbone %s'%(params.backbone)
        setattr(params,'backbone_type',BACKBONE2TYPE[params.backbone])
    else:
        assert params.backbone_type in ['generative','discriminative'], 'Invalid backbone type %s'%(params.backbone_type)
    
    # if params.backbone not in ['llama2-13b-orca-8k-3319']:
    try:
        config = AutoConfig.from_pretrained(params.backbone)
        config.return_dict = True
    except:
        config = AutoConfig.from_pretrained(os.path.join(params.backbone_cache_path,
                                                                params.backbone))
        config.return_dict = True

    if params.method == 'AdapterCL':
        model = AutoAdapterModel.from_pretrained(params.backbone, config=config)

    if params.method == 'CPFD':
        config.output_attentions = True

    # if params.backbone == 'llama2-13b-orca-8k-3319':
    #     model = AutoModelForCausalLM.from_pretrained("../llama2-13b-orca-8k-3319", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
    if params.backbone_revision is None or params.backbone_revision=='':
        try:
            model = AutoModelForCausalLM.from_pretrained(params.backbone, config=config)
        except:
            model = AutoModelForCausalLM.from_pretrained(os.path.join(params.backbone_cache_path,
                                                                    params.backbone),
                                                        config=config)
            
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(params.backbone, 
                                                        revision=params.backbone_revision,
                                                        cache_dir=os.path.join(params.backbone_cache_path,
                                                                os.path.join(os.path.basename(params.backbone),
                                                                params.backbone_revision)),
                                                        config=config)
        except:
            model = AutoModelForCausalLM.from_pretrained(os.path.join(params.backbone_cache_path,
                                                                os.path.join(os.path.basename(params.backbone),
                                                                params.backbone_revision)),
                                                        config=config)

    if params.backbone_random_init:
        model.apply(model._init_weights) # using apply() to init each submodule recursively

    if hasattr(params,'PEFT_type') and params.PEFT_type is not None and params.PEFT_type != 'None' :

        # Prompt Tuning
        if params.PEFT_type == 'PromptTuning':
            if params.PEFT_prompt_tuning_init_text is not None and params.PEFT_prompt_tuning_init_text!='':
                if params.PEFT_prompt_tuning_init_text=='auto':
                    prompt_tuning_init_text = get_auto_prompt_tuning_init_text(dataset=params.dataset)
                else:
                    prompt_tuning_init_text = params.PEFT_prompt_tuning_init_text
                peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, 
                                                inference_mode=False, 
                                                num_virtual_tokens=params.PEFT_num_virtual_tokens,
                                                prompt_tuning_init=PromptTuningInit.TEXT,
                                                prompt_tuning_init_text=prompt_tuning_init_text,
                                                token_dim=model.module.config.hidden_size if hasattr(model,'module') else model.config.hidden_size,
                                                tokenizer_name_or_path=params.backbone)
            else:
                peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, 
                                                inference_mode=False, 
                                                num_virtual_tokens=params.PEFT_num_virtual_tokens,
                                                token_dim=model.module.config.hidden_size if hasattr(model,'module') else model.config.hidden_size,
                                                tokenizer_name_or_path=params.backbone)

        # LoRA
        elif params.PEFT_type == 'LoRA':
            
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                     inference_mode=False, 
                                     target_modules=params.PEFT_lora_target_modules,
                                     r=params.PEFT_lora_r, 
                                     lora_alpha=params.PEFT_lora_alpha, 
                                     bias=params.PEFT_lora_bias,
                                     lora_dropout=params.PEFT_lora_dropout)
 
        else:
            raise NotImplementedError()
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if params.backbone == 'decapoda-research/llama-7b-hf':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(params.backbone, padding_side='left' if params.backbone_type == 'generative' else 'right')
        tokenizer.pad_token = '[PAD]'
        tokenizer.eos_token = '[PAD]'
    # elif params.backbone == 'llama2-13b-orca-8k-3319':
    #     tokenizer = AutoTokenizer.from_pretrained("../llama2-13b-orca-8k-3319", use_fast=False, padding_side='left' if params.backbone_type == 'generative' else 'right')
    elif params.backbone_revision is None or params.backbone_revision=='':
        try:
            tokenizer = AutoTokenizer.from_pretrained(params.backbone, 
                                                padding_side='left' if params.backbone_type == 'generative' else 'right')
        except:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(params.backbone_cache_path,
                                                                    params.backbone), 
                                                padding_side='left' if params.backbone_type == 'generative' else 'right')
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(params.backbone, padding_side='left' if params.backbone_type == 'generative' else 'right',
                                                            revision=params.backbone_revision,
                                                            cache_dir=os.path.join(params.backbone_cache_path,
                                                                os.path.join(os.path.basename(params.backbone),
                                                                params.backbone_revision)))
        except:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(params.backbone_cache_path,
                                                        os.path.join(os.path.basename(params.backbone),
                                                        params.backbone_revision)),
                                                    padding_side='left' if params.backbone_type == 'generative' else 'right')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def obtain_features(params, model, lm_input, tokenizer):
    '''
        Extract the last hidden state of lm_input

        Return:
            - extracted_feature: (batch_size, feature_dims)
    '''
    if params.backbone_type == 'generative':
        assert params.classification_type == 'sentence-level', 'Only implemented for sentence-level classification!'
        all_hidden_states = model.generate(**{
                            'input_ids':lm_input['input_ids'],
                            'attention_mask':lm_input['attention_mask']},
                            max_new_tokens=params.probing_n_feature, 
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True, 
                            output_hidden_states=True).hidden_states 
        assert params.backbone_extract_token == 'last_token', 'NotImplemented for backbone_extract_token %s'%(params.backbone_extract_token)
        # Concatenate probing_n_feature features
        num_new_token = len(all_hidden_states)
        extracted_feature = torch.concatenate([all_hidden_states[i][-1][:,-1,:].contiguous() for i in range(num_new_token)],dim=-1)  

    elif params.backbone_type == 'discriminative':
        all_hidden_states = model.forward(**{
                            'input_ids':lm_input['input_ids'],
                            'attention_mask':lm_input['attention_mask']},
                            output_hidden_states=True).hidden_states
        if params.classification_type == 'sentence-level':
            if params.backbone_extract_token == 'last_token':
                # last token feature
                last_token_idx = lm_input['attention_mask'].sum(dim=-1)-1 # (batch_size,)
                last_layer_states = all_hidden_states[-1] # (batch_size, seq_len, feature_dims)
                batch_size, seq_len, feature_dims = last_layer_states.shape
                last_token_idx = last_token_idx.reshape((-1,1,1)).repeat((1,1,feature_dims))
                extracted_feature = last_layer_states.gather(1,last_token_idx).squeeze().contiguous() # (batch_size, feature_dims)
            elif params.backbone_extract_token == 'cls_token':
                assert lm_input['input_ids'][0][0] == tokenizer.cls_token_id, 'Only impelmented for models whose cls_token is at the begining!'
                # cls token feature
                extracted_feature = all_hidden_states[-1][:,0,:].contiguous() 
            else:
                raise NotImplementedError()
        elif params.classification_type == 'word-level':
            last_layer_states = all_hidden_states[-1] # (batch_size, seq_len, feature_dims)
            extracted_feature = last_layer_states
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    return extracted_feature

def obtain_generate_ids(params, model, lm_input, tokenizer):
    '''
        Extract the new generated ids of lm_input

        Return:
            - extracted_feature: (batch_size, max_new_tokens)
    '''

    input_len = lm_input['input_ids'].shape[1]
    generate_ids_all = model.generate(**{
                                        'input_ids':lm_input['input_ids'], 
                                        'attention_mask':lm_input['attention_mask']}, 
                                        max_new_tokens=params.backbone_max_new_token, 
                                        pad_token_id=tokenizer.eos_token_id)
    
    generate_ids = generate_ids_all[:,input_len:].contiguous()

    return generate_ids