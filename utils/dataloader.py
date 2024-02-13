from copy import deepcopy
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.prompt import get_prompt, get_prompt_LAMOL, get_prompt_PCLL

def get_dataloader(params, CL_dataset, tokenizer):
    '''
        Build dataloader
    '''
    if params.classification_type == 'sentence-level' and params.method in ['LAMOL','L2KD','LAMOL_KD','LFPT5']:
        return get_dataloader_for_sentence_level_dataset_LAMOL(params, CL_dataset, tokenizer)
    elif params.classification_type == 'sentence-level' and params.method in ['PCLL']:
        return get_dataloader_for_sentence_level_dataset_PCLL(params, CL_dataset, tokenizer)
    elif params.classification_type == 'sentence-level':
        return get_dataloader_for_sentence_level_dataset_default(params, CL_dataset, tokenizer)
    elif params.classification_type == 'word-level':
        return get_dataloader_for_word_level_dataset_default(params, CL_dataset, tokenizer)
    else:
        raise NotImplementedError()
    
# =================================== Sentence-Level Tasks for Default Methods ======================================
def get_dataloader_for_sentence_level_dataset_default(params, CL_dataset, tokenizer):
    '''
        For sentence classification, the label of each sentence is a class name/class index.
    '''

    num_task = CL_dataset.continual_config['NUM_TASK']
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ''

    def preprocess_function_train_generative(examples):
        '''
            For training set for generative models (padding_side=left)
        '''
        with_ans_prompted_input_list = []
        no_ans_prompted_input_list = []
        answer_list = []
        target_list = []
        for i in range(len(examples[input_column])):
            _input = examples[input_column][i]
            _target = examples[target_column][i]
            with_ans_prompted_input_list.append(get_prompt(text=_input,
                                                           label=_target,
                                                           prompt_type=params.prompt_type,
                                                           eos_token=eos_token,
                                                           dataset=params.dataset))
            no_ans_prompted_input_list.append(get_prompt(text=_input,
                                                         label=None,
                                                         prompt_type=params.prompt_type,
                                                         eos_token=eos_token,
                                                         dataset=params.dataset))
            answer_list.append(' {0}{1}'.format(_target,eos_token))
            target_list.append(_target)

        # For training with Causal Language Modeling Loss
        print_max_len_information(tokenizer(with_ans_prompted_input_list)['input_ids'],params.max_seq_length)
        with_ans_lm_inputs = tokenizer(with_ans_prompted_input_list, 
                                       max_length=params.max_seq_length, 
                                       padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                                       truncation=True)
        with_ans_lm_inputs['input_ids_with_ans'] = deepcopy(with_ans_lm_inputs['input_ids'])
        with_ans_lm_inputs['attention_mask_with_ans'] = deepcopy(with_ans_lm_inputs['attention_mask'])
        with_ans_lm_inputs['labels_with_ans'] = deepcopy(with_ans_lm_inputs['input_ids'])
        with_ans_lm_inputs['label_idx_cil'] = deepcopy(examples['label_idx_cil'])
        with_ans_lm_inputs['label_idx_til'] = deepcopy(examples['label_idx_til'])
        with_ans_lm_inputs['target'] = target_list
        
        answer_lm_inputs = tokenizer(answer_list)
        for i,answer_tokens in enumerate(answer_lm_inputs['input_ids']):
            answer_len = len(answer_tokens)
            # Mask the answer tokens and only predict the answer tokens
            '''
                Example:

                Labels:      -100  -100 -100  -100   -100    answer tokens
                Index:        0     1     2     3      4        5     6
                Input:      [PAD] [PAD] [PAD] Input sentence answer tokens 

                NOTE: We do NOT need to shift the labels manually since this operation is implemented in GPT Models.
                For example, in forward() for GPTNeoXForCausalLM, the CausalLM loss is computed as follows:

                # we are doing next-token prediction; shift prediction scores and input ids by one
                shift_logits = lm_logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                
            '''
            with_ans_lm_inputs['labels_with_ans'][i][:-answer_len] = [-100]*(len(with_ans_lm_inputs['labels_with_ans'][i])-answer_len)

        # For training with Cross-Entropy Loss
        print_max_len_information(tokenizer(no_ans_prompted_input_list)['input_ids'],params.max_seq_length)
        no_ans_lm_inputs= tokenizer(no_ans_prompted_input_list, 
                                    max_length=params.max_seq_length, 
                                    padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                                    truncation=True)
        with_ans_lm_inputs['input_ids'] = deepcopy(no_ans_lm_inputs['input_ids'])
        with_ans_lm_inputs['attention_mask'] = deepcopy(no_ans_lm_inputs['attention_mask'])
        del no_ans_lm_inputs

        if 'instance_id' in examples.keys():
            with_ans_lm_inputs['instance_id'] = deepcopy(examples['instance_id'])
        if 'concept_id' in examples.keys():
            with_ans_lm_inputs['concept_id'] = deepcopy(examples['concept_id'])
        if 'relation_id' in examples.keys():
            with_ans_lm_inputs['relation_id'] = deepcopy(examples['relation_id'])
        
        return with_ans_lm_inputs
    
    def preprocess_function_predict_generative(examples):
        '''
            For validation and test set for generative models (padding_side=left)
        '''
        no_ans_prompted_input_list = []
        target_list = []
        for i in range(len(examples[input_column])):
            _input = examples[input_column][i]
            _target = examples[target_column][i]
            no_ans_prompted_input_list.append(get_prompt(text=_input,
                                                  label=None,
                                                  prompt_type=params.prompt_type,
                                                  eos_token=eos_token,
                                                  dataset=params.dataset))
            target_list.append(_target)

        print_max_len_information(tokenizer(no_ans_prompted_input_list)['input_ids'],params.max_seq_length)
        lm_inputs = tokenizer(no_ans_prompted_input_list, 
                              max_length=params.max_seq_length, 
                              padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                              truncation=True)
        lm_inputs['label_idx_cil'] = deepcopy(examples['label_idx_cil'])
        lm_inputs['label_idx_til'] = deepcopy(examples['label_idx_til'])
        lm_inputs['target'] = target_list

        if 'instance_id' in examples.keys():
            lm_inputs['instance_id'] = deepcopy(examples['instance_id'])
        if 'concept_id' in examples.keys():
            lm_inputs['concept_id'] = deepcopy(examples['concept_id'])
        if 'relation_id' in examples.keys():
            lm_inputs['relation_id'] = deepcopy(examples['relation_id'])

        return lm_inputs

    def preprocess_function_train_discriminative(examples):
        '''
            For training set for discriminative models (padding_side=right)
        '''
        with_ans_prompted_input_list = []
        no_ans_prompted_input_list = []
        answer_list = []
        target_list = []
        for i in range(len(examples[input_column])):
            _input = examples[input_column][i]
            _target = examples[target_column][i]
            with_ans_prompted_input_list.append(get_prompt(text=_input,
                                                           label=_target,
                                                           prompt_type=params.prompt_type,
                                                           eos_token=eos_token,
                                                           dataset=params.dataset))
            no_ans_prompted_input_list.append(get_prompt(text=_input,
                                                         label=None,
                                                         prompt_type=params.prompt_type,
                                                         eos_token=eos_token,
                                                         dataset=params.dataset))
            answer_list.append(' {0}{1}'.format(_target,eos_token))
            target_list.append(_target)

        # For training with Masked Language Modeling Loss
        print_max_len_information(tokenizer(with_ans_prompted_input_list)['input_ids'],params.max_seq_length)
        with_ans_lm_inputs = tokenizer(with_ans_prompted_input_list, 
                                       max_length=params.max_seq_length, 
                                       padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                                       truncation=True)
        with_ans_lm_inputs['input_ids_with_ans'] = deepcopy(with_ans_lm_inputs['input_ids'])
        with_ans_lm_inputs['attention_mask_with_ans'] = deepcopy(with_ans_lm_inputs['attention_mask'])
        with_ans_lm_inputs['labels_with_ans'] = deepcopy(with_ans_lm_inputs['input_ids'])
        with_ans_lm_inputs['label_idx_cil'] = deepcopy(examples['label_idx_cil'])
        with_ans_lm_inputs['label_idx_til'] = deepcopy(examples['label_idx_til'])
        with_ans_lm_inputs['target'] = target_list
        
        answer_lm_inputs = tokenizer(answer_list)
        for i, (attention_mask,answer_tokens) in enumerate(zip(with_ans_lm_inputs['attention_mask_with_ans'],answer_lm_inputs['input_ids'])):
            assert 'bert' in params.backbone, 'Only BERT is implemented for discriminative models for now!'
            # Remove the [CLS] token. [SEP] can be seen as a part of answer tokens.
            answer_len = len(answer_tokens) - 1
            nopad_input_len = sum(attention_mask)
            # Mask the answer tokens and only predict the answer tokens
            '''
                Example:

                Labels:      -100  answer  tokens  -100  -100  -100  -100
                Index:        0       1       2     3      4     5     6
                Input:      Input sentence answer tokens [PAD] [PAD] [PAD]

                NOTE:  We need to shift the labels manually since this operation is NOT implemented in BERT Models.
                For example, in forward() for BertForMaskedLM, the MaskedLM loss is computed as follows:

                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                
            '''
            # mask
            with_ans_lm_inputs['labels_with_ans'][i][:nopad_input_len-answer_len] = [-100]*len(with_ans_lm_inputs['labels_with_ans'][i][:nopad_input_len-answer_len])
            with_ans_lm_inputs['labels_with_ans'][i][nopad_input_len:] = [-100]*len(with_ans_lm_inputs['labels_with_ans'][i][nopad_input_len:])
            # shift
            with_ans_lm_inputs['labels_with_ans'][i][:-1] = with_ans_lm_inputs['labels_with_ans'][i][1:]
            with_ans_lm_inputs['labels_with_ans'][i][-1] = -100

        # For training with Cross-Entropy Loss
        print_max_len_information(tokenizer(no_ans_prompted_input_list)['input_ids'],params.max_seq_length)
        no_ans_lm_inputs= tokenizer(no_ans_prompted_input_list, 
                                    max_length=params.max_seq_length, 
                                    padding='max_length', 
                                    truncation=True)
        with_ans_lm_inputs['input_ids'] = deepcopy(no_ans_lm_inputs['input_ids'])
        with_ans_lm_inputs['attention_mask'] = deepcopy(no_ans_lm_inputs['attention_mask'])
        del no_ans_lm_inputs

        if 'instance_id' in examples.keys():
            with_ans_lm_inputs['instance_id'] = deepcopy(examples['instance_id'])
        if 'concept_id' in examples.keys():
            with_ans_lm_inputs['concept_id'] = deepcopy(examples['concept_id'])
        if 'relation_id' in examples.keys():
            with_ans_lm_inputs['relation_id'] = deepcopy(examples['relation_id'])

        return with_ans_lm_inputs
    
    def preprocess_function_predict_discriminative(examples):
        '''
            For validation and test set for discriminative models (padding_side=right)
        '''
        no_ans_prompted_input_list = []
        target_list = []
        for i in range(len(examples[input_column])):
            _input = examples[input_column][i]
            _target = examples[target_column][i]
            no_ans_prompted_input_list.append(get_prompt(text=_input,
                                                  label=None,
                                                  prompt_type=params.prompt_type,
                                                  eos_token=eos_token,
                                                  dataset=params.dataset))
            target_list.append(_target)

        print_max_len_information(tokenizer(no_ans_prompted_input_list)['input_ids'],params.max_seq_length)
        lm_inputs = tokenizer(no_ans_prompted_input_list, 
                              max_length=params.max_seq_length, 
                              padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                              truncation=True)
        lm_inputs['label_idx_cil'] = deepcopy(examples['label_idx_cil'])
        lm_inputs['label_idx_til'] = deepcopy(examples['label_idx_til'])
        lm_inputs['target'] = target_list

        if 'instance_id' in examples.keys():
            lm_inputs['instance_id'] = deepcopy(examples['instance_id'])
        if 'concept_id' in examples.keys():
            lm_inputs['concept_id'] = deepcopy(examples['concept_id'])
        if 'relation_id' in examples.keys():
            lm_inputs['relation_id'] = deepcopy(examples['relation_id'])

        return lm_inputs

    train_loader_list, dev_loader_list, test_loader_list = [], [], []

    input_column = 'input'
    target_column = 'target'

    for task_id in range(num_task):

        batch_size = params.batch_size//2 if task_id>0 and params.is_replay and params.Replay_batch_level else params.batch_size

        train_dataset = CL_dataset.continual_data[task_id]['train']
        train_dataset = train_dataset.map(preprocess_function_train_generative if params.backbone_type == 'generative' else preprocess_function_train_discriminative, 
                                          batched=True, 
                                          desc='Task %d Train Set'%(task_id+1), 
                                          batch_size=1000 if len(train_dataset)<20000 else 10000)
        if params.il_mode == 'IIL':
            train_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                            'label_idx_cil','label_idx_til',
                                                            'input_ids_with_ans', 'attention_mask_with_ans', 'labels_with_ans','target',
                                                            'instance_id','concept_id','relation_id'])
        else:
            train_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                            'label_idx_cil','label_idx_til',
                                                            'input_ids_with_ans', 'attention_mask_with_ans', 'labels_with_ans','target'])
        train_loader_list.append(
            DataLoader(train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        drop_last=False)
        )

        dev_dataset = CL_dataset.continual_data[task_id]['dev']
        dev_dataset = dev_dataset.map(preprocess_function_predict_generative if params.backbone_type == 'generative' else preprocess_function_predict_discriminative, 
                                      batched=True, 
                                      desc='Task %d Dev Set'%(task_id+1), 
                                      batch_size=1000)
        if params.il_mode == 'IIL':
            dev_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til','target',
                                                        'instance_id','concept_id','relation_id'])
        else:
            dev_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til','target'])
        dev_loader_list.append(
            DataLoader(dev_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

        test_dataset = CL_dataset.continual_data[task_id]['test']
        test_dataset = test_dataset.map(preprocess_function_predict_generative if params.backbone_type == 'generative' else preprocess_function_predict_discriminative, 
                                        batched=True, 
                                        desc='Task %d Test Set'%(task_id+1), 
                                        batch_size=1000)
        if params.il_mode == 'IIL':
            test_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til','target',
                                                        'instance_id','concept_id','relation_id'])
        else:
            test_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til','target'])
        test_loader_list.append(
            DataLoader(test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

    return train_loader_list, dev_loader_list, test_loader_list
# ==================================================================================================================



# =============================== Sentence-Level Tasks for PCLL ====================================================
def preprocess_function_train_and_predict_generative_PCLL(examples,params,tokenizer,num_task,task_id,input_column,target_column,eos_token,gen_token):
    '''
        For training set for generative models (padding_side=left)
    '''
    prompt_list = []
    inputsentence_list = []
    prompt_input_list = []
    prompt_input_anstoken_list = []
    prompt_input_anstoken_ans_list = []
    answer_list = []
    target_list = []
    for i in range(len(examples[input_column])):
        _input = examples[input_column][i]
        _target = examples[target_column][i]
        four_input_variants = get_prompt_PCLL(text=_input,
                                            label=_target,
                                            gen_token=gen_token,
                                            eos_token=eos_token,
                                            task_id=None if params.il_mode=='CIL' else task_id)
        prompt_list.append(four_input_variants[0])
        prompt_input_list.append(four_input_variants[1])
        prompt_input_anstoken_list.append(four_input_variants[2])
        prompt_input_anstoken_ans_list.append(four_input_variants[3])
        inputsentence_list.append(_input+eos_token)
        answer_list.append(' {0}{1}'.format(_target,eos_token))
        target_list.append(_target)


    # For training conditional VAE
    print_max_len_information(tokenizer(prompt_input_anstoken_ans_list)['input_ids'],params.max_seq_length)
    lm_inputs = tokenizer(prompt_list, 
                            max_length=params.max_seq_length, 
                            padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                            truncation=True)
    lm_inputs['label_idx_cil'] = deepcopy(examples['label_idx_cil'])
    lm_inputs['label_idx_til'] = deepcopy(examples['label_idx_til'])

    lm_inputs['input_ids_prompt'] = deepcopy(lm_inputs['input_ids'])
    lm_inputs['attention_mask_prompt'] = deepcopy(lm_inputs['attention_mask'])

    tmp_lm_inputs = tokenizer(prompt_input_list, 
                            max_length=params.max_seq_length, 
                            padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                            truncation=True)
    lm_inputs['input_ids_prompt_input'] = deepcopy(tmp_lm_inputs['input_ids'])
    lm_inputs['attention_mask_prompt_input'] = deepcopy(tmp_lm_inputs['attention_mask'])
    lm_inputs['labels_gen_prompt_input'] = deepcopy(tmp_lm_inputs['input_ids']) # for generating the input sentence only (used in conditional VAE)
    inputsentence_lm_inputs = tokenizer(inputsentence_list)
    for i, inputsentence_only in enumerate(inputsentence_lm_inputs['input_ids']):
        # Mask the answer tokens and only predict the answer tokens
        '''
            Example:

            Labels:      -100  -100 -100  -100   -100   Input Sentence
            Index:        0     1     2     3      4        5     6
            Input:      [PAD] [PAD] [PAD] Prompt Prefix Input Sentence 

            NOTE: We do NOT need to shift the labels manually since this operation is implemented in GPT Models.
            For example, in forward() for GPTNeoXForCausalLM, the CausalLM loss is computed as follows:

            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            
        '''
        inputsentence_len = sum(inputsentence_lm_inputs['attention_mask'][i])
        lm_inputs['labels_gen_prompt_input'][i][:-inputsentence_len] = [-100]*(len(lm_inputs['labels_gen_prompt_input'][i])-inputsentence_len)

    # For probing study
    tmp_lm_inputs = tokenizer(prompt_input_anstoken_list, 
                            max_length=params.max_seq_length, 
                            padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                            truncation=True)
    lm_inputs['input_ids_prompt_input_anstoken'] = deepcopy(tmp_lm_inputs['input_ids'])
    lm_inputs['attention_mask_prompt_input_anstoken'] = deepcopy(tmp_lm_inputs['attention_mask'])
    lm_inputs['input_ids'] = deepcopy(tmp_lm_inputs['input_ids']) # for test and evaluation
    lm_inputs['attention_mask'] = deepcopy(tmp_lm_inputs['attention_mask']) # for test and evaluation


    # For QA and generation loss (with Causal LM targets)
    tmp_lm_inputs = tokenizer(prompt_input_anstoken_ans_list, 
                            max_length=params.max_seq_length, 
                            padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                            truncation=True)
    lm_inputs['input_ids_prompt_input_anstoken_ans'] = deepcopy(tmp_lm_inputs['input_ids'])
    lm_inputs['attention_mask_prompt_input_anstoken_ans'] = deepcopy(tmp_lm_inputs['attention_mask'])
    lm_inputs['labels_qa_prompt_input_anstoken_ans'] = deepcopy(tmp_lm_inputs['input_ids']) # for generating answer only (used in LAMOL targets)
    lm_inputs['labels_gen_prompt_input_anstoken_ans'] = deepcopy(tmp_lm_inputs['input_ids']) # for generating the whole instance: prompt+inputsentence+answer (used in LAMOL targets)
    # Adjust the label for CausalLM according to the answer lens
    answer_lm_inputs = tokenizer(answer_list)
    for i,answer_tokens in enumerate(answer_lm_inputs['input_ids']):
        # Mask the answer tokens and only predict the answer tokens
        '''
            Example:

            Labels:      -100  -100 -100  -100   -100    answer tokens
            Index:        0     1     2     3      4        5     6
            Input:      [PAD] [PAD] [PAD] Input sentence answer tokens 

            Labels:      -100  -100 -100  Input sentence answer tokens
            Index:        0     1     2     3      4        5     6
            Input:      [PAD] [PAD] [PAD] Input sentence answer tokens 

            NOTE: We do NOT need to shift the labels manually since this operation is implemented in GPT Models.
            For example, in forward() for GPTNeoXForCausalLM, the CausalLM loss is computed as follows:

            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            
        '''
        answer_len = len(answer_tokens)
        lm_inputs['labels_qa_prompt_input_anstoken_ans'][i][:-answer_len] = [-100]*(len(lm_inputs['labels_qa_prompt_input_anstoken_ans'][i])-answer_len)
        non_pad_len = sum(lm_inputs['attention_mask_prompt_input_anstoken_ans'][i]) - 1 # remove the generation token at the begining
        lm_inputs['labels_gen_prompt_input_anstoken_ans'][i][:-non_pad_len] = [-100]*(len(lm_inputs['labels_gen_prompt_input_anstoken_ans'][i])-non_pad_len)

    lm_inputs['target'] = target_list

    if 'instance_id' in examples.keys():
        lm_inputs['instance_id'] = deepcopy(examples['instance_id'])
    if 'concept_id' in examples.keys():
        lm_inputs['concept_id'] = deepcopy(examples['concept_id'])
    if 'relation_id' in examples.keys():
        lm_inputs['relation_id'] = deepcopy(examples['relation_id'])

    return lm_inputs

def get_dataloader_for_sentence_level_dataset_PCLL(params, CL_dataset, tokenizer):
    '''
        For sentence classification, the label of each sentence is a class name/class index.

        This dataloader is for PCLL. 
        The training data contains QA and Generation instances
        The test data only contains QA instances.
        Different from LAMOL, PCLL defines "gen_token" as tokenizer.bos_token and don't use the answer token.
        according to the [implementation](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/pcll).
    '''

    assert params.backbone_type == 'generative', 'NotImplemented for backbone type %s'%(params.backbone_type)

    num_task = CL_dataset.continual_config['NUM_TASK']
    assert tokenizer.eos_token is not None
    eos_token = tokenizer.eos_token
    assert tokenizer.bos_token is not None
    gen_token = tokenizer.bos_token 

    train_loader_list, dev_loader_list, test_loader_list = [], [], []

    input_column = 'input'
    target_column = 'target'

    for task_id in range(num_task):

        batch_size = params.batch_size//2 if task_id>0 and params.is_replay and params.Replay_batch_level else params.batch_size

        train_dataset = CL_dataset.continual_data[task_id]['train']
        train_dataset = train_dataset.map(preprocess_function_train_and_predict_generative_PCLL, 
                                          batched=True, 
                                          desc='Task %d Train Set'%(task_id+1), 
                                          batch_size=1000,
                                          fn_kwargs={
                                              'params':params,
                                              'tokenizer':tokenizer,
                                              'num_task':num_task,
                                              'task_id':task_id,
                                              'input_column':input_column,
                                              'target_column':target_column,
                                              'eos_token':eos_token,
                                              'gen_token':gen_token,
                                          })
        if params.il_mode == 'IIL':
            train_dataset.set_format(type='torch', columns=[
                'input_ids', 'attention_mask',
                'label_idx_cil', 'label_idx_til',
                'input_ids_prompt', 'attention_mask_prompt',
                'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans',
                'target', 'instance_id', 'concept_id', 'relation_id'
            ])
        else:
            train_dataset.set_format(type='torch', columns=[
                'input_ids', 'attention_mask',
                'label_idx_cil', 'label_idx_til',
                'input_ids_prompt', 'attention_mask_prompt',
                'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans'
            ])
        train_loader_list.append(
            DataLoader(train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        drop_last=False)
        )

        dev_dataset = CL_dataset.continual_data[task_id]['dev']
        dev_dataset = dev_dataset.map(preprocess_function_train_and_predict_generative_PCLL, 
                                      batched=True, 
                                      desc='Task %d Dev Set'%(task_id+1), 
                                      batch_size=1000,
                                      fn_kwargs={
                                              'params':params,
                                              'tokenizer':tokenizer,
                                              'num_task':num_task,
                                              'task_id':task_id,
                                              'input_column':input_column,
                                              'target_column':target_column,
                                              'eos_token':eos_token,
                                              'gen_token':gen_token,
                                     })
        if params.il_mode == 'IIL':
            dev_dataset.set_format(type='torch', columns=[
                'input_ids', 'attention_mask',
                'label_idx_cil', 'label_idx_til',
                'input_ids_prompt', 'attention_mask_prompt',
                'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans',
                'target', 'instance_id', 'concept_id', 'relation_id'
            ])
        else:
            
            dev_dataset.set_format(type='torch', columns=[
                'input_ids', 'attention_mask',
                'label_idx_cil', 'label_idx_til',
                'input_ids_prompt', 'attention_mask_prompt',
                'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans'
            ])
        dev_loader_list.append(
            DataLoader(dev_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

        test_dataset = CL_dataset.continual_data[task_id]['test']
        test_dataset = test_dataset.map(preprocess_function_train_and_predict_generative_PCLL, 
                                        batched=True, 
                                        desc='Task %d Test Set'%(task_id+1), 
                                        batch_size=1000,
                                        fn_kwargs={
                                              'params':params,
                                              'tokenizer':tokenizer,
                                              'num_task':num_task,
                                              'task_id':task_id,
                                              'input_column':input_column,
                                              'target_column':target_column,
                                              'eos_token':eos_token,
                                              'gen_token':gen_token,
                                        })
        if params.il_mode == 'IIL':
            test_dataset.set_format(type='torch', columns=[
                'input_ids', 'attention_mask',
                'label_idx_cil', 'label_idx_til',
                'input_ids_prompt', 'attention_mask_prompt',
                'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans',
                'target', 'instance_id', 'concept_id', 'relation_id'
            ])
        else:
            test_dataset.set_format(type='torch', columns=[
                'input_ids', 'attention_mask',
                'label_idx_cil', 'label_idx_til',
                'input_ids_prompt', 'attention_mask_prompt',
                'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans'
            ])
        test_loader_list.append(
            DataLoader(test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

    return train_loader_list, dev_loader_list, test_loader_list
# ==================================================================================================================



# =============================== Sentence-Level Tasks for LAMOL-based Models ======================================
def preprocess_function_train_generative_LAMOL(examples,params,tokenizer,num_task,task_id,input_column,target_column,ans_token,eos_token,gen_token):
    '''
        For training set for generative models (padding_side=left)
    '''
    with_ans_prompted_input_list = []
    with_gen_ans_prompted_input_list = []
    no_ans_prompted_input_list = []
    answer_list = []
    target_list = []
    for i in range(len(examples[input_column])):
        _input = examples[input_column][i]
        _target = examples[target_column][i]
        # For QA target
        with_ans_prompted_input_list.append(get_prompt_LAMOL(text=_input,
                                                        label=_target,
                                                        gen_token='',
                                                        ans_token=ans_token,
                                                        eos_token=eos_token))
        _gen_token = '__%d__'%(task_id) if params.LAMOL_use_task_specific_gen_token else gen_token
        with_gen_ans_prompted_input_list.append(get_prompt_LAMOL(text=_input,
                                                        label=_target,
                                                        gen_token=_gen_token,
                                                        ans_token=ans_token,
                                                        eos_token=eos_token))
        # For probing or evaluation
        no_ans_prompted_input_list.append(get_prompt_LAMOL(text=_input,
                                                        label=None,
                                                        gen_token='',
                                                        ans_token=ans_token,
                                                        eos_token=eos_token))
        # NOTE: No space before and after answertoken in LAMOL!!!
        answer_list.append('{0}{1}'.format(_target,eos_token)) 
        target_list.append(_target)


    # For training with Causal Language Modeling Loss (QA target)
    print_max_len_information(tokenizer(with_ans_prompted_input_list)['input_ids'],params.max_seq_length)
    with_ans_lm_inputs = tokenizer(with_ans_prompted_input_list, 
                                    max_length=params.max_seq_length, 
                                    padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                                    truncation=True)
    with_ans_lm_inputs['label_idx_cil'] = deepcopy(examples['label_idx_cil'])
    with_ans_lm_inputs['label_idx_til'] = deepcopy(examples['label_idx_til'])
    with_ans_lm_inputs['input_ids_with_ans'] = deepcopy(with_ans_lm_inputs['input_ids'])
    with_ans_lm_inputs['attention_mask_with_ans'] = deepcopy(with_ans_lm_inputs['attention_mask'])
    with_ans_lm_inputs['labels_with_ans'] = deepcopy(with_ans_lm_inputs['input_ids'])
    # Adjust the label for CausalLM according to the answer lens
    answer_lm_inputs = tokenizer(answer_list)
    for i,answer_tokens in enumerate(answer_lm_inputs['input_ids']):
        answer_len = len(answer_tokens)
        # Mask the answer tokens and only predict the answer tokens
        '''
            Example:

            Labels:      -100  -100 -100  -100   -100    answer tokens
            Index:        0     1     2     3      4        5     6
            Input:      [PAD] [PAD] [PAD] Input sentence answer tokens 

            NOTE: We do NOT need to shift the labels manually since this operation is implemented in GPT Models.
            For example, in forward() for GPTNeoXForCausalLM, the CausalLM loss is computed as follows:

            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            
        '''
        with_ans_lm_inputs['labels_with_ans'][i][:-answer_len] = [-100]*(len(with_ans_lm_inputs['labels_with_ans'][i])-answer_len)

    # For training with Causal Language Modeling Loss (Generation target)
    print_max_len_information(tokenizer(with_gen_ans_prompted_input_list)['input_ids'],params.max_seq_length)
    with_gen_ans_lm_inputs = tokenizer(with_gen_ans_prompted_input_list, 
                                    max_length=params.max_seq_length, 
                                    padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                                    truncation=True)
    with_ans_lm_inputs['input_ids_with_gen_ans'] = deepcopy(with_gen_ans_lm_inputs['input_ids'])
    with_ans_lm_inputs['attention_mask_with_gen_ans'] = deepcopy(with_gen_ans_lm_inputs['attention_mask'])
    with_ans_lm_inputs['labels_with_gen_ans'] = deepcopy(with_gen_ans_lm_inputs['input_ids'])
    # Adjust the label for CausalLM
    gen_token_input_ids = tokenizer(['__%d__'%(t_id) if params.LAMOL_use_task_specific_gen_token else gen_token for t_id in range(num_task)])['input_ids']
    gen_token_len = [len(_gen_tokens) for _gen_tokens in gen_token_input_ids] # len of each generation token
    for i in range(len(with_ans_lm_inputs['labels_with_gen_ans'])):
        # Mask the answer tokens and only predict the answer tokens
        '''
            Example:

            Labels:      -100  -100   -100  Input sentence answer tokens
            Index:        0     1      2     3      4        5     6
            Input:      [PAD] [PAD] __gen__ Input sentence answer tokens 

            NOTE: We do NOT need to shift the labels manually since this operation is implemented in GPT Models.
            For example, in forward() for GPTNeoXForCausalLM, the CausalLM loss is computed as follows:

            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            
        '''
        total_len = len(with_ans_lm_inputs['attention_mask_with_gen_ans'][i])
        pad_len = total_len - sum(with_ans_lm_inputs['attention_mask_with_gen_ans'][i])
        pad_len += gen_token_len[task_id]
        with_ans_lm_inputs['labels_with_gen_ans'][i][:pad_len] = [-100]*pad_len

    # For probing or evaluation
    print_max_len_information(tokenizer(no_ans_prompted_input_list)['input_ids'],params.max_seq_length)
    no_ans_lm_inputs= tokenizer(no_ans_prompted_input_list, 
                                max_length=params.max_seq_length, 
                                padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                                truncation=True)
    with_ans_lm_inputs['input_ids'] = deepcopy(no_ans_lm_inputs['input_ids'])
    with_ans_lm_inputs['attention_mask'] = deepcopy(no_ans_lm_inputs['attention_mask'])
    with_ans_lm_inputs['target'] = deepcopy(target_list)

    if 'instance_id' in examples.keys():
        with_ans_lm_inputs['instance_id'] = deepcopy(examples['instance_id'])
    if 'concept_id' in examples.keys():
        with_ans_lm_inputs['concept_id'] = deepcopy(examples['concept_id'])
    if 'relation_id' in examples.keys():
        with_ans_lm_inputs['relation_id'] = deepcopy(examples['relation_id'])
    
    del no_ans_lm_inputs

    return with_ans_lm_inputs

def preprocess_function_predict_generative_LAMOL(examples,params,tokenizer,input_column,target_column,ans_token,eos_token):
    '''
        For validation and test set for generative models (padding_side=left)
    '''
    no_ans_prompted_input_list = []
    target_list = []
    for i in range(len(examples[input_column])):
        _input = examples[input_column][i]
        _target = examples[target_column][i]
        no_ans_prompted_input_list.append(get_prompt_LAMOL(text=_input,
                                                            label=None,
                                                            gen_token='',
                                                            ans_token=ans_token,
                                                            eos_token=eos_token))
        target_list.append(_target)

    print_max_len_information(tokenizer(no_ans_prompted_input_list)['input_ids'],params.max_seq_length)
    lm_inputs = tokenizer(no_ans_prompted_input_list, 
                            max_length=params.max_seq_length, 
                            padding='max_length', # All sentence MUST have the same lenth for probing and buffer
                            truncation=True)
    lm_inputs['label_idx_cil'] = deepcopy(examples['label_idx_cil'])
    lm_inputs['label_idx_til'] = deepcopy(examples['label_idx_til'])
    lm_inputs['target'] = target_list

    if 'instance_id' in examples.keys():
        lm_inputs['instance_id'] = deepcopy(examples['instance_id'])
    if 'concept_id' in examples.keys():
        lm_inputs['concept_id'] = deepcopy(examples['concept_id'])
    if 'relation_id' in examples.keys():
        lm_inputs['relation_id'] = deepcopy(examples['relation_id'])

    return lm_inputs

def get_dataloader_for_sentence_level_dataset_LAMOL(params, CL_dataset, tokenizer):
    '''
        For sentence classification, the label of each sentence is a class name/class index.

        This dataloader is for LAMOL-based models. 
        The training data contains QA and Generation instances
        The test data only contains QA instances.
        Besides, we define "ans_token":'__ans__', "gen_token":'__gen__'
        according to the [implementation](https://github.com/jojotenya/LAMOL).
    '''

    assert params.backbone_type == 'generative', 'NotImplemented for backbone type %s'%(params.backbone_type)

    num_task = CL_dataset.continual_config['NUM_TASK']
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ''

    ans_token = '__ans__'
    tokenizer.add_tokens([ans_token])
    if params.LAMOL_use_task_specific_gen_token:
        for t_id in range(num_task):
            gen_token = '__%d__'%(t_id)
            tokenizer.add_tokens([gen_token])
    else:
        gen_token = '__gen__'
        tokenizer.add_tokens([gen_token])

    train_loader_list, dev_loader_list, test_loader_list = [], [], []

    input_column = 'input'
    target_column = 'target'

    for task_id in range(num_task):

        batch_size = params.batch_size//2 if task_id>0 and params.is_replay and params.Replay_batch_level else params.batch_size

        train_dataset = CL_dataset.continual_data[task_id]['train']
        train_dataset = train_dataset.map(preprocess_function_train_generative_LAMOL, 
                                          batched=True, 
                                          desc='Task %d Train Set'%(task_id+1), 
                                          batch_size=1000,
                                          fn_kwargs={
                                              'params':params,
                                              'tokenizer':tokenizer,
                                              'num_task':num_task,
                                              'task_id':task_id,
                                              'input_column':input_column,
                                              'target_column':target_column,
                                              'ans_token':ans_token,
                                              'eos_token':eos_token,
                                              'gen_token':gen_token,
                                          })
        if params.il_mode == 'IIL':
            train_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                            'label_idx_cil','label_idx_til',
                                                            'input_ids_with_ans', 'attention_mask_with_ans', 'labels_with_ans', 
                                                            'input_ids_with_gen_ans', 'attention_mask_with_gen_ans', 'labels_with_gen_ans',
                                                            'target', 'instance_id', 'concept_id', 'relation_id'])
        else:
            train_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                            'label_idx_cil','label_idx_til',
                                                            'input_ids_with_ans', 'attention_mask_with_ans', 'labels_with_ans', 
                                                            'input_ids_with_gen_ans', 'attention_mask_with_gen_ans', 'labels_with_gen_ans'])
        train_loader_list.append(
            DataLoader(train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        drop_last=False)
        )

        dev_dataset = CL_dataset.continual_data[task_id]['dev']
        dev_dataset = dev_dataset.map(preprocess_function_predict_generative_LAMOL, 
                                      batched=True, 
                                      desc='Task %d Dev Set'%(task_id+1), 
                                      batch_size=1000,
                                      fn_kwargs={
                                        'params':params,
                                        'tokenizer':tokenizer,
                                        'input_column':input_column,
                                        'target_column':target_column,
                                        'ans_token':ans_token,
                                        'eos_token':eos_token,
                                      })
        if params.il_mode == 'IIL':
            dev_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til',
                                                        'target', 'instance_id', 'concept_id', 'relation_id'])
        else:
            dev_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til'])
        dev_loader_list.append(
            DataLoader(dev_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

        test_dataset = CL_dataset.continual_data[task_id]['test']
        test_dataset = test_dataset.map(preprocess_function_predict_generative_LAMOL, 
                                        batched=True, 
                                        desc='Task %d Test Set'%(task_id+1), 
                                        batch_size=1000,
                                        fn_kwargs={
                                        'params':params,
                                        'tokenizer':tokenizer,
                                        'input_column':input_column,
                                        'target_column':target_column,
                                        'ans_token':ans_token,
                                        'eos_token':eos_token,
                                      })
        if params.il_mode == 'IIL':
            test_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til',
                                                        'target', 'instance_id', 'concept_id', 'relation_id'])
        else:
            test_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil','label_idx_til'])
        test_loader_list.append(
            DataLoader(test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

    return train_loader_list, dev_loader_list, test_loader_list
# ===================================================================================================================



# ====================================== Word-Level Tasks for Default Methods =======================================
def get_dataloader_for_word_level_dataset_default(params, CL_dataset, tokenizer):
    '''
        For word classification, the label of each sentence is a list. 

        NOTE:
        - We do NOT use prompt for word-level classification tasks
        - 'input_ids_with_ans', 'attention_mask_with_ans', 'labels_with_ans' are not available since it is not expected to use Causal/Masked LM loss
    '''

    num_task = CL_dataset.continual_config['NUM_TASK']
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ''

    assert params.backbone_type == 'discriminative', 'Only implemented for discriminative models'
    assert params.il_mode == 'CIL', 'NotImplemented for il mode %s'%(params.il_mode)
    
    def preprocess_function_train_predict_discriminative(examples):
        '''
            For train, validation and test set for discriminative models (padding_side=right)
        '''
        assert 'bert' in params.backbone, 'Only BERT is implemented for discriminative models for now!'
        
        input_ids_list = []
        attention_mask_list = []
        label_idx_cil_list = []
        for i in range(len(examples[input_column])):
            x_list, y_list = [], []
            one_sent_x = examples[input_column][i]
            one_sent_y = examples[label_idx_column][i]
            for word, tag in zip(one_sent_x,one_sent_y):
                subs_ = tokenizer.tokenize(word)
                if len(subs_) == 0:
                    print('No subwords for %s'%(word))
                    continue
                x_list.extend(tokenizer.convert_tokens_to_ids(subs_))
                y_list.extend([CL_dataset.continual_config['label2idx'][tag]] + [-100]*(len(subs_)-1))
            input_ids_list.append([tokenizer.cls_token_id] + x_list + [tokenizer.sep_token_id])
            label_idx_cil_list.append([-100] + y_list + [-100])
        
        print_max_len_information(input_ids_list,params.max_seq_length)
        max_len = params.max_seq_length
        for i in range(len(examples[input_column])):
            seq_len = len(input_ids_list[i])
            if seq_len<=max_len:
                input_ids_list[i] = input_ids_list[i]+[tokenizer.pad_token_id]*(max_len-seq_len)
                label_idx_cil_list[i] = label_idx_cil_list[i]+[-100]*(max_len-seq_len)
                attention_mask = [1]*seq_len+[0]*(max_len-seq_len)
            else:
                input_ids_list[i] = input_ids_list[i][:max_len]
                label_idx_cil_list[i] = label_idx_cil_list[i][:max_len]
                attention_mask = [1]*max_len
            attention_mask_list.append(attention_mask)

        lm_inputs = {}
        lm_inputs['input_ids'] = torch.tensor(input_ids_list)
        lm_inputs['attention_mask'] = torch.tensor(attention_mask_list)
        lm_inputs['label_idx_cil'] = torch.tensor(label_idx_cil_list)

        assert lm_inputs['input_ids'].shape == (len(examples[input_column]),max_len)
        assert lm_inputs['attention_mask'].shape == (len(examples[input_column]),max_len)
        assert lm_inputs['label_idx_cil'].shape == (len(examples[input_column]),max_len)

        return lm_inputs

    train_loader_list, dev_loader_list, test_loader_list = [], [], []

    input_column = 'input'
    label_idx_column = 'label_idx_cil'

    for task_id in range(num_task):

        batch_size = params.batch_size//2 if task_id>0 and params.is_replay and params.Replay_batch_level else params.batch_size

        train_dataset = CL_dataset.continual_data[task_id]['train']
        train_dataset = train_dataset.map(preprocess_function_train_predict_discriminative, 
                                          batched=True, 
                                          desc='Task %d Train Set'%(task_id+1), 
                                          batch_size=1000)
        train_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                        'label_idx_cil'])
        train_loader_list.append(
            DataLoader(train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        drop_last=False)
        )

        dev_dataset = CL_dataset.continual_data[task_id]['dev']
        dev_dataset = dev_dataset.map(preprocess_function_train_predict_discriminative, 
                                      batched=True, 
                                      desc='Task %d Dev Set'%(task_id+1), 
                                      batch_size=1000)
        dev_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                      'label_idx_cil'])
        dev_loader_list.append(
            DataLoader(dev_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

        test_dataset = CL_dataset.continual_data[task_id]['test']
        test_dataset = test_dataset.map(preprocess_function_train_predict_discriminative, 
                                        batched=True, 
                                        desc='Task %d Test Set'%(task_id+1), 
                                        batch_size=1000)
        test_dataset.set_format(type='torch', columns=['input_ids','attention_mask',
                                                       'label_idx_cil'])
        test_loader_list.append(
            DataLoader(test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last=False)
        )

    return train_loader_list, dev_loader_list, test_loader_list
# ===================================================================================================================



# ============================================= Other Functions =====================================================
def print_max_len_information(input_ids_list, max_seq_length: int) -> None:
    '''
        Print the statistics of the sentence length (>max_seq_length)

        Args:
         - input_ids_list: List[list]
         - max_seq_length: the predefined max sequence length, e.g., 128
    '''
    cnt_dict = Counter([len(input_ids) for input_ids in input_ids_list])
    cnt_dict_sorted_key = sorted(cnt_dict.keys())
    accum = 0
    ratio = -1
    for k in cnt_dict_sorted_key:
        accum += cnt_dict[k]
        if k>=max_seq_length:
            if ratio == -1:
                ratio = 100-accum/len(input_ids_list)*100
                if ratio>1:
                    print('The params.max_seq_length may be too small! %.2f%% sentence are truncated!'%(ratio))
            print('Len <= %d (%.2f%%=%d/%d)'%(k,accum/len(input_ids_list)*100,accum,len(input_ids_list)))