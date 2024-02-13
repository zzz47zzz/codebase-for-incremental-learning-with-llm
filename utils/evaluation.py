import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
from random import shuffle

from utils.backbone import obtain_features, obtain_generate_ids
from utils.prompt import get_prompt_ICL

logger = logging.getLogger()

# ====================== the evaluation metric for each task ==========================================================================

# ------------------------------------- sentence level evaluation ----------------------------------------------
def evaluate_sent_level_acc_with_generation(model, eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy using generation for sentence-level classification tasks

        Return:
         - Acc (%)
    '''
    model.eval()
    acc_list_all = []
    save_dict = {}
    
    with torch.no_grad():
        for lm_input in eval_data_loader: 

            label_idx = lm_input['label_idx_cil']
            generate_ids = obtain_generate_ids(params=params, 
                                                model=model, 
                                                lm_input=lm_input, 
                                                tokenizer=tokenizer)

            # Gather from different processes in DDP
            generate_ids = accelerator.pad_across_processes(
                                                generate_ids, 
                                                dim=1, 
                                                pad_index=tokenizer.eos_token_id, 
                                                pad_first=False)
            generate_ids, label_idx = accelerator.gather_for_metrics((generate_ids, label_idx))

            generate_seq = tokenizer.batch_decode(generate_ids,skip_special_tokens=True)
            generate_seq = [pred.strip(' \n').lower().split(tokenizer.eos_token)[0] for pred in generate_seq]

            # Compare the target directly
            if idx2label == []:
                target_output = lm_input['target']

                acc_list = [1 if pred.strip().lower()==target.strip().lower() else 0 for pred, target in zip(generate_seq,target_output)]
                acc_list_all.extend(acc_list)

                if params.il_mode == 'IIL':
                    for i, (pred, target) in enumerate(zip(generate_seq,target_output)):
                        save_dict[lm_input['instance_id'][i].item()] = {
                            'instance_id': lm_input['instance_id'][i].item(),
                            'relation_id': lm_input['relation_id'][i].item(),
                            'concept_id': lm_input['concept_id'][i].item(),
                            'prediton': pred.strip().lower(),
                            'target': target.strip().lower(),
                        }

            # Compare the class idx
            else:
                label_idx = label_idx.detach().cpu().numpy()
                target_output = [idx2label[l] for l in label_idx]
                target_output = [target.strip(' \n').lower() for target in target_output]

                acc_list = [1 if pred==target else 0 for pred, target in zip(generate_seq,target_output)]
                acc_list_all.extend(acc_list)
    
    acc = np.round(np.mean(acc_list_all)*100,3)
    model.train()

    if params.il_mode == 'IIL':
        return acc, save_dict

    return acc

def evaluate_sent_level_acc_with_classifier(model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy with classifier for sentence-level classification tasks

        Return:
         - Acc (%)
    '''
    model.eval()
    acc_list_all = []
            
    with torch.no_grad():
        for lm_input in eval_data_loader: 
       
            extracted_features = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input,
                                                    tokenizer=tokenizer)
            if params.il_mode == 'CIL':
                logits = torch.concatenate([classifier(extracted_features) for classifier in classifier_list[:cur_task_id+1]],dim=-1)
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_cil']
            elif params.il_mode == 'TIL':
                # NOTE: In the TIL mode, the cur_task_id should be set as eval_task_id!
                logits = classifier_list[cur_task_id](extracted_features) 
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_til']
            else:
                raise NotImplementedError()
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            label_idx = label_idx.detach().cpu().numpy()
            acc_list = [1 if pred==target else 0 for pred, target in zip(preds,label_idx)]
            acc_list_all.extend(acc_list)
            acc = np.round(np.mean(acc_list_all)*100,3)
    
    model.train()

    return acc

def evaluate_sent_level_acc_with_classifier_adapter(model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy with classifiers and adapters for sentence-level classification tasks 

        Return:
         - Acc (%)
    '''
    model.eval()
    acc_list_all = []

    def set_adapter_to_task(t_id):
        try:
            model.set_adapter('task-%d'%(t_id))
        except:
            try:
                model.set_active_adapters('task-%d'%(t_id))
            except:
                raise NotImplementedError()

            
    with torch.no_grad():
        for lm_input in eval_data_loader: 
    
            if params.il_mode == 'CIL':
                # NOTE: requires O(num_tasks) forward!
                logits_list = []
                for t_id in range(cur_task_id+1):
                    set_adapter_to_task(t_id)
                    extracted_features = obtain_features(params=params, 
                                                        model=model, 
                                                        lm_input=lm_input,
                                                        tokenizer=tokenizer)
                    logits_list.append(classifier_list[t_id](extracted_features))
                logits = torch.cat(logits_list,dim=-1)
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_cil']

            elif params.il_mode == 'TIL':
                # NOTE: In the TIL mode, the cur_task_id should be set as eval_task_id!
                set_adapter_to_task(cur_task_id)
                extracted_features = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input,
                                                    tokenizer=tokenizer)
                logits = classifier_list[cur_task_id](extracted_features) 
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_til']
            else:
                raise NotImplementedError()
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            label_idx = label_idx.detach().cpu().numpy()
            acc_list = [1 if pred==target else 0 for pred, target in zip(preds,label_idx)]
            acc_list_all.extend(acc_list)
            acc = np.round(np.mean(acc_list_all)*100,3)
    
    model.train()

    return acc

def evaluate_sent_level_acc_with_classifier_model_list(model_list, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params):
    '''
        Evaluate the accuracy with classifier and model_list for sentence-level classification tasks

        Return:
         - Acc (%)
    '''
    assert params.il_mode == 'CIL', 'NotImplemented for il mode %s'%(params.il_mode)
    model_list[cur_task_id].eval()
    acc_list_all = []
            
    with torch.no_grad():
        for lm_input in eval_data_loader: 
            logits = []
            for t_id in range(cur_task_id+1):
                # For Distributed Data Parallel
                if hasattr(model_list[t_id],'module'):
                    model = model_list[t_id].module
                else:
                    model = model_list[t_id]
                extracted_feature = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input, 
                                                    tokenizer=tokenizer)
                logits.append(classifier_list[t_id](extracted_feature)) 

            logits = torch.concatenate(logits,dim=-1)
            preds = logits.argmax(dim=-1)
            label_idx = lm_input['label_idx_cil']
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))

            acc_list = [1 if pred==target else 0 for pred, target in zip(preds,label_idx)]
            acc_list_all.extend(acc_list)
    
    acc = np.round(np.mean(acc_list_all)*100,3)
    model_list[cur_task_id].train()
    
    model.train()

    return acc

def evaluate_with_generation_ICL(model, train_data_loader, eval_data_loader, tokenizer, accelerator, params):
    '''
        Evaluate the accuracy using generation for sentence-level classification tasks

        Return:
         - Acc (%)
    '''

    assert params.il_mode == 'IIL'

    model.eval()
    acc_list_all = []
    save_dict = {}

    train_data_loader = DataLoader(train_data_loader.dataset, 
                                    batch_size=params.batch_size, 
                                    shuffle=False if params.ICL_same_instance or params.ICL_same_concept else True,
                                    drop_last=False)
    train_data_loader = accelerator.prepare(train_data_loader)
    total_sample = len(eval_data_loader.dataset)

    with torch.no_grad():
        for train_lm_input, eval_lm_input in zip(train_data_loader, eval_data_loader): 
            bs = eval_lm_input['input_ids'].shape[0]
            prompted_input_all = []

            # Prepare eval sample
            eval_inputs = tokenizer.batch_decode(eval_lm_input['input_ids'],skip_special_tokens=True) 

            for bs_i in range(bs):
                # Prepare N-shot sample                
                train_input_ids = []
                train_target = []
                if params.ICL_same_concept:
                    eval_concept = eval_lm_input['concept_id'][bs_i]
                    match_idx = torch.where(train_data_loader.dataset['concept_id']==eval_concept.item())[0]
                    select_idx = np.random.randint(match_idx.shape[0],size=params.ICL_n_shot)
                    for s_idx in select_idx:
                        s_idx = int(s_idx)
                        train_input_ids.append(train_data_loader.dataset['input_ids'][s_idx])
                        train_target.append(train_data_loader.dataset['target'][s_idx])
                elif params.ICL_same_instance:
                    if params.ICL_n_shot>1:
                        select_idx = np.random.randint(total_sample,size=params.ICL_n_shot-1)
                        for s_idx in select_idx:
                            s_idx = int(s_idx)
                            train_input_ids.append(train_data_loader.dataset['input_ids'][s_idx])
                            train_target.append(train_data_loader.dataset['target'][s_idx])
                    train_input_ids.append(train_lm_input['input_ids'][bs_i])
                    train_target.append(train_lm_input['target'][bs_i])
                else:
                    select_idx = np.random.randint(total_sample,size=params.ICL_n_shot)
                    for s_idx in select_idx:
                        s_idx = int(s_idx)
                        train_input_ids.append(train_data_loader.dataset['input_ids'][s_idx])
                        train_target.append(train_data_loader.dataset['target'][s_idx])

                train_inputs = tokenizer.batch_decode(train_input_ids,skip_special_tokens=True)
                shuffle_demonstration = list(range(params.ICL_n_shot))
                shuffle(shuffle_demonstration)
                _train_input_list = []
                for i_shot in shuffle_demonstration:
                    _train_input_list.append(f'{train_inputs[i_shot]} {train_target[i_shot]}')

                prompted_input = get_prompt_ICL(_train_input_list, 
                                                eval_inputs[bs_i],
                                                )
                prompted_input_all.append(prompted_input)

            prompted_eval_lm_input = tokenizer(prompted_input_all,
                                                padding='longest',
                                                return_tensors="pt").data
            prompted_eval_lm_input = {
                'input_ids': prompted_eval_lm_input['input_ids'].to(model.device),
                'attention_mask': prompted_eval_lm_input['attention_mask'].to(model.device)
            }

            generate_ids = obtain_generate_ids(params=params, 
                                                model=model, 
                                                lm_input=prompted_eval_lm_input, 
                                                tokenizer=tokenizer)

            # Gather from different processes in DDP
            generate_ids = accelerator.pad_across_processes(
                                                generate_ids, 
                                                dim=1, 
                                                pad_index=tokenizer.eos_token_id, 
                                                pad_first=False)
            generate_ids = accelerator.gather_for_metrics(generate_ids)

            generate_seq = tokenizer.batch_decode(generate_ids,skip_special_tokens=True)
            generate_seq = [pred.strip(' \n').lower().split(tokenizer.eos_token)[0].split('\n')[0].strip() 
                            for pred in generate_seq]

            # Compare the target directly
            target_output = eval_lm_input['target']

            acc_list = [1 if pred.strip().lower()==target.strip().lower() else 0 for pred, target in zip(generate_seq,target_output)]
            acc_list_all.extend(acc_list)

            if params.il_mode == 'IIL':
                for i, (pred, target) in enumerate(zip(generate_seq,target_output)):
                    save_dict[eval_lm_input['instance_id'][i].item()] = {
                        'instance_id': eval_lm_input['instance_id'][i].item(),
                        'relation_id': eval_lm_input['relation_id'][i].item(),
                        'concept_id': eval_lm_input['concept_id'][i].item(),
                        'prediton': pred.strip().lower(),
                        'target': target.strip().lower(),
                    }

            if len(acc_list_all)%(5*bs)==0: 
                logger.info('ICL Progress %.2f%%(=%d/%d)'%
                    (len(acc_list_all)/total_sample*100,
                    len(acc_list_all),
                    total_sample)
                )
    
    acc = np.round(np.mean(acc_list_all)*100,3)
    model.train()

    if params.il_mode == 'IIL':
        return acc, save_dict

    return acc

# ------------------------------------- sentence level evaluation ----------------------------------------------
def evaluate_word_level_acc_with_classifier(model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy with classifier for word-level classification tasks

        Return:
         - macro f1 (%)
    '''
    assert params.il_mode == 'CIL', 'NotImplemented for il mode %s'%(params.il_mode)
    model.eval()
            
    with torch.no_grad():
        # Only test once because the last test set contains all labels
        gold_lines = []
        pred_lines = []

        for lm_input in eval_data_loader: 
            extracted_features = obtain_features(params=params, 
                                                model=model, 
                                                lm_input=lm_input,
                                                tokenizer=tokenizer)

            logits = torch.concatenate([classifier(extracted_features) for classifier in classifier_list[:cur_task_id+1]],dim=-1)
            preds = logits.argmax(dim=-1)
            label_idx = lm_input['label_idx_cil']
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            preds, label_idx = preds.cpu(), label_idx.cpu()

            for _sent_gold, _sent_pred in zip(label_idx,preds):
                gold_lines.append([idx2label[_gold.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])
                pred_lines.append([idx2label[_pred.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])

        # micro f1 (average over all instances)
        mi_f1 = f1_score(gold_lines, pred_lines)*100
        # macro f1 (default, average over each class)
        ma_f1 = f1_score(gold_lines, pred_lines, average='macro')*100

        logger.info('Evaluate: ma_f1 = %.2f, mi_f1=%.2f'%(ma_f1,mi_f1))
    
    model.train()

    return ma_f1

def evaluate_word_level_acc_with_classifier_adapter(model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy with classifiers and adapters for word-level classification tasks

        Return:
         - macro f1 (%)
    '''
    assert params.il_mode == 'CIL', 'NotImplemented for il mode %s'%(params.il_mode)
    model.eval()

    def set_adapter_to_task(t_id):
        try:
            model.set_adapter('task-%d'%(t_id))
        except:
            try:
                model.set_active_adapters('task-%d'%(t_id))
            except:
                raise NotImplementedError()
            
    with torch.no_grad():
        # Only test once because the last test set contains all labels
        gold_lines = []
        pred_lines = []

        for lm_input in eval_data_loader: 
            logits_list = []
            for t_id in range(cur_task_id+1):
                set_adapter_to_task(t_id)
                extracted_features = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input,
                                                    tokenizer=tokenizer)
                # We use the O logits in the first classifier
                logits_list.append(classifier_list[t_id](extracted_features))
            logits = torch.cat(logits_list,dim=-1)

            # Remove the prompt token manually
            if params.PEFT_type == 'PromptTuning':
                logits = logits[:,params.PEFT_num_virtual_tokens:,:] 

            preds = logits.argmax(dim=-1)
            label_idx = lm_input['label_idx_cil']
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            preds, label_idx = preds.cpu(), label_idx.cpu()

            for _sent_gold, _sent_pred in zip(label_idx,preds):
                gold_lines.append([idx2label[_gold.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])
                pred_lines.append([idx2label[_pred.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])

        # micro f1 (average over all instances)
        mi_f1 = f1_score(gold_lines, pred_lines)*100
        # macro f1 (default, average over each class)
        ma_f1 = f1_score(gold_lines, pred_lines, average='macro')*100

        logger.info('Evaluate: ma_f1 = %.2f, mi_f1=%.2f'%(ma_f1,mi_f1))
    
    model.train()

    return ma_f1
# ==================================================================================================================

# ====================== the evaluation metric computed after whole IL training ==========================================================================
def compute_average_acc(result_matrix):
    '''
        Compute the average accuracy (after training the last task) for continual learning

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - aver_acc: float
    '''
    return np.mean(result_matrix[-1,:])

def compute_average_inc_acc(result_matrix):
    '''
        Compute the average incremental accuracy (average over all incremental steps)

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - aver_inc_acc: float
    '''
    NUM_TASK = result_matrix.shape[0]
    # Compute the average acc of each incremental steps
    aver_acc_list = [np.mean(result_matrix[i,:i+1]) for i in range(NUM_TASK)]
    if len(aver_acc_list)==0:
        return -1
    return np.mean(aver_acc_list)

def compute_forgetting(result_matrix):
    '''
        Compute the forgetting for continual learning

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - fgt: float
    '''
    NUM_TASK = result_matrix.shape[0]
    fgt_list = []
    for t_id in range(0,NUM_TASK-1):
        fgt_list.append(np.max(result_matrix[:,t_id])-result_matrix[-1,t_id])

    if len(fgt_list)==0:
        return -1
    return np.mean(fgt_list)

def compute_backward_transfer(result_matrix):
    '''
        Compute the backward transfer for continual learning

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - bwt: float
    '''
    NUM_TASK = result_matrix.shape[0]
    bwt_list = []
    for t_id in range(0,NUM_TASK-1):
        bwt_list.append(result_matrix[-1,t_id]-result_matrix[t_id,t_id])

    if len(bwt_list)==0:
        return -1
    return np.mean(bwt_list)

def compute_forward_transfer(result_matrix, random_result):
    '''
        Compute the forward transfer for continual learning

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step
            - random_result: (NUM_TASK); the result of random guess for each task

        Returns:
            - fwt: float
    '''
    NUM_TASK = result_matrix.shape[0]
    fwt_list = []
    for t_id in range(1,NUM_TASK):
        fwt_list.append(result_matrix[t_id-1,t_id]-random_result[t_id])

    if len(fwt_list)==0:
        return -1
    return np.mean(fwt_list)
# ==================================================================================================================
