from typing import Any
from copy import deepcopy
import numpy as np 
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer

from utils.backbone import obtain_features

def get_buffer(params, continual_config: dict, accelerator: Accelerator):
    return ReplayBuffer(params, continual_config, accelerator)

class ReplayBuffer(object):
    '''
        The class for replay buffer
    '''

    def __init__(self, params, continual_config, accelerator) -> None:
        self.params = params
        self.continual_config = continual_config
        self.accelerator = accelerator

        self.cnt_class_samples = [0 for _ in range(self.continual_config['NUM_CLASS'])]
        self.buffer_data = [[] for _ in range(self.continual_config['NUM_CLASS'])]

    def update_buffer(self, task_id: int, train_data: DataLoader, model: nn.Module, tokenizer: AutoTokenizer) -> None:
        '''
            Update the buffer:


            Args:
                - task_id: evaluation after learning the {task_id}-th task
                - train_data: the training data of the {task_id}-th task
                - model: the backbone of the current training model
                - classifier_list: the classifier of the current training model
                - tokenizer: the tokenizer of the backbone transformers
        '''
        if not self.params.Replay_fix_budge_each_class:
            # OCILNER retain a set of exemplars for each class c K = 5
            if self.params.method == 'OCILNER':
                num_samples_per_class = 5
            else:
                num_samples_per_class = self.params.Replay_buffer_size//self.continual_config['ACCUM_NUM_CLASS'][task_id]  
        else:
            if self.params.method == 'OCILNER':
                num_samples_per_class = 5
            else:
                num_samples_per_class = self.params.Replay_buffer_size//self.continual_config['NUM_CLASS']

        # Reduce buffer for older tasks
        if task_id>0 and not self.params.Replay_fix_budge_each_class:
            for t_id in range(task_id):
                for c_idx in self.continual_config['CUR_CLASS'][t_id]:
                    while self.cnt_class_samples[c_idx]>num_samples_per_class:
                        self.buffer_data[c_idx].pop()
                        self.cnt_class_samples[c_idx] -= 1
            
        # Add new samples from the current task into buffer
        if self.params.Replay_sampling_algorithm == 'random':
            for lm_input in train_data:
                for b_idx in range(lm_input['label_idx_cil'].shape[0]):
                    if self.params.classification_type == 'sentence-level':
                        _label_idx = lm_input['label_idx_cil'][b_idx]
                        if _label_idx not in self.continual_config['CUR_CLASS'][task_id]:
                            continue
                        if self.cnt_class_samples[_label_idx]<num_samples_per_class:
                            self.buffer_data[_label_idx].append(deepcopy({k:v[b_idx] for k,v in lm_input.items()}))
                            self.cnt_class_samples[_label_idx] += 1
                    elif self.params.classification_type == 'word-level':
                        # NOTE: If one entity appear in the sentence, we save the whole sentence as the reserved sample for this entity
                        # Therefore, we ensure that there are at least num_samples_per_class instance for each entity!  
                        _label_idx_list = lm_input['label_idx_cil'][b_idx]
                        for _label_idx in _label_idx_list:
                            if _label_idx not in self.continual_config['CUR_CLASS'][task_id]:
                                continue
                            if self.cnt_class_samples[_label_idx]<num_samples_per_class:
                                self.buffer_data[_label_idx].append(deepcopy({k:v[b_idx] for k,v in lm_input.items()}))
                                self.cnt_class_samples[_label_idx] += 1
                    else:
                        raise NotImplementedError()

        elif self.params.Replay_sampling_algorithm == 'herding':

            assert self.params.classification_type == 'sentence-level', 'NotImplemented for classification type %s'%(self.params.classification_type)

            # Obtain the unshuffled dataset
            unshuffle_train_dataloader = DataLoader(train_data.dataset, 
                                            batch_size=self.params.batch_size, 
                                            shuffle=False,
                                            drop_last=False)

            unshuffle_train_dataloader = self.accelerator.prepare(unshuffle_train_dataloader)

            # Compute the class mean of all training samples from the {task-id} task
            with torch.no_grad():
                
                cur_class_idx = self.continual_config['CUR_CLASS'][task_id]
                features_all, labels_idx_all = [], []

                for lm_input in unshuffle_train_dataloader:
                    extracted_feature = obtain_features(params=self.params, 
                                                        model=model, 
                                                        lm_input=lm_input, 
                                                        tokenizer=tokenizer).clone().detach().cpu()
                    extracted_feature = extracted_feature/torch.norm(extracted_feature,dim=-1,keepdim=True)
                    features_all.append(extracted_feature)
                    label_idx = lm_input['label_idx_cil'].cpu()
                    labels_idx_all.append(label_idx)

                features_all = torch.cat(features_all,dim=0)
                labels_idx_all = torch.cat(labels_idx_all,dim=0)

                # Compute the distance of each samples to the class mean
                for class_idx in cur_class_idx:

                    class_mask = (labels_idx_all==class_idx)
                    local2global_idx = torch.where(class_mask)[0].tolist()
                    features_class = features_all[class_mask]
                    features_class_mean = torch.mean(features_class,dim=0).reshape(1,-1)
                    running_features_sum = torch.zeros_like(features_class_mean)

                    assert self.cnt_class_samples[class_idx] == 0

                    while self.cnt_class_samples[class_idx] < num_samples_per_class:
                        # select sample
                        dist_all = torch.linalg.norm((features_class_mean-(running_features_sum+features_class)/(self.cnt_class_samples[class_idx]+1)),ord=2,dim=-1)
                        min_idx = torch.argmin(dist_all).item()
                        # add into buffer
                        global_idx = local2global_idx[min_idx]
                        self.buffer_data[class_idx].append(deepcopy({k:v.to(model.device) for k,v in unshuffle_train_dataloader.dataset[global_idx].items()}))
                        self.cnt_class_samples[class_idx] += 1
                        # update variable
                        running_features_sum += features_class[min_idx:min_idx+1]
                        features_class[min_idx] = features_class[min_idx]+1e6
        else:
            raise NotImplementedError()

        # check the sum
        if self.params.method != 'OCILNER':
            assert self.len()<=self.params.Replay_buffer_size

    def get_one_batch(self) -> dict:
        assert self.len()>0, 'No buffer!'
        # in experience replay, new data and old data are combined together for each batch
        batch_size = self.params.batch_size // 2
        select_idx = random.choices(list(range(self.len())),k=batch_size)
        lm_input_list = []
        for s_idx in select_idx:
            _s_idx = s_idx
            for c_idx in range(self.continual_config['NUM_CLASS']):
                if _s_idx>=self.cnt_class_samples[c_idx]:
                    _s_idx -= self.cnt_class_samples[c_idx]
                else:
                    lm_input_list.append(self.buffer_data[c_idx][_s_idx])
                    break
        
        buffer_lm_input = {}
        for k,v in lm_input_list[0].items():
            if isinstance(v,torch.Tensor):
                buffer_lm_input[k] = torch.stack([lm_input_list[i][k] for i in range(batch_size)],dim=0)
            elif isinstance(v,str):
                buffer_lm_input[k] = [lm_input_list[i][k] for i in range(batch_size)]
            else:
                print('NOT implemented for combining %s!'%(k))

        return buffer_lm_input
    
    def get_all_data(self):
        assert self.len()>0, 'No buffer!'
        # in experience replay, new data and old data are combined together for each batch

        lm_input_list = []
        for c_idx in range(self.continual_config['NUM_CLASS']):
            lm_input_list.extend(self.buffer_data[c_idx])
        
        buffer_lm_input = {}
        for k,v in lm_input_list[0].items():
            if isinstance(v,torch.Tensor):
                buffer_lm_input[k] = torch.stack([lm_input_list[i][k] for i in range(len(lm_input_list))],dim=0)
            elif isinstance(v,str):
                buffer_lm_input[k] = [lm_input_list[i][k] for i in range(len(lm_input_list))]
            else:
                print('NOT implemented for combinging %s!'%(k))

        return buffer_lm_input

    def len(self) -> int:

        return np.sum(self.cnt_class_samples)