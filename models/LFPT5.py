import numpy as np
import logging
import torch
import torch.nn as nn
from typing import List
from copy import deepcopy
from datasets import Dataset
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F

from utils.metric import ResultSummary
from utils.backbone import get_backbone
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader, preprocess_function_train_generative_LAMOL
from utils.buffer import get_buffer
from utils.evaluation import evaluate_sent_level_acc_with_generation
from models.Base import BaseLearner

logger = logging.getLogger()

def get_LFPT5_params(parser):
    '''
        The parameters of model LFPT5
    '''

    parser.add_argument("--LFPT5_distill_weight", type=float, default=0.01, help="The weight of the distillation term in LAMOL_KD")
    parser.add_argument("--LFPT5_lambda", type=float, default=0.0, help="The weight of the generation target in LAMOL_KD")
    parser.add_argument("--LFPT5_gamma", type=float, default=0.20, help="The ratio of psesudo old samples w.r.t the training data of new task.")
    parser.add_argument("--LFPT5_topk", type=int, default=20, help="The top-k sampling for generating psesudo old samples.")
    parser.add_argument("--LFPT5_answertoken_weight", type=float, default=100, help="The weight of answer token in cross-entropy loss.")
    parser.add_argument("--LAMOL_use_task_specific_gen_token", type=bool, default=True, help="If using task-specific generation token for generating psesudo old samples.")

    parser.add_argument("--PEFT_type", type=str, default='PromptTuning', choices=['PromptTuning'], help="The peft type")
    
    # For PromptTuning (more details can be found in this (url)[https://huggingface.co/docs/peft]
    parser.add_argument("--PEFT_num_virtual_tokens", type=int, default=10, help="The number of tokens for prompt tuning")
    parser.add_argument("--PEFT_prompt_tuning_init_text", type=str, default='auto', help="The initialization words for prompt tuning")

class LFPT5(BaseLearner):
    '''
        LFPT5: We further utilize knowledge distillation based on LAMOL. 
        Besides, LFPT5 utilize prompt tuning instead of fine-tuning the whole model.
        Since LFPT5 is originally design for Lifelong Few-shot Language Learning 
        which can be regraded as a combination of both CIL and TIL.
        In this file, we implement the CIL and TIL mode of LFPT5 separately.
        Furthermore, we extend LFPT5 to decoder-only backbones for comparing with other baselines.
        This repository is implemented according to the [official released repository](https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning.git).

        - [LFPT5: A UNIFIED FRAMEWORK FOR LIFELONG FEW-SHOT LANGUAGE LEARNING BASED ON PROMPT TUNING OF T5](https://openreview.net/forum?id=7mozamSFNt4)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)

        assert params.classifier in ['None'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'LFPT5')
        assert params.il_mode in ['IIL','CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'LFPT5')
        assert params.classification_type == 'sentence-level', 'NotImplemented for classification type %s'%(params.classification_type)
        assert params.backbone_type == 'generative', 'NotImplemented for backbone type %s'%(params.backbone_type)
        assert not params.is_replay, 'NotImplemented for is_replay = %s'%(params.is_replay)

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)
        self.ce_loss = None
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        # In TIL mode, we need a prompt for each task.
        # In CIL mode, we only use the default prompt for all tasks.
        if self.params.il_mode == 'TIL':
            num_task=self.CL_dataset.continual_config['NUM_TASK']
            for t_id in range(num_task):
                peft_config = self.model.peft_config['default']
                self.model.add_adapter(adapter_name='task-%d'%(t_id),peft_config=peft_config)

    def build_classifier(self):
        self.classifier_list = None

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
    
    def accelerate_prepare(self):
        self.model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
        self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
        self.teacher_model = None
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)
        
        # Prepare teacher model
        if task_id>0:
            if self.teacher_model is not None:
                self.teacher_model.cpu()
                del self.teacher_model
            self.teacher_model = deepcopy(self.model)

        if self.params.il_mode == 'TIL':
            self.model.set_adapter('task-%d'%(task_id))

    def end_task(self, task_id):
        super().end_task(task_id)

    # ==============================================================================================

    # ================================= Epoch-Level Functions =======================================
    def train_epochs(self, task_id):
        '''
            Training the model with serveral epochs
        '''

        if task_id>0:
            train_dataset = self.train_loader_list[task_id].dataset
            pseudo_buf_dataset_list = self.generate_pseudo_buffer_samples(task_id=task_id,
                                                                     num_samples=int(len(train_dataset)*self.params.LFPT5_gamma))
            cur_train_loader = DataLoader(
                ConcatDataset((train_dataset,*pseudo_buf_dataset_list)),
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False
            )
            cur_train_loader = self.accelerator.prepare(cur_train_loader)
        else:
            cur_train_loader = self.train_loader_list[task_id]

        total_epochs = self.params.training_epochs

        for epoch_id in range(total_epochs):

            if self.accelerator.is_main_process:
                logger.info("------------------------ epoch %d ------------------------" %(epoch_id+1))

            self.begin_epoch(task_id, epoch_id)

            for lm_input in cur_train_loader:
                self.observe_batch(task_id, epoch_id, lm_input) 

            self.end_epoch(task_id, epoch_id)

    def begin_epoch(self, task_id, epoch_id):
        '''
            Start of each epoch
        '''
        # Avoid overwrite the result of the same global step
        if ((self.params.evaluate_interval>0) and (epoch_id>0 and epoch_id%self.params.evaluate_interval==0)) or \
            (task_id==0 and epoch_id==0):
            self.evaluate_model(task_id=task_id)
        self.loss_list = []
        self.model.train()

    def observe_batch(self, task_id, epoch_id, lm_input):
        '''
            Observe a batch of data
        '''
        # Update step
        self.step += 1
        self.global_step += 1

        # For Distributed Data Parallel
        if hasattr(self.model,'module'):
            model = self.model.module
        else:
            model = self.model
        if hasattr(self.teacher_model,'module'):
            teacher_model = self.teacher_model.module
        else:
            teacher_model = self.teacher_model

        if self.params.il_mode == 'TIL':
            model.set_adapter('task-%d'%(task_id))


        # Compute loss
        # Training with Causal Language Modeling Loss
        if task_id == 0:
            qa_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                    'attention_mask':lm_input['attention_mask_with_ans'],
                                    'labels':lm_input['labels_with_ans']}).loss

            generation_loss = model(**{'input_ids':lm_input['input_ids_with_gen_ans'], 
                                    'attention_mask':lm_input['attention_mask_with_gen_ans'],
                                    'labels':lm_input['labels_with_gen_ans']}).loss

            # if self.ce_loss is None:
            #     ce_weight = torch.tensor([1.]*generation_logits.shape[-1]).to(model.device)
            #     for anstoken_id in self.tokenizer('__ans__').input_ids:
            #         ce_weight[anstoken_id] = self.params.LFPT5_answertoken_weight
            #     self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)

            # Remove the prompt and shift logits
            # shift_logits = generation_logits[:, self.params.PEFT_num_virtual_tokens:-1, :].contiguous()
            # labels = lm_input['labels_with_gen_ans'][:, 1:].contiguous()
            # Reweight the answer token
            # We find that the model hard to learn to generate answer token '__ans__' when using prompt tuning
            # generation_loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

            # TODO: DEBUG
            total_loss = qa_loss + self.params.LFPT5_lambda*generation_loss
            # total_loss = generation_loss

        else:

            cur_mask = (lm_input['label_idx_cil']!=-1)
            if torch.sum(cur_mask)==0:
                total_loss_cur = torch.tensor(0.).to(model.device)
            else:
                # For current task's data (causal language modeling)
                qa_loss_cur = model(**{'input_ids':lm_input['input_ids_with_ans'][cur_mask], 
                                        'attention_mask':lm_input['attention_mask_with_ans'][cur_mask],
                                        'labels':lm_input['labels_with_ans'][cur_mask]}).loss
                # TODO: DEBUG
                generation_loss_cur = model(**{'input_ids':lm_input['input_ids_with_gen_ans'][cur_mask], 
                                        'attention_mask':lm_input['attention_mask_with_gen_ans'][cur_mask],
                                        'labels':lm_input['labels_with_gen_ans'][cur_mask]}).loss
                # shift_logits = generation_logits[:, self.params.PEFT_num_virtual_tokens:-1, :].contiguous()
                # labels = lm_input['labels_with_gen_ans'][:, 1:].contiguous()
                # Reweight the answer token
                # We find that the model hard to learn to generate answer token '__ans__' when using prompt tuning
                # generation_loss_cur = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                total_loss_cur = qa_loss_cur + self.params.LFPT5_lambda*generation_loss_cur 

            buf_mask = (lm_input['label_idx_cil']==-1)
            if torch.sum(buf_mask)==0:
                total_loss_buf = torch.tensor(0.).to(model.device)
            else:
                # For pseudo buffer data (knowledge distillation)
                with torch.no_grad():
                    qa_teacher_logits = teacher_model(**{'input_ids':lm_input['input_ids_with_ans'][buf_mask], 
                                        'attention_mask':lm_input['attention_mask_with_ans'][buf_mask],
                                        'labels':lm_input['labels_with_ans'][buf_mask]}).logits
                    qa_teacher_logits = qa_teacher_logits[lm_input['labels_with_ans'][buf_mask]!=-100]

                qa_student_logits = model(**{'input_ids':lm_input['input_ids_with_ans'][buf_mask], 
                                        'attention_mask':lm_input['attention_mask_with_ans'][buf_mask],
                                        'labels':lm_input['labels_with_ans'][buf_mask]}).logits
                qa_student_logits = qa_student_logits[lm_input['labels_with_ans'][buf_mask]!=-100]

                qa_loss_buf = self.kl_loss(F.log_softmax(qa_student_logits,dim=-1),
                                        F.softmax(qa_teacher_logits,dim=-1))

                total_loss_buf = qa_loss_buf

            total_loss = total_loss_cur + self.params.LFPT5_distill_weight*total_loss_buf

        # Backward
        model.train()
        self.optimizer.zero_grad()        
        self.accelerator.backward(total_loss)
        
        scalar_loss = total_loss.item()
        if not(np.isnan(scalar_loss)) or not(np.isinf(scalar_loss)):
            self.optimizer.step()
            self.loss_list.append(scalar_loss)

        # Print training information
        if self.params.info_per_steps and self.step%self.params.info_per_steps==0:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info("Epoch %d, Step %d: Total_loss=%.3f,"%(
                        epoch_id+1, self.step, mean_loss
                ))
            self.accelerator.log({'loss':mean_loss},step=self.global_step)

    def end_epoch(self, task_id, epoch_id):
        '''
            End of each epoch
        '''
        # Print training information
        if len(self.loss_list)>0:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info("Epoch %d, Step %d: Total_loss=%.3f"%(
                            epoch_id+1, self.step, mean_loss
                    ))
            self.accelerator.log({'loss':mean_loss},step=self.global_step)
            
        # For evaluation
        if (self.params.evaluate_interval>0) and epoch_id%self.params.evaluate_interval==0:
            il_mode = self.params.il_mode
            if il_mode == 'IIL':
                acc, save_dict = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            else:
                acc = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            if self.accelerator.is_main_process:
                logger.info("Mode %s, Current Task %d, Epoch %d, Step %d: Dev_acc=%.3f" % (
                    il_mode, task_id, epoch_id+1, self.step, acc
                ))
            self.accelerator.log({'Dev_Acc_Task_%d'%(task_id):acc},step=self.global_step)
            dev_score = acc

            if dev_score > self.best_score:
                if self.accelerator.is_main_process:
                    logger.info("Find better model!!")

        # Saving GPU memory
        torch.cuda.empty_cache()
    # ===========================================================================================


    # ================== Evaluation, Logging, Saving and Loading Functions ======================
    def evaluate_current_task(self,
                                eval_task_id: int, 
                                cur_task_id: int, 
                                phase: str,
                                il_mode: str) -> dict:
        '''
            Evaluate the model on the current task

            Params: 
                - eval_task_id: the id of the task to be evaluated, 
                this information should NOT be provided to the CIL model during inference!
                - cur_task_id: the id recording how many tasks the model has learned,
                this information can be provided to the CIL model during inference.
                - phase: 'train','dev'or'test'
                - il_mode: 'CIL' or 'TIL'

            Return:
                - acc: CIL accuracy (%) or 'TIL': TIL accuracy (%)
        '''

        assert phase in ['train','test','dev']
        if phase=='train':
            data_loader = self.train_loader_list
        elif phase=='dev':
            data_loader = self.dev_loader_list
        else:
            data_loader = self.test_loader_list

        # For Distributed Data Parallel
        if hasattr(self.model,'module'):
            model = self.model.module
        else:
            model = self.model

        if self.classifier_list is None:

            if self.params.il_mode=='TIL':
                self.model.set_adapter('task-%d'%(eval_task_id))
            
            acc = evaluate_sent_level_acc_with_generation(
                model=model,
                eval_data_loader=data_loader[eval_task_id],
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config['idx2label']
            )

            return  acc

        else:
            raise NotImplementedError()
    # ===========================================================================================

    # ======================== Other Model-Specific Functions ===================================
    def generate_pseudo_buffer_samples(self, task_id: int, num_samples: int) -> List[Dataset]:
        '''
            Generate pseudo old samples with generative models

            Args:
                - task_id: the current task id
                - num_samples: the number of samples to be generated

            Return:
                pseudo_dataset_list
        '''

        # For Distributed Data Parallel
        if hasattr(self.model,'module'):
            model = self.model.module
        else:
            model = self.model

        input_column = 'input'
        target_column = 'target'
        ans_token = '__ans__'
        eos_token = self.tokenizer.eos_token
        num_task = self.CL_dataset.continual_config['NUM_TASK']

        pseudo_dataset_list = []
        
        generate_batch_size = 8

        with torch.no_grad():
        
            for t_id in range(task_id):

                if self.params.il_mode == 'IIL':
                    pesudo_samples_dict = {
                        'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': [],
                        'instance_id': [], 'concept_id': [], 'relation_id': [], 
                    }
                else:
                        pesudo_samples_dict = {
                        'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': []
                    }

                cnt_num_samples = num_samples//task_id

                gen_token= '__%d__'%(t_id) if self.params.LAMOL_use_task_specific_gen_token else '__gen__'

                while cnt_num_samples > 0:
                        
                    generate_num = generate_batch_size if cnt_num_samples>=generate_batch_size else cnt_num_samples

                    lm_input = self.tokenizer([gen_token for _ in range(generate_num)], # Add a space after generation token
                                                padding='longest',
                                                return_tensors='pt')
                    lm_input = {k:v.to(model.device) for k,v in lm_input.items()}
                    
                    max_input_len = np.max([len(lm_input['input_ids'][i]) for i in range(generate_num)])

                    generate_ids_all = model.generate(**lm_input, 
                                            max_new_tokens=self.params.max_seq_length-max_input_len, 
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            do_sample=True,
                                            top_k=self.params.LFPT5_topk) 
                    generate_ids = generate_ids_all[:,max_input_len:].contiguous()
                    generated_samples = self.tokenizer.batch_decode(generate_ids)

                    for _one_sample in generated_samples:
                        if _one_sample.count('__ans__')!=1:
                            continue
                        _question, _answer = _one_sample.split('__ans__')
                        _answer = _answer.replace(self.tokenizer.eos_token,'')
                        pesudo_samples_dict['input'].append(_question)
                        pesudo_samples_dict['target'].append(_answer)
                        pesudo_samples_dict['label_idx_cil'].append(-1)
                        pesudo_samples_dict['label_idx_til'].append(-1)
                        if self.params.il_mode == 'IIL':
                            pesudo_samples_dict['instance_id'].append(-1)
                            pesudo_samples_dict['concept_id'].append(-1)
                            pesudo_samples_dict['relation_id'].append(-1)
                        
                    cnt_num_samples -= generate_num
            
                if len(pesudo_samples_dict['input'])==0:
                    logger.info('No pseudo samples are generated in the correct format for task %d!'%(t_id+1))
                    continue
                pseudo_dataset = Dataset.from_dict(pesudo_samples_dict)
                pseudo_dataset = pseudo_dataset.map(preprocess_function_train_generative_LAMOL, 
                                                    batched=True, 
                                                    desc='Generate pseudo samples for task %d'%(t_id+1), 
                                                    batch_size=1000,
                                                    fn_kwargs={
                                                        'params':self.params,
                                                        'tokenizer':self.tokenizer,
                                                        'num_task':num_task,
                                                        'task_id':t_id,
                                                        'input_column':input_column,
                                                        'target_column':target_column,
                                                        'ans_token':ans_token,
                                                        'eos_token':eos_token,
                                                        'gen_token':gen_token,
                                                    })
                if self.params.il_mode == 'IIL':
                    pseudo_dataset.set_format(type='torch', 
                                              columns=['input_ids','attention_mask','label_idx_cil',
                                                       'input_ids_with_ans', 'attention_mask_with_ans', 'labels_with_ans', 
                                                       'input_ids_with_gen_ans', 'attention_mask_with_gen_ans', 'labels_with_gen_ans',
                                                       'target', 'instance_id', 'concept_id', 'relation_id'])
                else:
                    pseudo_dataset.set_format(type='torch', 
                                              columns=['input_ids','attention_mask','label_idx_cil',
                                                       'input_ids_with_ans', 'attention_mask_with_ans', 'labels_with_ans', 
                                                       'input_ids_with_gen_ans', 'attention_mask_with_gen_ans', 'labels_with_gen_ans'])

                pseudo_dataset_list.append(pseudo_dataset)

        return pseudo_dataset_list

    # ===========================================================================================