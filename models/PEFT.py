import numpy as np
import logging
import torch
import torch.nn as nn

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_sent_level_acc_with_generation, evaluate_sent_level_acc_with_classifier_adapter, evaluate_word_level_acc_with_classifier_adapter
from models.Base import BaseLearner

logger = logging.getLogger()

def get_PEFT_params(parser):
    '''
        The parameters of model PEFT
    '''
    parser.add_argument("--PEFT_type", type=str, default='PromptTuning', choices=['PromptTuning','LoRA'], help="The peft type")
    
    # For PromptTuning (more details can be found in this (url)[https://huggingface.co/docs/peft]
    parser.add_argument("--PEFT_num_virtual_tokens", type=int, default=10, help="The number of tokens for prompt tuning")
    parser.add_argument("--PEFT_prompt_tuning_init_text", type=str, default='auto', help="The initialization words for prompt tuning")
    # For LoRA (more details can be found in this (url)[https://huggingface.co/docs/peft]
    parser.add_argument("--PEFT_lora_r", type=int, default=4, help="The rank of lora")
    parser.add_argument("--PEFT_lora_alpha", type=int, default=8, help="The scaling of lora")
    parser.add_argument("--PEFT_lora_bias", type=str, default="none", help="The bias of lora")
    parser.add_argument("--PEFT_lora_dropout", type=float, default=0.1, help="The dropout rate of lora")
    parser.add_argument("--PEFT_lora_target_modules", type=str, nargs='+', default=None, help="The target_modules of lora")

class PEFT(BaseLearner):
    '''
        Learning a separate soft prompt/lora for each tasks while fixing the whole PLMs.
        
        For CIL, the logits is the concatnation of the logits of each tasks and thus requires O(num_task) forward passes.

    '''
    def __init__(self, params, CL_dataset, accelerator):
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['None','Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'PEFT')
        assert params.il_mode in ['CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'PEFT')
        assert params.classification_type in ['sentence-level','word-level'], 'NotImplemented for classification type %s'%(params.classification_type)
        assert not params.is_replay, 'NotImplemented for data replay!'

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params, num_task=self.CL_dataset.continual_config['NUM_TASK'])
        num_task=self.CL_dataset.continual_config['NUM_TASK']
        for t_id in range(num_task):
            peft_config = self.model.peft_config['default']
            self.model.add_adapter(adapter_name='task-%d'%(t_id),peft_config=peft_config)

    def build_classifier(self):
        feature_dim = self.model.module.config.hidden_size if hasattr(self.model,'module') else self.model.config.hidden_size
        out_dim_list = self.CL_dataset.continual_config['CUR_NUM_CLASS']
        self.classifier_list = get_classifier(self.params, feature_dim, out_dim_list)
        if self.classifier_list is not None and self.classifier_list != []:
            self.ce_loss = nn.CrossEntropyLoss()

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)
        
    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None

    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, self.classifier_list)
        self.wrap_model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list = list(self.accelerator.prepare(*self.dev_loader_list))
        self.test_loader_list = list(self.accelerator.prepare(*self.test_loader_list))
        if self.classifier_list is not None:
            self.classifier_list = list(self.accelerator.prepare(*self.classifier_list))
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)

        self.model.set_adapter('task-%d'%(task_id))

    def end_task(self, task_id):
        super().end_task(task_id)
    # ==============================================================================================

    # ================================= Epoch-Level Functions =======================================
    def train_epochs(self, task_id):
        '''
            Training the model with serveral epochs
        '''

        total_epochs = self.params.training_epochs

        for epoch_id in range(total_epochs):

            if self.accelerator.is_main_process:
                logger.info("------------------------ epoch %d ------------------------" %(epoch_id+1))

            self.begin_epoch(task_id, epoch_id)

            for lm_input in self.train_loader_list[task_id]:
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

        model.set_adapter('task-%d'%(task_id))
        # Compute loss
        # Training with Causal Language Modeling Loss
        if self.params.classifier == 'None':
            assert self.params.classification_type == 'sentence-level'
            total_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                    'attention_mask':lm_input['attention_mask_with_ans'],
                                    'labels':lm_input['labels_with_ans']}).loss
            
        elif self.params.classifier in ['Linear','CosineLinear']:
            extracted_feature = obtain_features(params=self.params, 
                                                    model=model, 
                                                    lm_input=lm_input, 
                                                    tokenizer=self.tokenizer)
            logits = self.classifier_list[task_id](extracted_feature)
            
            if self.params.classification_type == 'sentence-level':
                label_idx = lm_input['label_idx_til'] # the TIL label is used since the adapter is set
            elif self.params.classification_type == 'word-level':
                assert self.params.il_mode == 'CIL'
                assert self.params.PEFT_type in ['LoRA','PromptTuning'], 'NotImplemented for PEFT type %s'%(self.params.PEFT_type)

                # Remove the prompt token manually
                if self.params.PEFT_type == 'PromptTuning':
                    logits = logits[:,self.params.PEFT_num_virtual_tokens:,:] 

                # Compute the logits of O classes
                if task_id>0:
                    with torch.no_grad():
                        model.set_adapter('task-%d'%(0))
                        extracted_feature = obtain_features(params=self.params, 
                                                    model=model, 
                                                    lm_input=lm_input, 
                                                    tokenizer=self.tokenizer)
                        model.set_adapter('task-%d'%(task_id))
                        logits_first_task = self.classifier_list[0](extracted_feature)
                        # Remove the prompt token manually
                        if self.params.PEFT_type == 'PromptTuning':
                            logits_first_task = logits_first_task[:,self.params.PEFT_num_virtual_tokens:,:] 
                        logits_O = logits_first_task[:,:,0:1]
                    logits = torch.cat((logits_O,logits),dim=-1)

                logits = logits.reshape(-1,logits.shape[-1])
                label_idx = lm_input['label_idx_cil'].reshape(-1)
                # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
                label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
                label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
                
                # Convert to til label (starting from 1, we preserve 0 as "O" class)
                if task_id>0:
                    label_idx[label_idx>0] = label_idx[label_idx>0] - self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id] + 1
            else:
                raise NotImplementedError()

            total_loss = self.ce_loss(logits,label_idx)
        else:
            raise NotImplementedError()

        # Backward
        self.model.train()
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
            acc = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            if self.accelerator.is_main_process:
                logger.info("%s, Current Task %d, Epoch %d, Step %d: Dev_acc=%.3f" % (
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
        assert il_mode in ['CIL','TIL'], 'NotImplemented for il_mode %s'%(il_mode)
        if phase=='train':
            data_loader = self.train_loader_list
        elif phase=='dev':
            data_loader = self.dev_loader_list
        else:
            data_loader = self.test_loader_list

        if self.params.classifier == 'None':
            
            assert self.params.classification_type == 'sentence-level'

            if self.params.il_mode=='CIL':
                self.model.set_adapter('task-%d'%(cur_task_id))
            elif self.params.il_mode=='TIL':
                self.model.set_adapter('task-%d'%(eval_task_id))
            else:
                raise NotImplementedError()

            acc = evaluate_sent_level_acc_with_generation(
                model=self.model,
                eval_data_loader=data_loader[eval_task_id],
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config['idx2label']
            )

            return  acc

        elif self.params.classifier in ['Linear','CosineLinear']:

            if self.params.classification_type == 'sentence-level':
                acc = evaluate_sent_level_acc_with_classifier_adapter(
                    model=self.model,
                    classifier_list=self.classifier_list,
                    cur_task_id=cur_task_id if self.params.il_mode=='CIL' else eval_task_id,
                    eval_data_loader=data_loader[eval_task_id],
                    tokenizer=self.tokenizer,
                    accelerator=self.accelerator,
                    params=self.params,
                    idx2label=self.CL_dataset.continual_config['idx2label']
                )
            elif self.params.classification_type == 'word-level':
                acc = evaluate_word_level_acc_with_classifier_adapter(
                    model=self.model,
                    classifier_list=self.classifier_list,
                    cur_task_id=cur_task_id if self.params.il_mode=='CIL' else eval_task_id,
                    eval_data_loader=data_loader[eval_task_id],
                    tokenizer=self.tokenizer,
                    accelerator=self.accelerator,
                    params=self.params,
                    idx2label=self.CL_dataset.continual_config['idx2label']
                )
            else:
                raise NotImplementedError()

            return acc
        
        else:
            NotImplementedError()
    # ===========================================================================================