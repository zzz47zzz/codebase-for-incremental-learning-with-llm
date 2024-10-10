import numpy as np
import logging
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from datasets import Dataset

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.buffer import get_buffer
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_sent_level_acc_with_generation, evaluate_sent_level_acc_with_classifier, evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_SEQ_params(parser):
    '''
        The parameters of model SEQ and WarmupAndFix
    '''
    parser.add_argument("--SEQ_peft_type", type=str, default='None', choices=['None','PromptTuning','LoRA'], help="The PEFT type")
    parser.add_argument("--SEQ_fix_old_classifier", default=False, type=bool, help="if fix the old classifiers")
    parser.add_argument("--SEQ_fix_encoder", default=False, type=bool, help="if fix the encoder")
    parser.add_argument("--SEQ_warmup_epoch_before_fix_encoder", type=int, default=0, help="The number of fine-tuning epoch before fixing the encoder (only for the first task). NOTE: Only used when SEQ_fix_encoder is True")
    parser.add_argument("--SEQ_warmup_target", type=str, default='causal-lm', choices=['causal-lm','ce-loss'], help="The training target during warmup epochs (only for the first task). NOTE: Only used when SEQ_fix_encoder is True")
    parser.add_argument("--SEQ_preallocated_classifier", type=bool, default=False, help="If pre-allocating classifiers for optimization?")
    parser.add_argument("--SEQ_use_prototype_for_prediction", type=bool, default=False, help="If using prototype as classifier weights")

    parser.add_argument("--PEFT_type", type=str, default='None', choices=['None','PromptTuning','LoRA'], help="The peft type")
    
    # For PromptTuning (more details can be found in this (url)[https://huggingface.co/docs/peft]
    parser.add_argument("--PEFT_num_virtual_tokens", type=int, default=10, help="The number of tokens for prompt tuning")
    parser.add_argument("--PEFT_prompt_tuning_init_text", type=str, default='auto', help="The initialization words for prompt tuning")
    # For LoRA (more details can be found in this (url)[https://huggingface.co/docs/peft]
    parser.add_argument("--PEFT_lora_r", type=int, default=8, help="The rank of lora")
    parser.add_argument("--PEFT_lora_alpha", type=int, default=16, help="The scaling of lora")
    parser.add_argument("--PEFT_lora_bias", type=str, default="none", help="The bias of lora")
    parser.add_argument("--PEFT_lora_dropout", type=float, default=0.05, help="The dropout rate of lora")
    parser.add_argument("--PEFT_lora_target_modules", type=str, nargs='+', default=None, help="The target_modules of lora")

class SEQ(BaseLearner):
    '''
        Training a model sequentially to a number of tasks
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['None','Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'SEQ')
        assert params.il_mode in ['IIL','CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'SEQ')
        assert params.classification_type in ['sentence-level','word-level'], 'NotImplemented for classification type %s'%(params.classification_type)
        if params.SEQ_use_prototype_for_prediction:
            assert params.classifier in ['Linear','CosineLinear']

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)

    def build_classifier(self):
        feature_dim = self.model.module.config.hidden_size if hasattr(self.model,'module') else self.model.config.hidden_size
        out_dim_list = self.CL_dataset.continual_config['CUR_NUM_CLASS']
        self.classifier_list = get_classifier(self.params, feature_dim, out_dim_list)
        if self.classifier_list is not None:
            self.ce_loss = nn.CrossEntropyLoss()

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)
        # Mapping ENCODER/CLASSIFIER and task_id to the index in the param_groups of the optimizer
        num_task = self.CL_dataset.continual_config['NUM_TASK']
        if self.params.classifier!='None' and (not (self.params.SEQ_fix_encoder and self.params.SEQ_warmup_epoch_before_fix_encoder==0)):
            assert len(self.optimizer.param_groups) == 1+num_task
            self.param_groups_mapping = {}
            self.param_groups_mapping['encoder'] = 0
            self.param_groups_mapping['classifier'] = {i:1+i for i in range(num_task)}
        elif self.params.classifier!='None' and (self.params.SEQ_fix_encoder and self.params.SEQ_warmup_epoch_before_fix_encoder==0):
            assert len(self.optimizer.param_groups) == num_task
            self.param_groups_mapping = {}
            self.param_groups_mapping['classifier'] = {i:i for i in range(num_task)}
        else:
            assert len(self.optimizer.param_groups) == 1
            self.param_groups_mapping = {}
            self.param_groups_mapping['encoder'] = 0

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
        if self.params.is_replay:
            self.buffer = get_buffer(self.params,self.CL_dataset.continual_config,self.accelerator)
    
    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, self.classifier_list)
        self.wrap_model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        if len(self.dev_loader_list)>1:
            self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
            self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
        else:
            self.dev_loader_list = [self.accelerator.prepare(*self.dev_loader_list)]
            self.test_loader_list = [self.accelerator.prepare(*self.test_loader_list)]
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)

        # Fix a component by setting its learning rate as zero
        if task_id>0 and self.params.classifier!='None' and self.params.SEQ_fix_old_classifier:
            self.optimizer.param_groups[self.param_groups_mapping['classifier'][task_id-1]]['lr']=0
        
    def end_task(self, task_id):
        super().end_task(task_id)

        if self.params.is_replay:
            # For Distributed Data Parallel
            if hasattr(self.wrap_model,'module'):
                wrap_model = self.wrap_model.module
            else:
                wrap_model = self.wrap_model
            model = wrap_model.model

            self.buffer.update_buffer(task_id, self.train_loader_list[task_id], model, self.tokenizer)
    # ==============================================================================================

    # ================================= Epoch-Level Functions =======================================
    def train_epochs(self, task_id):
        '''
            Training the model with serveral epochs
        '''

        if task_id>0 and self.params.is_replay and not self.params.Replay_batch_level:
            train_dataset = self.train_loader_list[task_id].dataset
            buf_dataset = Dataset.from_dict(self.buffer.get_all_data())
            buf_dataset.set_format(type='torch')
            cur_train_loader = DataLoader(
                ConcatDataset((train_dataset,buf_dataset)),
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False
            )
            cur_train_loader = self.accelerator.prepare(cur_train_loader)
        else:
            cur_train_loader = self.train_loader_list[task_id]

        total_epochs = self.params.training_epochs
        # add epochs for warming up with causal lm loss
        if task_id==0 and self.params.SEQ_fix_encoder:
            total_epochs += self.params.SEQ_warmup_epoch_before_fix_encoder

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

        # Fix a component by setting its learning rate as zero
        if task_id==0 and self.params.SEQ_fix_encoder and epoch_id==self.params.SEQ_warmup_epoch_before_fix_encoder:
            if 'encoder' in self.param_groups_mapping.keys():
                self.optimizer.param_groups[self.param_groups_mapping['encoder']]['lr']=0
                for name, param in self.model.named_parameters():
                    param.requires_grad = False

    def observe_batch(self, task_id, epoch_id, lm_input):
        '''
            Observe a batch of data
        '''
        # Update step
        self.step += 1
        self.global_step += 1

        # Sample from buffer and combine old data with new data
        if task_id>0 and self.params.is_replay and self.params.Replay_batch_level:
            buffer_lm_input = self.buffer.get_one_batch()
            lm_input = {k:torch.cat([lm_input[k],buffer_lm_input[k].to(lm_input[k].device)],dim=0) 
                        for k in lm_input.keys() if k in buffer_lm_input.keys()}
            
        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            wrap_model = self.wrap_model.module
        else:
            wrap_model = self.wrap_model

        model = wrap_model.model
        classifier_list = wrap_model.classifier_list

        # Compute loss
        # Training with Causal Language Modeling Loss
        if self.params.classifier == 'None' or \
            (task_id==0 and self.params.SEQ_fix_encoder and epoch_id<self.params.SEQ_warmup_epoch_before_fix_encoder and self.params.SEQ_warmup_target=='causal-lm'):
            total_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                    'attention_mask':lm_input['attention_mask_with_ans'],
                                    'labels':lm_input['labels_with_ans']}).loss
        # Training with Cross-Entropy Loss
        elif self.params.classifier in ['Linear','CosineLinear'] and \
            ((not self.params.SEQ_use_prototype_for_prediction) or \
            (self.params.SEQ_use_prototype_for_prediction and task_id==0 and self.params.SEQ_fix_encoder and epoch_id<self.params.SEQ_warmup_epoch_before_fix_encoder and self.params.SEQ_warmup_target=='ce-loss')):
            extracted_feature = obtain_features(params=self.params, 
                                                model=model, 
                                                lm_input=lm_input, 
                                                tokenizer=self.tokenizer)
            if self.params.il_mode == 'CIL':

                if self.params.SEQ_preallocated_classifier:
                    logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list],dim=-1)
                else:
                    logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list[:task_id+1]],dim=-1)
                
                if self.params.classification_type == 'sentence-level':
                    label_idx = lm_input['label_idx_cil']
                elif self.params.classification_type == 'word-level':
                    logits = logits.reshape(-1,logits.shape[-1])
                    label_idx = lm_input['label_idx_cil'].reshape(-1)
                    # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
                    label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
                    if not self.params.is_replay:
                        label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
                else:
                    raise NotImplementedError()
                
                total_loss = self.ce_loss(logits,label_idx)
                
            elif self.params.il_mode == 'TIL':

                total_loss = torch.tensor(0.).to(model.device)

                for t_id in range(task_id+1):

                    class_idx_range_bg = (self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][t_id])
                    class_idx_range_ed = (self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][t_id])
                    task_mask = torch.logical_and(
                        lm_input['label_idx_cil']>=class_idx_range_bg,
                        lm_input['label_idx_cil']<class_idx_range_ed
                    )
                    if task_mask.sum().item()==0:
                        continue

                    logits = classifier_list[t_id](extracted_feature[task_mask])

                    if self.params.classification_type == 'sentence-level':
                        label_idx = lm_input['label_idx_til'][task_mask]
                    else:
                        raise NotImplementedError()
            
                    total_loss += self.ce_loss(logits,label_idx)

            else:
                raise NotImplementedError()
            
            
        elif self.params.SEQ_use_prototype_for_prediction:
            # Skip training
            return None
        else:
            raise NotImplementedError()

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

        # Update the classifier weight using class mean features
        if self.params.SEQ_use_prototype_for_prediction and \
            (not (task_id==0 and self.params.SEQ_fix_encoder and epoch_id<self.params.SEQ_warmup_epoch_before_fix_encoder)):
            # NOTE: We do not update the classifier weight during warmup stage!
            self.update_prototype_weight(task_id)
            
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
                - il_mode: 'IIL' or 'CIL' or 'TIL'

            Return:
                - acc: CIL accuracy (%) or 'TIL': TIL accuracy (%)
        '''

        assert phase in ['train','test','dev']
        assert il_mode in ['IIL','CIL','TIL'], 'NotImplemented for il_mode %s'%(il_mode)
        if phase=='train':
            data_loader = self.train_loader_list
        elif phase=='dev':
            data_loader = self.dev_loader_list
        else:
            data_loader = self.test_loader_list

        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            wrap_model = self.wrap_model.module
        else:
            wrap_model = self.wrap_model
        
        model = wrap_model.model
        classifier_list = wrap_model.classifier_list

        if classifier_list is None:

            # NOTE: When not using classifier, the SEQ model does not need (benefit from) task identity 
            acc = evaluate_sent_level_acc_with_generation(
                model=model,
                eval_data_loader=data_loader[eval_task_id],
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config['idx2label']
            )

            return  acc
        
        elif self.params.classifier in ['Linear','CosineLinear']:

            if self.params.classification_type == 'sentence-level':
                acc = evaluate_sent_level_acc_with_classifier(
                    model=model,
                    classifier_list=classifier_list,
                    cur_task_id=cur_task_id if self.params.il_mode=='CIL' else eval_task_id,
                    eval_data_loader=data_loader[eval_task_id],
                    tokenizer=self.tokenizer,
                    accelerator=self.accelerator,
                    params=self.params,
                    idx2label=self.CL_dataset.continual_config['idx2label']
                )
            elif self.params.classification_type == 'word-level':
                acc = evaluate_word_level_acc_with_classifier(
                    model=model,
                    classifier_list=classifier_list,
                    cur_task_id=cur_task_id,
                    eval_data_loader=data_loader[eval_task_id],
                    tokenizer=self.tokenizer,
                    accelerator=self.accelerator,
                    params=self.params,
                    idx2label=self.CL_dataset.continual_config['idx2label']
                )
            else:
                raise NotImplementedError()

            return  acc

        else:
            raise NotImplementedError()
    # ===========================================================================================

    # ======================== Other Model-Specific Functions ===================================
    def update_prototype_weight(self, task_id):
        '''
            Compute the class mean features and use them as the weight of the CURRENT classifier.

            NOTE: The old classifier is fix since prototype classifier can not be trained.
        '''
        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            wrap_model = self.wrap_model.module
        else:
            wrap_model = self.wrap_model
        
        model = wrap_model.model
        classifier_list = wrap_model.classifier_list

        cur_class_prototypes = torch.zeros_like(classifier_list[task_id].weight.data).to(model.device)
        cur_class_idx = self.CL_dataset.continual_config['CUR_CLASS'][task_id]
        cnt_class_samples = {class_idx:0 for class_idx in cur_class_idx}

        # Compute the class mean of all training samples
        with torch.no_grad():
            for lm_input in self.train_loader_list[task_id]:
                extracted_feature = obtain_features(params=self.params, 
                                                    model=model, 
                                                    lm_input=lm_input, 
                                                    tokenizer=self.tokenizer)
                logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list[:task_id+1]],dim=-1)
                if self.params.classification_type == 'sentence-level':
                    label_idx = lm_input['label_idx_cil']
                elif self.params.classification_type == 'word-level':
                    extracted_feature = extracted_feature.reshape(-1,extracted_feature.shape[-1])
                    logits = logits.reshape(-1,logits.shape[-1])
                    label_idx = lm_input['label_idx_cil'].reshape(-1)
                    # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
                    label_idx = label_idx.masked_fill(label_idx>=logits.shape[-1],0) 
                    if not self.params.is_replay:
                        label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
                else:
                    raise NotImplementedError()
                
                for class_idx in cur_class_idx:
                    if class_idx not in label_idx:
                        continue
                    class_mask = (label_idx==class_idx)
                    cnt_class_samples[class_idx] += class_mask.sum().item()
                    cur_class_prototypes[class_idx-cur_class_idx[0]] += extracted_feature[class_mask].sum(dim=0)

        # Set the weight of the current classifier
        for class_idx in cur_class_idx:
            if cnt_class_samples[class_idx]==0:
                continue
            classifier_list[task_id].weight.data[class_idx-cur_class_idx[0]] = cur_class_prototypes[class_idx-cur_class_idx[0]]/cnt_class_samples[class_idx]