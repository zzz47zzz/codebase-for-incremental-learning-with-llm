import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.wrapmodel import WrapModel
from utils.buffer import get_buffer
from utils.evaluation import evaluate_sent_level_acc_with_generation, evaluate_sent_level_acc_with_classifier, evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_CLSER_params(parser):
    '''
        The parameters of model CLSER
    '''
    parser.add_argument("--CLSER_alpha_fast", type=float, default=0.999, help="The momentum of the fast model")
    parser.add_argument("--CLSER_alpha_slow", type=float, default=0.999, help="The momentum of the slow model")
    parser.add_argument("--CLSER_freq_fast", type=float, default=0.9, help="The update frequency of the fast model")
    parser.add_argument("--CLSER_freq_slow", type=float, default=0.7, help="The update frequency of the slow model")
    parser.add_argument("--CLSER_lambda", type=float, default=0.1, help="The weight of the MSE loss")

class CLSER(BaseLearner):
    '''
        CLSER: CLSER is based on DER++. 
        Besides, CLSER stores two additional models (fast model and slow model) for computing MSE loss.

        - [Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System](https://openreview.net/forum?id=uxxFrDwrE7Y)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)

        assert params.classifier in ['None','Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'SEQ')
        assert params.il_mode in ['CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'SEQ')
        assert params.classification_type in ['sentence-level'], 'NotImplemented for classification type %s'%(params.classification_type)
        assert params.is_replay, 'NotImplemented for is_replay = %s'%(params.is_replay)

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)

    def build_classifier(self):
        feature_dim = self.model.module.config.hidden_size if hasattr(self.model,'module') else self.model.config.hidden_size
        out_dim_list = self.CL_dataset.continual_config['CUR_NUM_CLASS']
        self.classifier_list = get_classifier(self.params, feature_dim, out_dim_list)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = get_buffer(self.params,self.CL_dataset.continual_config,self.accelerator)
    
    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, self.classifier_list)
        self.wrap_model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
        self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
        self.wrap_fast_model = None
        self.wrap_slow_model = None
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)
        
        # Prepare fast ans slow model
        if task_id==1:
            self.wrap_fast_model = deepcopy(self.wrap_model)
            self.wrap_slow_model = deepcopy(self.wrap_model)

    def end_task(self, task_id):
        super().end_task(task_id)
        
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

        # Sample from buffer and combine old data with new data
        if task_id>0:
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
        if self.params.classifier == 'None':
            if task_id==0:
                total_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                        'attention_mask':lm_input['attention_mask_with_ans'],
                                        'labels':lm_input['labels_with_ans']}).loss
            else:
                with torch.no_grad():
                    # NOTE: Only buffer samples are used for distillation in CLSER
                    num_buffer_sample = lm_input['input_ids_with_ans'].shape[0]//2
                    shift_teacher_fast_logits = self.wrap_fast_model.model(**{'input_ids':lm_input['input_ids_with_ans'][num_buffer_sample:], 
                                        'attention_mask':lm_input['attention_mask_with_ans'][num_buffer_sample:],
                                        'labels':lm_input['labels_with_ans'][num_buffer_sample:]}).logits[:, :-1, :].contiguous()
                    shift_teacher_slow_logits = self.wrap_slow_model.model(**{'input_ids':lm_input['input_ids_with_ans'][num_buffer_sample:], 
                                        'attention_mask':lm_input['attention_mask_with_ans'][num_buffer_sample:],
                                        'labels':lm_input['labels_with_ans'][num_buffer_sample:]}).logits[:, :-1, :].contiguous()
                    shift_teacher_fast_scores = F.softmax(shift_teacher_fast_logits,dim=-1)
                    shift_teacher_slow_scores = F.softmax(shift_teacher_slow_logits,dim=-1)

                student_logits = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                        'attention_mask':lm_input['attention_mask_with_ans'],
                                        'labels':lm_input['labels_with_ans']}).logits
                
                # CE loss on all samples
                # Shift the logits manually for generative models
                ce_loss = self.ce_loss(student_logits[:, :-1, :].contiguous().reshape(-1,student_logits.shape[-1]),
                              lm_input['labels_with_ans'][:, 1:].contiguous().flatten())

                shift_labels = lm_input['labels_with_ans'][num_buffer_sample:][:, 1:].contiguous()
                non_pad_mask = (shift_labels!=-100)
                flatten_non_pad_shift_labels = shift_labels[non_pad_mask]
                one_hot_mask = F.one_hot(flatten_non_pad_shift_labels, num_classes=student_logits.shape[-1]) > 0
                sel_idx = shift_teacher_fast_scores[non_pad_mask][one_hot_mask] > shift_teacher_slow_scores[non_pad_mask][one_hot_mask]
                sel_idx = sel_idx.unsqueeze(1)

                ema_teacher_logits = torch.where(
                    sel_idx,
                    shift_teacher_fast_logits[non_pad_mask],
                    shift_teacher_slow_logits[non_pad_mask],
                )
                mse_loss_old = self.mse_loss(
                    student_logits[num_buffer_sample:][:, :-1, :].contiguous()[non_pad_mask],
                    ema_teacher_logits,
                )

                total_loss = ce_loss + self.params.CLSER_lambda*mse_loss_old

        # Training with Cross-Entropy Loss
        elif self.params.classifier in ['Linear','CosineLinear']:
            extracted_feature = obtain_features(params=self.params, 
                                                model=model, 
                                                lm_input=lm_input, 
                                                tokenizer=self.tokenizer)
            if self.params.il_mode == 'CIL':

                student_logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list[:task_id+1]],dim=-1)
                
                if self.params.classification_type == 'sentence-level':
                    label_idx = lm_input['label_idx_cil']
                else:
                    raise NotImplementedError()
                
                if task_id == 0:
                    total_loss = self.ce_loss(student_logits, label_idx)
                else:
                    with torch.no_grad():
                        # NOTE: Only buffer samples are used for distillation in CLSER
                        num_buffer_sample = lm_input['input_ids_with_ans'].shape[0]//2
                        teacher_fast_features = obtain_features(params=self.params, 
                                                model=self.wrap_fast_model.model, 
                                                lm_input=lm_input, 
                                                tokenizer=self.tokenizer)[num_buffer_sample:]
                        teacher_slow_features = obtain_features(params=self.params, 
                                                model=self.wrap_slow_model.model, 
                                                lm_input=lm_input, 
                                                tokenizer=self.tokenizer)[num_buffer_sample:]
                        teacher_fast_logits = torch.concatenate([classifier(teacher_fast_features) for classifier in self.wrap_fast_model.classifier_list[:task_id+1]],dim=-1)
                        teacher_slow_logits = torch.concatenate([classifier(teacher_slow_features) for classifier in self.wrap_slow_model.classifier_list[:task_id+1]],dim=-1)
                        teacher_fast_scores = F.softmax(teacher_fast_logits,dim=-1)
                        teacher_slow_scores = F.softmax(teacher_slow_logits,dim=-1)
                
                    ce_loss = self.ce_loss(student_logits,label_idx)
                    
                    one_hot_mask = F.one_hot(label_idx[num_buffer_sample:], num_classes=student_logits.shape[-1]) > 0
                    sel_idx = teacher_fast_scores[one_hot_mask] > teacher_slow_scores[one_hot_mask]
                    sel_idx = sel_idx.unsqueeze(1)

                    ema_teacher_logits = torch.where(
                        sel_idx,
                        teacher_fast_logits,
                        teacher_slow_logits,
                    )
                    mse_loss_old = self.mse_loss(
                        student_logits[num_buffer_sample:],
                        ema_teacher_logits,
                    )
                    
                    total_loss = ce_loss + self.params.CLSER_lambda*mse_loss_old
                
            elif self.params.il_mode == 'TIL':

                total_loss = torch.tensor(0.).to(model.device)
     
                for t_id in range(0,task_id+1):

                    class_idx_range_bg = (self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][t_id])
                    class_idx_range_ed = (self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][t_id])
                    task_mask = torch.logical_and(
                        lm_input['label_idx_cil']>=class_idx_range_bg,
                        lm_input['label_idx_cil']<class_idx_range_ed
                    )

                    if task_mask.sum().item()==0:
                        continue
                    
                    student_logits = classifier_list[t_id](extracted_feature[task_mask])

                    if self.params.classification_type == 'sentence-level':
                        label_idx = lm_input['label_idx_til'][task_mask]
                    else:
                        raise NotImplementedError()
                    
                    # New samples
                    if task_id == 0:
                        total_loss += self.ce_loss(student_logits,label_idx)
                    # Buffer samples
                    else:
                        with torch.no_grad():
                            # NOTE: Only buffer samples are used for distillation in CLSER
                            num_buffer_sample = lm_input['input_ids_with_ans'].shape[0]//2
                            teacher_fast_features = obtain_features(params=self.params, 
                                                    model=self.wrap_fast_model.model, 
                                                    lm_input=lm_input, 
                                                    tokenizer=self.tokenizer)
                            teacher_slow_features = obtain_features(params=self.params, 
                                                    model=self.wrap_slow_model.model, 
                                                    lm_input=lm_input, 
                                                    tokenizer=self.tokenizer)
                            teacher_fast_logits = self.wrap_fast_model.classifier_list[t_id](teacher_fast_features[task_mask])
                            teacher_slow_logits = self.wrap_slow_model.classifier_list[t_id](teacher_slow_features[task_mask])
                            teacher_fast_scores = F.softmax(teacher_fast_logits,dim=-1)
                            teacher_slow_scores = F.softmax(teacher_slow_logits,dim=-1)

                        ce_loss_old = self.ce_loss(student_logits,label_idx)

                        one_hot_mask = F.one_hot(label_idx, num_classes=student_logits.shape[-1]) > 0
                        sel_idx = teacher_fast_scores[one_hot_mask] > teacher_slow_scores[one_hot_mask]
                        sel_idx = sel_idx.unsqueeze(1)

                        ema_teacher_logits = torch.where(
                            sel_idx,
                            teacher_fast_logits,
                            teacher_slow_logits,
                        )
                        mse_loss_old = self.mse_loss(
                            student_logits,
                            ema_teacher_logits,
                        )
                        
                        total_loss += ce_loss_old + self.params.CLSER_lambda*mse_loss_old

            else:
                raise NotImplementedError()
        
        else:
            raise NotImplementedError()
        
        if task_id > 0:
            if torch.rand(1) < self.params.CLSER_freq_fast:
                self.update_fast_model_variables()
            if torch.rand(1) < self.params.CLSER_freq_slow:
                self.update_slow_model_variables()

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
            else:
                raise NotImplementedError()

            return  acc

        else:
            raise NotImplementedError()
    # ===========================================================================================


    # ============================ Other Model-Specific Functions ===============================
    def update_fast_model_variables(self):
        alpha = min(1 - 1 / (self.step + 1),  self.params.CLSER_alpha_fast)
        for ema_param, param in zip(self.wrap_fast_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(other=param.data, alpha=1 - alpha)

    def update_slow_model_variables(self):
        alpha = min(1 - 1 / (self.step + 1), self.params.CLSER_alpha_slow)
        for ema_param, param in zip(self.wrap_slow_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(other=param.data, alpha=1 - alpha)

    # ===========================================================================================