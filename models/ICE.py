import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from seqeval.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_ICE_params(parser):
    '''
        The parameters of model ICE
    '''

    parser.add_argument("--ICE_Delta", type=float, default=0, help="constant value δ as the logit of the Other category")
    parser.add_argument("--ICE_fix_encoder", type=bool, default=True)
    parser.add_argument("--ICE_fix_old_classifier", type=bool, default=True)
    parser.add_argument("--ICE_O", type=bool, default=False)
    parser.add_argument("--ICE_PLO", type=bool, default=False)
    

class ICE(BaseLearner):
    '''
        an Individual Classifiers with Frozen Feature Extractor (ICE) framework
        
        Teamwork Is Not Always Good: An Empirical Study of Classifier Drift in Class-incremental Information Extraction(2023 ACL)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'ICE')
        assert params.il_mode in ['CIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'ICE')
        assert params.classification_type == 'word-level', 'NotImplemented for classification_type %s'%(params.classification_type)
        assert not params.is_replay, 'NotImplemented for data replay!'

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
        self.distill_loss = nn.KLDivLoss(reduction='batchmean')

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)
        # Mapping ENCODER/CLASSIFIER and task_id to the index in the param_groups of the optimizer
        num_task = self.CL_dataset.continual_config['NUM_TASK']
        if self.params.classifier!='None' and self.params.ICE_fix_encoder:
            self.param_groups_mapping = {}
            self.param_groups_mapping['encoder'] = 0
            self.param_groups_mapping['classifier'] = {i:i+1 for i in range(num_task)}
            


    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None

    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, self.classifier_list)
        self.wrap_model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
        self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
        self.wrap_teacher_model = None
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)
            
        if task_id>0 and self.params.classifier!='None' and self.params.ICE_fix_old_classifier:
            self.optimizer.param_groups[self.param_groups_mapping['classifier'][task_id-1]]['lr']=0
        if self.params.ICE_fix_encoder:
            self.optimizer.param_groups[self.param_groups_mapping['encoder']]['lr']=0

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
        if (self.params.evaluate_interval>0) and \
            ((epoch_id>0 and epoch_id%self.params.evaluate_interval==0) or (task_id==0 and epoch_id==0)):
            self.evaluate_model(task_id=task_id)
        self.loss_list = []
        
    def observe_batch(self, task_id, epoch_id, lm_input):
        '''
            Observe a batch of data
        '''
        # Update step
        self.step += 1
        self.global_step += 1

        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            wrap_model = self.wrap_model.module
        else:
            wrap_model = self.wrap_model

        model = wrap_model.model
        classifier_list = wrap_model.classifier_list

        # Compute loss
        extracted_feature = obtain_features(params=self.params, 
                                                model=model, 
                                                lm_input=lm_input, 
                                                tokenizer=self.tokenizer)
        
        logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list[:task_id+1]],dim=-1)
        logits = logits.reshape(-1,logits.shape[-1])
        label_idx = lm_input['label_idx_cil'].reshape(-1)
        # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
        label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
        if not self.params.is_replay:
            label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
        
        if task_id == 0:
            # Training with Cross-Entropy Loss only
            logits[:, 0] = self.params.ICE_Delta
            total_loss = self.ce_loss(logits, label_idx)     
        else:
            old_logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list[:task_id]],dim=-1)
            old_logits = old_logits.reshape(-1, old_logits.shape[-1])
            if self.params.ICE_O:
                old_dims = old_logits.shape[-1]
                mask = torch.zeros_like(logits) + float("-inf")
                mask[:, 0] = 0
                mask[:, old_dims:] = 0
                logits[:, 0] = self.params.ICE_Delta
                logits += mask
                ce_mask = (label_idx!=-100)
                if torch.sum(ce_mask).item()>0:
                    ce_loss = self.ce_loss(logits[ce_mask],label_idx[ce_mask])
                else:
                    ce_loss = torch.tensor(0.).to(logits.device)
            elif self.params.ICE_PLO:
                logits[:, 0] = self.params.ICE_Delta
                ce_mask = (label_idx!=-100)
                if torch.sum(ce_mask).item()>0:
                    ce_loss = self.ce_loss(logits[ce_mask],label_idx[ce_mask])
                else:
                    ce_loss = torch.tensor(0.).to(logits.device)
                    
            else:
                 raise NotImplementedError('Not implemented for ICE')

            total_loss = ce_loss

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
        assert il_mode == 'CIL', 'NotImplemented for il_mode %s'%(il_mode)
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

        return  acc

    # ===========================================================================================