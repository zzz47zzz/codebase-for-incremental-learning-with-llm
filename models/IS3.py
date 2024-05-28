import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from seqeval.metrics import f1_score
from sklearn.metrics import confusion_matrix

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_IS3_params(parser):
    '''
        The parameters of model IS3 (Ours)
    '''
    parser.add_argument("--IS3_student_temperate", type=float, default=1, help="The temperature of student model")
    parser.add_argument("--IS3_teacher_temperate", type=float, default=1, help="The temperature of teacher model")
    parser.add_argument("--IS3_distill_weight", type=float, default=2, help="The weight of the distillation loss")

    parser.add_argument("--pro_weight", type=float, default=0.1)
    parser.add_argument("--grad_weight", type=float, default=0.6)
    parser.add_argument("--new_grad_weight", type=float, default=1.)


class IS3(BaseLearner):
    '''
        IS3: Decompose the catastrophic forgetting problem into E2O and O2E.
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'IS3')
        assert params.il_mode in ['CIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'IS3')
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
        self.prototype = {}
        self.ce_loss = nn.CrossEntropyLoss()
        self.distill_loss = nn.KLDivLoss(reduction='batchmean')
        self.consistency_loss = nn.MSELoss(reduction='none')

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)
        num_task = self.CL_dataset.continual_config['NUM_TASK']
        if self.params.classifier!='None':
            self.param_groups_mapping = {}
            self.param_groups_mapping['encoder'] = 0
            self.param_groups_mapping['classifier'] = {i:1+i for i in range(num_task)}

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

        # Prepare teacher model
        if task_id>0:
            if self.wrap_teacher_model is not None:
                self.wrap_teacher_model.cpu()
                del self.wrap_teacher_model
            
            self.wrap_teacher_model = deepcopy(self.wrap_model)
        
    def end_task(self, task_id):
        super().end_task(task_id)

        entity_dict = {}
        warp_model = self.wrap_model
        with torch.no_grad():
            for lm_input in self.train_loader_list[task_id]: 
                extracted_features = warp_model.model.forward(**{
                                                'input_ids':lm_input['input_ids'],
                                                'attention_mask':lm_input['attention_mask']},
                                                output_hidden_states=True).hidden_states[-1]
                extracted_features = F.normalize(extracted_features, dim=-1)
                
                label_idx = lm_input['label_idx_cil']
        
                for i in range(0, label_idx.shape[0]):
                    for j in range(0, label_idx.shape[1]):
                        old_class = label_idx[i][j].item()
                        if old_class > 0 and old_class < self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id]:
                            if old_class not in entity_dict:
                                entity_dict[old_class] = []
                            entity_dict[old_class].append(extracted_features[i, j, :])

            for k, v in entity_dict.items():
                pc = torch.mean(torch.stack(v, dim=0), dim=0)
                if k not in self.prototype:
                    self.prototype[k] = pc 
                vec = torch.stack(v, dim=0)
                center = pc.unsqueeze(0).repeat(vec.shape[0], 1)
                dis = torch.norm(vec-center,dim=-1)
            
            self.prototype = dict(sorted(self.prototype.items()))
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

        # For Distributed Data Parallel
        if hasattr(self.wrap_teacher_model,'module'):
            wrap_teacher_model = self.wrap_teacher_model.module
        else:
            wrap_teacher_model = self.wrap_teacher_model

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
            total_loss = self.ce_loss(logits, label_idx)
        else:
            # Training with Cross-Entropy Loss + Distillation Loss + Prototype Loss
            # Compute teacher's logits (only for task from 0 to task_id-1)
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                logits_teacher = logits_teacher.reshape(-1,logits_teacher.shape[-1])


            ce_mask = torch.logical_and(label_idx!=-100,label_idx!=0)
            distill_mask = (label_idx==0)

            if torch.sum(ce_mask).item()>0:
                ce_loss = self.ce_loss(logits[ce_mask],label_idx[ce_mask])
            else:
                ce_loss = torch.tensor(0.).to(logits.device)

            pro_features = []
            pro_label = []
            old_class = logits_teacher.shape[-1]
            index = list(range(1, old_class))
            if torch.sum(ce_mask).item()>0:
                for i in range(0, len(index)):
                    if index[i] in self.prototype:
                        temp = self.prototype[index[i]]
                        pro_features.append(temp.unsqueeze(0))
                        pro_label.append(index[i])

                pro_features = torch.cat(pro_features, dim=0)
                pro_label = torch.tensor(pro_label).to(logits.device)
                pro_logits = torch.concatenate([classifier(pro_features) for classifier in classifier_list[:task_id+1]],dim=-1)
                pro_loss = self.ce_loss(pro_logits, pro_label)
            else:
                pro_loss = torch.tensor(0.).to(logits.device)

            if torch.sum(distill_mask).item()>0:
                distill_loss = self.distill_loss(F.log_softmax(logits[distill_mask]/self.params.IS3_student_temperate,dim=-1)[:,:logits_teacher.shape[-1]],
                                F.softmax(logits_teacher[distill_mask]/self.params.IS3_teacher_temperate,dim=-1))
            else:
                distill_loss = torch.tensor(0.).to(logits.device)

            total_loss = ce_loss + self.params.IS3_distill_weight*distill_loss + self.params.pro_weight*pro_loss
            
        if task_id>0:
            remain_loss = self.params.IS3_distill_weight*distill_loss + self.params.pro_weight*pro_loss
        # Backward
        model.train()
        self.optimizer.zero_grad()      
        if task_id >0:
            if torch.sum(ce_mask).item()>0:
                self.accelerator.backward(ce_loss, retain_graph=True)
                classifier_list = self.wrap_model.classifier_list[:task_id+1]
                for i in range(0, len(classifier_list)):
                    if i < len(classifier_list)-1:
                        classifier = classifier_list[i]
                        revised_grad = self.params.grad_weight * classifier.weight.grad.data
                        if i == 0:
                            classifier.weight.grad.data[1:, :] = revised_grad[1:, :]
                        else:
                            classifier.weight.grad.data = revised_grad
                    else:
                        new_classifier = classifier_list[i]
                        new_classifier.weight.grad.data = self.params.new_grad_weight * new_classifier.weight.grad.data
            
            if remain_loss.item()>0:
                self.accelerator.backward(remain_loss)
        else:
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

        acc = self.evaluate_word_level_acc_with_classifier(
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
    def compute_ce_loss(self, logits, label_idx, ce_mask, logits_teacher=None, pad_mask=None):
        if self.params.is_bio_object:
            label_old_idx = torch.argmax(logits_teacher, dim=-1)
            # label_second = torch.argsort(logits, dim=-1, descending=True)[:, 1]

            label_idx[label_idx<0] = 0
            label_one_hot = F.one_hot(label_idx, num_classes=logits.shape[-1])
            logits_old = logits * (torch.ones_like(label_one_hot) - label_one_hot)
            label_idx[pad_mask] = -100

            ce_loss_new = self.ce_loss(logits[ce_mask],label_idx[ce_mask])
            ce_loss_old = self.ce_loss(logits_old[ce_mask], label_old_idx[ce_mask])
            ce_loss = ce_loss_new + self.params.ce_teacher_weight*ce_loss_old
        else:
            ce_loss = self.ce_loss(logits[ce_mask],label_idx[ce_mask])
        
        return ce_loss

    
    def compute_distill_loss(self, stable_model_logits, plastic_model_logits, teacher_label, logits, distill_mask):

        stable_model_prob = F.softmax(stable_model_logits, 1)
        plastic_model_prob = F.softmax(plastic_model_logits, 1)# 5

        # label_mask = F.one_hot(teacher_label, num_classes=stable_model_logits.shape[-1]) > 0
        # sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
        stable_value, stabel_idx = torch.max(stable_model_prob, dim=-1)
        plastic_value, plastic_idx = torch.max(plastic_model_prob, dim=-1)
        
        sel_idx = stable_value > plastic_value
        sel_idx = sel_idx.unsqueeze(1)

        ema_logits = torch.where(
            sel_idx,
            stable_model_logits,
            plastic_model_logits,
        )


        distill_loss = self.distill_loss(F.log_softmax(logits[distill_mask]/self.params.IS3_student_temperate,dim=-1),#[:,:ema_logits.shape[-1]],
                                F.softmax(ema_logits[distill_mask]/self.params.IS3_teacher_temperate,dim=-1))
        
        return distill_loss
    
    
    def evaluate_word_level_acc_with_classifier(self, model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
        '''
            Evaluate the accuracy with classifier for word-lvel classification tasks

            Return:
            - macro f1 (%)
        '''
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
            
    


        