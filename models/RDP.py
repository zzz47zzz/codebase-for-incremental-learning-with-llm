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
from utils.evaluation import evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_RDP_params(parser):
    '''
        The parameters of model RDP
    '''
    parser.add_argument("--RDP_student_temperate", type=float, default=1, help="The temperature of student model")
    parser.add_argument("--RDP_teacher_temperate", type=float, default=1, help="The temperature of teacher model")
    parser.add_argument("--RDP_proto_temperature", type=float, default=1, help="The temperature of protype")
    parser.add_argument("--RDP_distill_weight", type=float, default=2, help="The weight of the distillation loss")
    parser.add_argument("--RDP_soft_param", type=float, default=0.3, help="The weight of the softlabel loss")
    parser.add_argument("--RDP_regular_param", type=float, default=0.1, help="The weight of the self-entropy loss")
    parser.add_argument("--is_fix_trained_classifier", type=bool, default=True, help="Fix old classifier")
    
    


class RDP(BaseLearner):
    '''
         a method called task Relation Distillation and Prototypical pseudo label (RDP)
        
        Task Relation Distillation and Prototypical Pseudo Label for Incremental Named Entity Recognition(2023CIKM)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'RDP')
        assert params.il_mode in ['CIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'RDP')
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
        num_task = self.CL_dataset.continual_config['NUM_TASK']
        if self.params.classifier!='None' and self.params.is_fix_trained_classifier:
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

        # Prepare teacher model
        if task_id>0:
            if self.wrap_teacher_model is not None:
                self.wrap_teacher_model.cpu()
                del self.wrap_teacher_model
            self.wrap_teacher_model = deepcopy(self.wrap_model)
            # update prototype          
            
            self.before_prototype(self.train_loader_list, task_id, self.wrap_teacher_model)
            
        if task_id>0 and self.params.classifier!='None' and self.params.is_fix_trained_classifier:
            self.optimizer.param_groups[self.param_groups_mapping['classifier'][task_id-1]]['lr']=0

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
            # Training with Cross-Entropy Loss only
            total_loss = self.ce_loss(logits,label_idx)     
        else:
            # Training with Cross-Entropy Loss + Distillation Loss
            # Compute teacher's logits (only for task from 0 to task_id-1)
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                logits_teacher = logits_teacher.reshape(-1,logits_teacher.shape[-1])
            #re-label: label_idx 0 also is ce_mask
            ce_mask = (label_idx!=-100)
            distill_mask = (label_idx==0)
            
            # prototype label
            probs = torch.softmax(logits_teacher, dim=-1)
            weights = self.get_prototype_weight(extracted_feature_teacher)   # (bs, seq_len, old_classes)
            weights = weights.view(-1, weights.shape[-1])
            rectified = weights * probs
            rectified = rectified / rectified.sum(-1, keepdim=True)
            _, pseudo_labels_rec = rectified.max(dim=-1)

            label_idx[distill_mask] = pseudo_labels_rec[distill_mask]
            
            if torch.sum(ce_mask).item()>0:
                ce_loss = self.ce_loss(logits[ce_mask],label_idx[ce_mask])
            else:
                ce_loss = torch.tensor(0.).to(logits.device)
            
            original_labels = label_idx.clone()          
            old_outputs = torch.sigmoid(logits_teacher)
            # assign to self.variable

            loss_soft_label = BCEWithLogitsLossWithIgnoreIndexSoftLabel(reduction='none', ignore_index=-100)(logits, old_outputs, self.old_classes, original_labels)
            Regularizer_soft = self.reg_pesudo_label(logits)

            loss_soft_label = loss_soft_label.mean()
            Regularizer_soft = Regularizer_soft.mean()
            
            
            if torch.sum(distill_mask).item()>0:
                distill_loss = self.distill_loss(F.log_softmax(logits[distill_mask]/self.params.RDP_student_temperate,dim=-1)[:,:logits_teacher.shape[-1]],
                                                F.softmax(logits_teacher[distill_mask]/self.params.RDP_teacher_temperate,dim=-1))
            else:
                distill_loss = torch.tensor(0.).to(logits.device)

            total_loss = ce_loss + self.params.RDP_distill_weight*distill_loss + self.params.RDP_soft_param * loss_soft_label + self.params.RDP_regular_param * Regularizer_soft

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
    
    # ======================== Other Model-Specific Functions ===================================
    
    def _update_running_stats(self, labels_down, features, prototypes, count_features):
        labels_down = labels_down.reshape(features.shape[0], features.shape[1]).unsqueeze(-1).long()
        cl_present = torch.unique(input=labels_down)
   
        cl_present=torch.where((cl_present < self.old_classes) & (cl_present != -100), cl_present, -100)
        cl_present = torch.unique(input=cl_present)
      
        if cl_present[0] == -100:
            cl_present = cl_present[1:]

        features_local_mean = torch.zeros([self.old_classes, self.hidden_size]).to(features.device)


        for cl in cl_present:
            features_cl = features[(labels_down == cl).expand(-1, -1, features.shape[-1])].view(features.shape[-1], -1).detach()
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                            prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features

    def update_prototypes(self, train_loader, task_id, wrap_teacher_model):
        prototypes = torch.zeros([self.old_classes, self.hidden_size])
        prototypes.requires_grad = False
        prototypes = prototypes.to(self.features_teacher.device)
        count_features = torch.zeros([self.old_classes], dtype=torch.long)
        count_features.requires_grad = False
        count_features = count_features.to(self.features_teacher.device)

        for lm_input in train_loader[task_id]:
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                logits_teacher = logits_teacher.reshape(-1,logits_teacher.shape[-1])

            probas = torch.softmax(extracted_feature_teacher, dim=-1)
            _, pseudo_probas = probas.max(dim=-1)
            
            label_idx = lm_input['label_idx_cil'].reshape(-1)
            # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
            label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
            mask_bg = (label_idx == 0)
            label_idx[mask_bg] = pseudo_probas.view(-1)[mask_bg]
            # extracted_feature_teacher = extracted_feature_teacher.view(-1)
            prototypes, count_features = self._update_running_stats(label_idx.unsqueeze(-1).long(), extracted_feature_teacher, prototypes,
                                                                    count_features)

        return prototypes, count_features

    def get_prototype_weight(self, feat):
        feat_proto_distance = self.feat_prototype_distance(feat)
        weight = F.softmax(-feat_proto_distance * self.params.RDP_proto_temperature, dim=-1)
        return weight

    def feat_prototype_distance(self, feat):
        bs, seq_len, _ = feat.shape
        feat_proto_distance = -torch.ones((bs, seq_len, self.old_classes)).to(feat.device)
        for i in range(self.old_classes):
            feat_proto_distance[:, :, i] = torch.norm(self.prototypes[i].reshape(1,1,-1).expand(bs,seq_len,-1) - feat, 2, dim=-1,)
        return feat_proto_distance

    
    def before_prototype(self, train_loader, task_id, wrap_teacher_model):
        for lm_input in train_loader[task_id]:
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                logits_teacher = logits_teacher.reshape(-1,logits_teacher.shape[-1])
                break
        self.old_classes = logits_teacher.shape[-1]
        self.hidden_size = self.model.module.config.hidden_size if hasattr(self.model,'module') else self.model.config.hidden_size
        self.features_teacher = extracted_feature_teacher
        self.prototypes, self.count_features = self.update_prototypes(
                train_loader, task_id, wrap_teacher_model)
    
    
    def reg_pesudo_label(self, output):
        output = torch.softmax(output, dim=-1) # (bsz, seq_len, all_dims)
        loss = -(output * torch.log(output)).mean(dim=-1)
        return loss
    
    
    
    # ===========================================================================================
    
# ======================== Other Model-Specific Module ===================================    
class BCEWithLogitsLossWithIgnoreIndexSoftLabel(nn.Module):

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, old_outputs, old_classes, targets):
        # inputs of size B x seq_len x all_dims
        n_cl = torch.tensor(inputs.shape[-1]).to(inputs.device)   
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[-1] + 1).float() #B x seq_len x all_dims + 1

        targets = targets[ :, :inputs.shape[-1]]  # remove -100 from 1hot B x seq_len x all_dim
       
        targets[ :, :old_classes] = old_outputs

        outputs = torch.log_softmax(inputs, dim=-1)
        labels = torch.softmax(targets, dim=-1)
   
        loss = -(outputs * labels).mean(dim=-1)
 
        return loss
    
    