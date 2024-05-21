import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_CPFD_params(parser):
    '''
        The parameters of model CPFD
    '''
    parser.add_argument("--CPFD_student_temperate", type=float, default=1, help="The temperature of student model")
    parser.add_argument("--CPFD_teacher_temperate", type=float, default=1, help="The temperature of teacher model")
    parser.add_argument("--CPFD_distill_weight", type=float, default=2, help="The weight of the distillation loss")
    parser.add_argument("--CPFD_classif_adaptive_factor", type=float, default=2)
    parser.add_argument("--CPFD_classif_adaptive_min_factor", type=float, default=0)
    parser.add_argument("--CPFD_threshold", type=float, default=0.001)
    parser.add_argument("--CPFD_adaptive_distill_weight", default=True, help="If using adaptive weight")
    parser.add_argument("--CPFD_adaptive_schedule", type=str, default='root', choices=['root','linear','square'], help="The schedule for adaptive weight")
    parser.add_argument("--is_fix_trained_classifier", type=bool, default=True, help="Fix old classifier")


class CPFD(BaseLearner):
    '''
        a pooled feature distillation loss that skillfully navigates the trade-off between retaining knowledge of old entity types and acquiring new ones
        Continual Named Entity Recognition without Catastrophic Forgetting(2023 EMNLP)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'CPFD')
        assert params.il_mode in ['CIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'CPFD')
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
            
            self.before(self.train_loader_list, task_id, self.wrap_teacher_model)
            
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
        label_idx = lm_input['label_idx_cil']
        # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
        label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
        if not self.params.is_replay:
            label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
        
        if task_id == 0:
            # Training with Cross-Entropy Loss only
            total_loss = self.ce_loss(logits, label_idx.reshape(-1))     
        else:
            # Training with Cross-Entropy Loss + Distillation Loss
            # Compute teacher's logits (only for task from 0 to task_id-1)
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                
                
            ce_mask = (label_idx!=-100)# re-label O class
            distill_mask = (label_idx==0)

            if torch.sum(ce_mask).item()>0:
                # label_idx[bs, seqlen] logits_teacher[bs, seqlen, old_dim]
                # the method use information of batch 
                ce_loss = self.CPFD_ce_loss(label_idx, logits_teacher, logits)
                ce_loss = ce_loss[ce_mask].mean().view(-1)  # scalar
            else:
                ce_loss = torch.tensor(0.).to(logits.device)
            
            logits_teacher = logits_teacher.reshape(-1,logits_teacher.shape[-1])
            distill_mask = distill_mask.view(-1)
            if torch.sum(distill_mask).item()>0:
                distill_loss = self.distill_loss(F.log_softmax(logits[distill_mask]/self.params.CPFD_student_temperate,dim=-1)[:,:logits_teacher.shape[-1]],
                                                F.softmax(logits_teacher[distill_mask]/self.params.CPFD_teacher_temperate,dim=-1))
                
                all_attention_features = self.obtain_attention(model=model, 
                                                lm_input=lm_input)
                with torch.no_grad():
                    refer_all_attention_features = self.obtain_attention(model=wrap_teacher_model.model, 
                                                lm_input=lm_input)

                distill_attention_features_loss = self.CPFD_distill_attention_features_loss(all_attention_features, refer_all_attention_features)
            else:
                distill_loss = torch.tensor(0.).to(logits.device)
                distill_attention_features_loss = torch.tensor(0.).to(logits.device)
            
            all_dims = logits.shape[-1]
            refer_dims = logits_teacher.shape[-1]    
            if not self.params.CPFD_adaptive_distill_weight: 
                distill_loss_coefficient = self.params.CPFD_distill_weight
            elif self.params.CPFD_adaptive_schedule=='root': 
                distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),0.5) 
            elif self.params.CPFD_adaptive_schedule=='linear':
                distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),1)
            elif self.params.CPFD_adaptive_schedule=='square':
                distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),2)
            else:
                raise Exception('Invalid %s'%(self.params.CPFD_adaptive_schedule))

            total_loss = ce_loss + self.params.CPFD_distill_weight*distill_loss_coefficient*(distill_loss + distill_attention_features_loss)

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
    
    def obtain_attention(self, model, lm_input):
        attentions_states = model.forward(**{
                                'input_ids':lm_input['input_ids'],
                                'attention_mask':lm_input['attention_mask']},
                                output_hidden_states=True).attentions
        return attentions_states
    
    def entropy(self, probabilities):
        """Computes the entropy per token.

        :param probabilities: Tensor of shape (bsz,seq_len,refer_dims).
        :return: One entropy per token, shape (bsz,seq_len)
        """
        factor = 1 / math.log(probabilities.shape[-1] + 1e-8)
        return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=-1)
    
    def CPFD_ce_loss(self, labels, refer_logits, logits):
        classif_adaptive_factor = 1.0
        
        mask_background = (labels < self.old_classes) & (labels != -100) 
    
        probs = torch.softmax(refer_logits, dim=-1) # (bsz,seq_len,refer_dims)
        _, pseudo_labels = probs.max(dim=-1) # 

        mask_valid_pseudo = self.entropy(probs) < self.thresholds[pseudo_labels] # (bsz, seq_len)
        
        
        # labels = labels.reshape(refer_logits.shape[0], refer_logits.shape[1])
        # All old labels that are NOT confident enough to be used as pseudo labels:
        labels[~mask_valid_pseudo & mask_background] = -100

        # All old labels that are confident enough to be used as pseudo labels:
        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                    mask_background]
        
        if self.params.CPFD_classif_adaptive_factor:
            # Number of old/bg tokens that are certain
            num = (mask_valid_pseudo & mask_background).float().sum(dim=-1)
            # Number of old/bg tokens
            den =  mask_background.float().sum(dim=-1)
            # If all old/bg tokens are certain the factor is 1 (loss not changed)
            # Else the factor is < 1, i.e. the loss is reduced to avoid
            # giving too much importance to new tokens
            classif_adaptive_factor = num / (den + 1e-6)

            classif_adaptive_factor = classif_adaptive_factor[:, None]

            if self.params.CPFD_classif_adaptive_min_factor:
                classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.params.CPFD_classif_adaptive_min_factor)


        loss = nn.CrossEntropyLoss(reduction='none')(logits, labels.view(-1)) #   (bsz,seq_len)
        loss = loss.reshape(-1, self.features_teacher.shape[1])
        loss = classif_adaptive_factor * loss

        # type balance
    
        pre_sample_weights = self.calculate_sample_weight(labels)
        sample_weights = torch.ones(loss.size()).to(logits.device)

        for i in range(pre_sample_weights.size(0)):
            sample_weights[i][(labels[i] > 0) & (labels[i] < self.old_classes)] = pre_sample_weights[i] 
  
        loss = sample_weights * loss

        return loss
    
    def calculate_sample_weight(self, labels):
        background = labels == 0
        old_token = (labels < self.old_classes) & (labels != -100)
        old_token = old_token & (~background)
        new_token = labels >= self.old_classes
        old_token = torch.sum(old_token, 1)
        new_token = torch.sum(new_token, 1)
        new_token[new_token==0] = 1 
        weight = 0.5 + F.sigmoid(old_token/new_token)
        return weight
    
    def before(self, train_loader, task_id, wrap_teacher_model):
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
        self.thresholds = self.find_median(train_loader, task_id, wrap_teacher_model)
    
    def find_median(self, train_loader, task_id, wrap_teacher_model):
        bg_entropy_values_total = []
        bg_pseudo_labels_total = []
        for lm_input in train_loader[task_id]:
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                # logits_teacher = logits_teacher.reshape(-1,logits_teacher.shape[-1])

            label_idx = lm_input['label_idx_cil']
            # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
            label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
            
            mask_bg = label_idx == 0  # (bsz, seq_len) 
            probas = torch.softmax(logits_teacher, dim=-1) # (bsz,seq_len,refer_dims)

            _, pseudo_labels = probas.max(dim=-1) 

            bg_entropy_values = self.entropy(probas)[mask_bg].view(-1)
            bg_entropy_values_total.extend(bg_entropy_values.detach().cpu().numpy().tolist())

            bg_pseudo_labels = pseudo_labels[mask_bg].view(-1) # bsz*seq_len
            bg_pseudo_labels_total.extend(bg_pseudo_labels.detach().cpu().numpy().tolist())

    
        bg_entropy_values_total = np.array(bg_entropy_values_total,dtype=np.float32)
        bg_pseudo_labels_total = np.array(bg_pseudo_labels_total, dtype=np.int32)
        thresholds = np.zeros(self.old_classes, dtype=np.float32) #old_classes
        base_threshold = self.params.CPFD_threshold #0.001
        for c in range(len(thresholds)):
            thresholds[c] = np.median(bg_entropy_values_total[bg_pseudo_labels_total==c])
            thresholds[c] = max(thresholds[c], base_threshold)

        return torch.from_numpy(thresholds).to(self.features_teacher.device)

    def CPFD_distill_attention_features_loss(self, all_attention_features, refer_all_attention_features):
        distill_attention_features_loss = torch.tensor(0., requires_grad=True).to(self.features_teacher.device)
        for attention_features, refer_attention_features in zip(all_attention_features, refer_all_attention_features):
            assert attention_features.shape == refer_attention_features.shape, (attention_features.shape, refer_attention_features.shape)  # (bsz, heads=12, seq_len, seq_len)
            
            bsz, heads, seq_len, seq_len = attention_features.shape

            
            attention_features = torch.where(attention_features <= -1e2, torch.zeros_like(attention_features).to(self.features_teacher.device),
                                            attention_features)
            refer_attention_features = torch.where(refer_attention_features <= -1e2, torch.zeros_like(refer_attention_features).to(self.features_teacher.device),
                                            refer_attention_features)

            # pooled feature distillation
            pfd1 = torch.mean(attention_features, dim=1)
            rpfd1 = torch.mean(refer_attention_features, dim=1)

            pfd2 = torch.mean(attention_features, dim=2)
            rpfd2 = torch.mean(refer_attention_features, dim=2)

            pfd3 = torch.mean(attention_features, dim=3)
            rpfd3 = torch.mean(refer_attention_features, dim=3)

            layer_loss1 = nn.MSELoss(reduction='mean')(pfd1,rpfd1)
            layer_loss2 = nn.MSELoss(reduction='mean')(pfd2,rpfd2)
            layer_loss3 = nn.MSELoss(reduction='mean')(pfd3,rpfd3)

            layer_loss = layer_loss1 + layer_loss2 + layer_loss3

            distill_attention_features_loss += layer_loss

        distill_attention_features_loss = distill_attention_features_loss/len(all_attention_features)
        
        return distill_attention_features_loss
    # ===========================================================================================