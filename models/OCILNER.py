import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn.modules import Module
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss
from seqeval.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader
from datasets import Dataset

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.buffer import get_buffer
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_OCILNER_params(parser):
    '''
        The parameters of model OCILNER
    '''
    parser.add_argument("--OCILNER_student_temperate", type=float, default=1, help="The temperature of student model")
    parser.add_argument("--OCILNER_teacher_temperate", type=float, default=1, help="The temperature of teacher model")
    parser.add_argument("--OCILNER_distill_weight", type=float, default=2, help="The weight of the distillation loss")
    parser.add_argument("--OCILNER_is_replay", type=bool, default=True, help="If using replay sammples")
    parser.add_argument("--loss_name1", type=str, default="supcon_ce")
    parser.add_argument("--loss_name2", type=str, default="supcon_o_bce")
    parser.add_argument("--OCILNER_beta", type=float, default=0.98)
    parser.add_argument("--start_train_o_epoch", type=int, default=3)
    parser.add_argument("--per_class_samples", type=int, default=5)

    


class OCILNER(BaseLearner):
    '''
        an entity-aware contrastive learning method that adaptively detects entity clusters in "O"
        
        Learning “O” Helps for Learning More: Handling the Unlabeled Entity Problem for Class-incremental NER(2023 ACL)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'OCILNER')
        assert params.il_mode in ['CIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'OCILNER')
        assert params.classification_type == 'word-level', 'NotImplemented for classification_type %s'%(params.classification_type)

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)

    def build_classifier(self):
        feature_dim = self.model.module.config.hidden_size if hasattr(self.model,'module') else self.model.config.hidden_size
        out_dim_list = self.CL_dataset.continual_config['CUR_NUM_CLASS']
        self.classifier_list = get_classifier(self.params, feature_dim, out_dim_list)

        self.kd_loss_fct = KdLoss()
        self.ce_loss_fct = CrossEntropyLoss()
        

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
        if self.params.OCILNER_is_replay:
            self.buffer = get_buffer(self.params,self.CL_dataset.continual_config,self.accelerator)

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
            
            self.entity_dict, self.prototype, self.theta_proto = self.get_entity_dict_and_prototype(self.wrap_teacher_model, task_id)

      
    def end_task(self, task_id):
        super().end_task(task_id)

        if self.params.OCILNER_is_replay:
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
        # if task_id>0 and self.params.OCILNER_is_replay:
        #     train_dataset = self.train_loader_list[task_id].dataset
        #     buf_dataset = Dataset.from_dict(self.buffer.get_all_data())
        #     buf_dataset.set_format(type='torch')
        #     cur_train_loader = DataLoader(
        #         ConcatDataset((train_dataset,buf_dataset)),
        #         batch_size=self.params.batch_size,
        #         shuffle=True,
        #         drop_last=False
        #     )
        #     cur_train_loader = self.accelerator.prepare(cur_train_loader)
        # else:
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
        if (self.params.evaluate_interval>0) and \
            ((epoch_id>0 and epoch_id%self.params.evaluate_interval==0) or (task_id==0 and epoch_id==0)):
            self.evaluate_model(task_id=task_id)
        self.loss_list = []
        if epoch_id >= self.params.start_train_o_epoch:
            self.buffer.update_buffer(task_id, self.train_loader_list[task_id], self.wrap_model, self.tokenizer)    
            entity_dict, prototype, theta_proto = self.get_entity_dict_and_prototype(self.wrap_model, task_id)
            self.prototype_dists = self.get_prototype_dists(entity_dict)
                    

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
    
        extracted_feature = F.normalize(extracted_feature.view(-1, extracted_feature.shape[-1]), dim=1)
        logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list[:task_id+1]],dim=-1)
        label_idx = lm_input['label_idx_cil']
        # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
        label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
        # if not self.params.OCILNER_is_replay:
        label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
        
        loss_name = self.params.loss_name1

        if epoch_id >= self.params.start_train_o_epoch:
            loss_name = self.params.loss_name2
            cls = NNClassification()
            # auto-selected positive samples for 'O'
            top_emissions, _ = cls.get_top_emissions_with_th(extracted_feature,
                                                                label_idx,
                                                                th_dists=torch.median(self.prototype_dists).item())
        else:
            top_emissions = None
        
        self.num_labels = logits.shape[-1]

        extracted_feature = extracted_feature.unsqueeze(1)  # [batch_size*seq_length, 1, embedding_size]
        label_idx = label_idx.view(-1)  # [batch_size*seq_length]
        logits = logits.reshape(-1, logits.shape[-1])
        if task_id == 0:
            # Training with Cross-Entropy Loss only
            if loss_name == "supcon_ce":
                supcon_loss = self.compute_supcon_loss(extracted_feature, label_idx, topk_th=True)
                ce_loss = self.ce_loss_fct(logits, label_idx)
                loss = supcon_loss + ce_loss
            elif loss_name == "supcon_o_bce":
                supcon_o_loss = self.compute_supcon_o_loss(extracted_feature, label_idx, top_emissions, topk_th=True)
                bce_loss = self.compute_bce_loss(logits, label_idx, self.num_labels)
                loss = supcon_o_loss + bce_loss

            total_loss = loss
        else:
            # Training with Cross-Entropy Loss + Distillation Loss
            # Compute teacher's logits (only for task from 0 to task_id-1)
            
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                extracted_feature_teacher = F.normalize(extracted_feature_teacher.view(-1, extracted_feature_teacher.shape[-1]), dim=1)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)

            ce_mask = torch.logical_and(label_idx!=-100,label_idx!=0)
            distill_mask = (label_idx==0)
            logits_teacher = logits_teacher.reshape(-1, logits_teacher.shape[-1])
            
            if torch.sum(distill_mask).item()>0:
                label_idx = self.relabel(lm_input, wrap_teacher_model, label_idx)
                label_idx = label_idx.view(-1)
            
            if loss_name == "supcon_ce":
                # kd_loss = kd_loss_fct(s_logits, t_logits, t=2)
                supcon_loss = self.compute_supcon_loss(extracted_feature, label_idx, topk_th=True)
                if torch.sum(ce_mask).item()>0:
                    ce_loss = self.ce_loss_fct(logits[ce_mask], label_idx[ce_mask])
                else:
                    ce_loss = torch.tensor(0.).to(logits.device)
                if torch.sum(distill_mask).item()>0:
                    distill_loss = self.kd_loss_fct(logits[distill_mask][:, :logits_teacher.shape[-1]], logits_teacher[distill_mask], t=2)
                else:
                    distill_loss = torch.tensor(0.).to(logits.device)
                    
                loss = supcon_loss+ce_loss + distill_loss
            elif loss_name == "supcon_o_bce":
                t_logits = logits_teacher
                supcon_o_loss = self.compute_supcon_o_loss(extracted_feature, label_idx, top_emissions, topk_th=True)
                bce_loss = self.compute_bce_loss(logits, label_idx, self.num_labels, t_logits)
                loss = supcon_o_loss + bce_loss 

            total_loss = loss

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

        acc = self.evaluate_word_level_acc_with_ncm_classifier(
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
    
    def evaluate_word_level_acc_with_ncm_classifier(self, model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
        '''
            Evaluate the accuracy with classifier for word-lvel classification tasks

            Return:
            - macro f1 (%)
        '''
        model.eval()
        self.buffer.update_buffer(cur_task_id, self.train_loader_list[cur_task_id], model, self.tokenizer) 
        prototype= self.get_ncm_prototype(self.wrap_model, cur_task_id)

        with torch.no_grad():
            # Only test once because the last test set contains all labels
            gold_lines = []
            pred_lines = []

            for lm_input in eval_data_loader: 
                extracted_features = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input,
                                                    tokenizer=tokenizer)
                
                n_tags = self.CL_dataset.continual_config["ACCUM_NUM_CLASS"][cur_task_id]

                feature = extracted_features.view(-1, extracted_features.shape[-1])  # (batch_size, feature_size)
                for j in range(feature.size(0)):  # Normalize
                    feature.data[j] = feature.data[j] / feature.data[j].norm()
                #feature = feature.unsqueeze(1)  # (batch_size, 1, feature_size)

                means = torch.stack([prototype[cls] if cls in prototype else torch.zeros_like(feature[0]) for cls in range(0, n_tags) ])  # (n_classes, feature_size)
            
                dists = -torch.matmul(feature, means.T)  # (batch_size, n_classes)
                _, pred_label = dists.min(1)
                preds = pred_label.view(extracted_features.shape[0], extracted_features.shape[1])

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
    
    def get_entity_dict_and_prototype(self, warp_teacher_model, task_id):
        buf_dataset = self.buffer.get_all_data()
        buf_label_idx = buf_dataset['label_idx_cil']
        buf_label_idx = buf_label_idx.masked_fill(buf_label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
        buf_input_ids = buf_dataset['input_ids']
        buf_attention_mask =  buf_dataset['attention_mask']
        entity_dict = {}
        buf_features = []
        for i in range(0, buf_input_ids.shape[0]):
            sentence = buf_input_ids[i].unsqueeze(0)
            _attention_mask = buf_attention_mask[i].unsqueeze(0)
            with torch.no_grad():
                _buf_features = warp_teacher_model.model.forward(**{
                                                'input_ids':sentence,
                                                'attention_mask':_attention_mask},
                                                output_hidden_states=True).hidden_states[-1]
                _buf_features = F.normalize(_buf_features, dim=-1)
            buf_features.append(_buf_features)
        buf_features = torch.cat(buf_features, dim=0)
        for i in range(0, buf_label_idx.shape[0]):
            for j in range(0, buf_label_idx.shape[1]):
                old_class = buf_label_idx[i][j].item()
                if old_class > 0 and old_class < self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id]:
                    if old_class not in entity_dict:
                        entity_dict[old_class] = []
                    if len(entity_dict[old_class]) < self.params.per_class_samples:
                        entity_dict[old_class].append(buf_features[i, j, :])

        prototype = {}
        for k, v in entity_dict.items():
            pc = torch.mean(torch.stack(v, dim=0), dim=0)
            if k not in prototype:
                prototype[k] = pc 
        
        prototype = dict(sorted(prototype.items()))
        
        theta_proto = 0
        theta = float('inf')
        for k, v in entity_dict.items():
            for _v in v:
                cos_similarity = F.cosine_similarity(prototype[k], _v, dim=0)
                if cos_similarity < theta:
                    theta = cos_similarity
        theta_proto = self.params.OCILNER_beta * theta
        
        return entity_dict, prototype, theta_proto
    
    def get_ncm_prototype(self, warp_model, task_id):
        entity_dict = {}
        buf_dataset = self.buffer.get_all_data()
        buf_label_idx = buf_dataset['label_idx_cil']
        buf_label_idx = buf_label_idx.masked_fill(buf_label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
        buf_input_ids = buf_dataset['input_ids']
        buf_attention_mask =  buf_dataset['attention_mask']
        buf_features = []
        for i in range(0, buf_input_ids.shape[0]):
            sentence = buf_input_ids[i].unsqueeze(0)
            _attention_mask = buf_attention_mask[i].unsqueeze(0)
            with torch.no_grad():
                _buf_features = warp_model.model.forward(**{
                                                'input_ids':sentence,
                                                'attention_mask':_attention_mask},
                                                output_hidden_states=True).hidden_states[-1]
                _buf_features = F.normalize(_buf_features, dim=-1)
            buf_features.append(_buf_features)
        buf_features = torch.cat(buf_features, dim=0)
        for i in range(0, buf_label_idx.shape[0]):
            for j in range(0, buf_label_idx.shape[1]):
                old_class = buf_label_idx[i][j].item()
                if old_class >= 0 and old_class < self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id]:
                    if old_class not in entity_dict:
                        entity_dict[old_class] = []
                    if len(entity_dict[old_class]) < self.params.per_class_samples:
                        entity_dict[old_class].append(buf_features[i, j, :])

        with torch.no_grad():
            for lm_input in self.train_loader_list[task_id]: 
                extracted_features = warp_model.model.forward(**{
                                                'input_ids':lm_input['input_ids'],
                                                'attention_mask':lm_input['attention_mask']},
                                                output_hidden_states=True).hidden_states[-1]
                extracted_features = F.normalize(extracted_features, dim=-1)
                
                label_idx = lm_input['label_idx_cil']
                label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
                for i in range(0, label_idx.shape[0]):
                    for j in range(0, label_idx.shape[1]):
                        old_class = label_idx[i][j].item()
                        if old_class > 0 and old_class < self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id]:
                            if old_class not in entity_dict:
                                entity_dict[old_class] = []
                            entity_dict[old_class].append(extracted_features[i, j, :])

        prototype = {}
        for k, v in entity_dict.items():
            pc = torch.mean(torch.stack(v, dim=0), dim=0)
            if k not in prototype:
                prototype[k] = pc 
        
        prototype = dict(sorted(prototype.items()))
        
        return prototype
        
    def get_prototype_dists(self, entity_dict):
        prototype_dists = []
        for k, v in entity_dict.items():
            distance = []
            for i in range(0, len(v)):
                for j in range(i+1, len(v)):
                    cos_similarity = F.cosine_similarity(v[i], v[j], dim=0)
                    distance.append(cos_similarity.item())
            prototype_dists.append(float(sum(distance)/len(distance)))
        print(prototype_dists)
        prototype_dists_tensor = torch.tensor(prototype_dists)
        return prototype_dists_tensor
    
    def relabel(self, lm_input, wrap_teacher_model, label_idx):
        with torch.no_grad():
            extracted_feature = obtain_features(params=self.params, 
                                                        model=wrap_teacher_model.model, 
                                                        lm_input=lm_input, 
                                                        tokenizer=self.tokenizer)
            extracted_feature = F.normalize(extracted_feature.view(-1, extracted_feature.shape[-1]), dim=1)
            
        for i in range(0, extracted_feature.shape[0]):
            cos_similarity = []
            if label_idx[i] == 0:
                for k, v in self.prototype.items():
                    cos_similarity.append(F.cosine_similarity(extracted_feature[i], v, dim=0))
                cos_similarity = torch.tensor(cos_similarity) 
                cos_value, cos_index = torch.max(cos_similarity, dim=0)
                # if cos_index.item() > 2:
                #     print(cos_similarity)
                #     print(cos_index)
                if cos_value > self.theta_proto :
                    label_idx[i] = cos_index #+ 1

        return label_idx
                
    
    def compute_supcon_loss(self, features, labels, topk_th, pseudo_labels=None, entity_top_emissions=None):

        supcon_loss_fct = SupConLoss(temperature=0.1, topk_th=topk_th)
        if pseudo_labels is None:
            pseudo_labels = labels
            
        
        pseudo_labels = pseudo_labels.view(-1)
        supcon_loss = supcon_loss_fct(features, pseudo_labels,
                                              entity_topk=entity_top_emissions)
        
        return supcon_loss
        
    
    def compute_supcon_o_loss(self, features, labels, top_emissions, topk_th, pseudo_labels=None, entity_top_emissions=None, negative_top_emissions=None):
        supcon_o_loss_fct = SupConLoss_o(temperature=0.1, topk_th=topk_th)
        if pseudo_labels is None:
            pseudo_labels = labels
        pseudo_labels = pseudo_labels.view(-1)
        supcon_o_loss = supcon_o_loss_fct(features, pseudo_labels, top_emissions,
                                                  negative_top_emissions,
                                                  entity_topk=entity_top_emissions)
        return supcon_o_loss
    
    def compute_bce_loss(self, logits, label_idx, num_labels, old_target=None):
        bce_loss_fct = BceLoss(o_weight=None)
        bce_loss = bce_loss_fct(logits, label_idx, num_labels, old_target)
        return bce_loss
        
class BceLoss(Module):
    def __init__(self, o_weight=None):
        super(BceLoss, self).__init__()
        self.o_weight = o_weight
    def forward(self, output, labels, num_labels, old_target=None, cal_O=False):
        pad_token_label_id = CrossEntropyLoss().ignore_index

        pos_weight = torch.ones(num_labels).to(output.device)
        if self.o_weight is not None:
            pos_weight[0] = self.o_weight
        if not cal_O:
            pos_weight = pos_weight[1:]
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        idx = torch.where(labels >= 0)[0]
        valid_output = output[idx] # (b*l, num_labels)
        valid_labels = labels[idx]
        target = get_one_hot(valid_labels, num_labels) # (b*l,num_labels)

        if old_target is None:
            if not cal_O:
                target = target[:, 1:]
                valid_output = valid_output[:, 1:]
            return criterion(valid_output, target)

        else:
            old_target = old_target[idx]
            old_target = torch.sigmoid(old_target)
            old_task_size = old_target.shape[1]
            mask_O = (valid_labels < old_task_size).view(-1,1).repeat(1, num_labels) # (b*l,num_labels)
            mask_new = (valid_labels >= old_task_size).view(-1,1).repeat(1, num_labels)

            target_new = target.clone()
            target_new[:, :old_task_size] = 0
            target_new = target_new * mask_new

            target_O = target.clone()
            target_O[:, :old_task_size] = old_target
            target_O = target_O * mask_O

            target = target_new + target_O

            if not cal_O:
                target = target[:, 1:]
                valid_output = valid_output[:, 1:]

            return criterion(valid_output, target)
        
class SupConLoss_o(Module):
    # contrastive learning for "O"
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, topk_th=False):
        super(SupConLoss_o, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.topk_th = topk_th
    def forward(self, features, labels=None, topk=None, negative_topk=None,entity_topk=None,
                mask=None, ignore_index=CrossEntropyLoss().ignore_index, per_types=6,
                aug_feature=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feat_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [batch_size * n_views, feat_dim]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # [batch_size_new * n_views, batch_size * n_views]
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # choose positive and negative token for type 'O'
        if negative_topk is None and topk is not None:
            topk = topk.view(-1, topk.shape[-1])  # (bsz*sql, topk_len)
            # labels == 'O': positive topk
            mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
            if self.topk_th is False:
                mask_o = torch.scatter(torch.zeros_like(mask), 1, topk.long(), 1)
            else:
                mask_o = torch.zeros_like(mask) + topk
            mask = torch.where(labels == 0, mask_o, mask)
            logits_mask[labels.view(-1) == ignore_index] = torch.zeros(logits_mask.shape[1]).to(device)
            idx_no_o = torch.where(labels.view(-1) != 0)[0]
            if self.topk_th is False:
                no_negative = torch.cat((topk,
                                        idx_no_o.expand(topk.shape[0], idx_no_o.shape[0])), dim=1)
                logits_mask_o = torch.scatter(torch.zeros_like(logits_mask), 1,
                                            no_negative.long(), 1)
            else:
                no_negative = idx_no_o.expand(logits_mask.shape[0], idx_no_o.shape[0])
                logits_mask_o = torch.scatter(torch.zeros_like(logits_mask), 1,
                                              no_negative.long(), 1)
                logits_mask_o = logits_mask_o + topk
            logits_mask = torch.where(labels == 0, logits_mask_o, logits_mask)
            logits_mask = torch.scatter(logits_mask, 1,
                                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        if negative_topk is not None and topk is not None:
            topk = topk.view(-1, topk.shape[-1])
            negative_topk = negative_topk.view(-1, negative_topk.shape[-1]) # (bsz*sql, negative_topk_len)
            mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
            mask_o = torch.scatter(torch.zeros_like(mask), 1, topk.long(), 1)
            mask = torch.where(labels == 0, mask_o, mask)
            idx_no_o = torch.where(labels.view(-1) != 0)[0]
            negative_topk = torch.cat((topk, negative_topk,
                                       idx_no_o.expand(negative_topk.shape[0], idx_no_o.shape[0])), dim=1)
            logits_mask_o = torch.scatter(torch.zeros_like(logits_mask), 1, negative_topk.long(), 1)
            logits_mask = torch.where(labels == 0, logits_mask_o, logits_mask)
            logits_mask = torch.scatter(logits_mask, 1,
                                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

            # set different temperature for "O" negative top k tokens
            temp_o = torch.scatter(torch.ones_like(anchor_dot_contrast), 1,
                                   negative_topk.long(), (self.temperature+0.02)/self.temperature)
            temp = torch.where(labels == 0, temp_o, torch.ones_like(anchor_dot_contrast))
            anchor_dot_contrast = torch.div(anchor_dot_contrast, temp)
            # topk_idx = 0
            
        # select negative token for entity types
        if entity_topk is not None:
            num_labels = torch.max(labels.view(-1)).item() + 1
            old_num_labels = num_labels - per_types
            idx_o = torch.where(labels.view(-1) == 0)[0]
            logits_mask_o = torch.scatter(logits_mask, 1,
                                          idx_o.expand(logits_mask.shape[0], idx_o.shape[0]).long(),
                                          0)
            logits_mask_o = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)
            logits_mask_o = torch.scatter(logits_mask_o, 1,
                                          entity_topk.long(), 1)
            if aug_feature is not None:
                logits_mask_o[:, -aug_feature.shape[1]:] =\
                    torch.ones(logits_mask_o.shape[0], aug_feature.shape[1]).to(device)
            logits_mask = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)

        # delete the anchors that don not have positive data
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
        logits_mask[labels.view(-1) == ignore_index] = torch.zeros(logits_mask.shape[1]).to(device)
        idx = torch.where(mask.sum(dim=1)*(labels.view(-1)+1) != 0)[0]
        logits_mask = logits_mask[idx]
        mask = mask[idx]
        logits = logits[idx]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        if torch.isnan(mean_log_prob_pos).sum()>0:
            print("***mean_log_prob_pos is nan***")
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, len(idx)).mean()

        return loss
    
class KdLoss(Module):
    def __init__(self):
        super(KdLoss, self).__init__()

    def forward(self, student_features, teacher_features, t):
        loss_func = KLDivLoss(reduction="batchmean")
        student_old = log_softmax_t(student_features, dim=-1, t=t)
        teacher = softmax_t(teacher_features, dim=-1, t=t)
        loss = loss_func(student_old, teacher)
        return loss

def softmax_t(inputs, dim, t):
    inputs = inputs.div(t)
    Softmax = nn.Softmax(dim=dim)
    outputs = Softmax(inputs)
    return outputs


def log_softmax_t(inputs, dim, t):
    inputs = inputs.div(t)
    LogSoftmax = nn.LogSoftmax(dim=dim)
    outputs = LogSoftmax(inputs)
    return outputs

class SupConLoss(Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, topk_th=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.topk_th = topk_th

    def forward(self, features, labels=None, mask=None, entity_topk=None,
                ignore_index=CrossEntropyLoss(), per_types=6, aug_feature=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feat_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [batch_size * n_views, feat_dim]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)   # [batch_size_new * n_views, batch_size * n_views]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # select anchor
        mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
        logits_mask[labels.view(-1) == ignore_index] = torch.zeros(logits_mask.shape[1]).to(device)
        if entity_topk is not None:
            # logits_mask[labels.view(-1) != 0, labels.view(-1) == 0] = torch.zeros(1).to(device)
            num_labels = torch.max(labels.view(-1)).item()+1
            old_num_labels = num_labels - per_types
            idx_o = torch.where(labels.view(-1) == 0)[0]
            logits_mask_o = torch.scatter(logits_mask, 1,
                                          idx_o.expand(logits_mask.shape[0], idx_o.shape[0]).long(),
                                          0)
            logits_mask_o = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)
            logits_mask_o = torch.scatter(logits_mask_o, 1,
                                          entity_topk.long(), 1)
            if aug_feature is not None:
                logits_mask_o[:, -aug_feature.shape[1]:] = \
                    torch.ones(logits_mask_o.shape[0], aug_feature.shape[1]).to(device)
            logits_mask = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)

        idx = torch.where(labels.view(-1)*mask.sum(dim=-1)*(labels.view(-1)+1) != 0)[0]
        # print(idx)
        logits_mask = logits_mask[idx]
        mask = mask[idx]
        logits = logits[idx]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        if torch.isnan(mean_log_prob_pos).sum()>0:
            print("***mean_log_prob_pos is nan***")
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, logits.shape[0]).mean()

        return loss
    
def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)

    return one_hot

class NNClassification():
    def __init__(self):
        super(NNClassification, self).__init__()

    def nn_classifier_dot(self, reps, support_reps, support_tags):
        batch_size, sent_len, ndim = reps.shape
        # support_reps [bsz*squ_len*baz_num, ndim]
        scores = self._euclidean_metric_dot(reps.view(-1, ndim), support_reps, True)
        emissions = self.get_nn_emissions(scores, support_tags)
        tags = torch.argmax(emissions, 1)
        return tags.view(batch_size, sent_len)
    
    def nn_classifier_dot_score(self, reps, support_reps, support_tags):
        batch_size, sent_len, ndim = reps.shape
        # support_reps [bsz*squ_len*baz_num, ndim]
        scores = self._euclidean_metric_dot(reps.view(-1, ndim), support_reps, True)
        # tags = support_tags[torch.argmax(scores, 1)]
        emissions = self.get_nn_emissions(scores, support_tags)
        tags = torch.argmax(emissions, 1)
        scores_dists = self._euclidean_metric_dot_2(reps.view(-1, ndim), support_reps, True)
        dists = scores_dists.max(1)[0]
        return tags.view(batch_size, sent_len), dists.view(batch_size, sent_len, -1)

    def nn_classifier_dot_prototype(self, reps, support_reps, support_tags, exemplar_means):
        batch_size, sent_len, ndim = reps.shape

        n_tags = torch.max(support_tags) + 1
        feature = reps.view(-1, reps.shape[-1])  # (batch_size, ndim)
        for j in range(feature.size(0)):  # Normalize
            feature.data[j] = feature.data[j] / feature.data[j].norm()
        # feature = feature.unsqueeze(1)  # (batch_size, 1, feature_size)
        means = torch.stack([exemplar_means[cls] for cls in range(n_tags)])# (n_classes, ndim)
        dists = torch.matmul(feature, means.T)  # (batch_size, n_classes)
        # prediction tags and emissions
        dists[:, 0] = torch.zeros(1).to(reps.device)
        emissions, tags = dists.max(1)

        prototype_dists = []
        support_reps = support_reps.view(-1, support_reps.shape[-1])
        for j in range(support_reps.size(0)):  # Normalize
            support_reps.data[j] = support_reps.data[j] / support_reps.data[j].norm()
        support_reps = F.normalize(support_reps)
        for i in range(n_tags):
            support_reps_dists = torch.matmul(support_reps[support_tags==i], means[i].T)
            prototype_dists.append(support_reps_dists.min(-1)[0])
        prototype_dists = torch.stack(prototype_dists).to(reps.device)
        return tags.view(batch_size, sent_len), emissions.view(batch_size, sent_len, -1), prototype_dists


    def get_top_emissions(self, reps, reps_labels, top_k, largest):
        if top_k <= 0:
            return None
        device = (torch.device('cuda')
                  if reps.is_cuda
                  else torch.device('cpu'))
        batch_size, sent_len, ndim = reps.shape
        reps_labels = reps_labels.view(-1)
        scores = self._euclidean_metric_dot_2(reps.view(-1, ndim), reps.view(-1, ndim), True)
        # emissions = self.get_nn_emissions(scores, reps_tags)
        # print(scores.size())
        # mask diag and labels != 'O'
        if largest is True:
            scores = torch.where(reps_labels == 0, scores.double(), -100.)
            scores = torch.scatter(scores, 1,
                                   torch.arange(scores.shape[0]).view(-1, 1).to(device), -100.)
        else:

            scores = torch.where(reps_labels == 0, scores.double(), 100.)
            scores = torch.scatter(scores, 1,
                                   torch.arange(scores.shape[0]).view(-1, 1).to(device), 100.)
        top_emissions = torch.topk(scores, top_k, dim=1, largest=largest).indices
        return top_emissions.view(-1, top_k)
    def get_top_emissions_with_th(self, reps, reps_labels, th_dists):
        # if top_k <= 0:
        #     return None
        batch_size_sent_len, ndim = reps.shape
        reps_labels = reps_labels.view(-1)
        scores = self._euclidean_metric_dot_2(reps.view(-1, ndim), reps.view(-1, ndim), True)
        scores = torch.where(reps_labels == 0, scores.double(), -100.)
        scores = torch.scatter(scores, 1,
                               torch.arange(scores.shape[0]).view(-1, 1).to(reps.device), -100.)

        top_emissions = scores > th_dists
        return top_emissions, scores

    def _euclidean_metric(self, a, b, normalize=False):
        if normalize:
            a = F.normalize(a)
            b = F.normalize(b)
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b) ** 2).sum(dim=2)
        # logits = torch.matmul(a, b.T)
        # logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        # logits = logits-logits_max.detach()
        return logits
    def _euclidean_metric_dot(self, a, b, normalize=False):
        if normalize:
            a = F.normalize(a)
            b = F.normalize(b)
        logits = torch.matmul(a, b.T)
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits-logits_max.detach()
        return logits
    def _euclidean_metric_dot_2(self, a, b, normalize=False):
        if normalize:
            a = F.normalize(a)
            b = F.normalize(b)
        logits = torch.matmul(a, b.T)
        return logits.detach()
    def get_nn_emissions(self, scores, tags):
        """
        Obtain emission scores from NNShot
        """
        n, m = scores.shape
        n_tags = torch.max(tags) + 1
        emissions = -100000. * torch.ones(n, n_tags).to(scores.device)
        for t in range(n_tags):
            mask = (tags == t).float().view(1, -1)
            masked = scores * mask
            # print(masked)
            masked = torch.where(masked < 0, masked, torch.tensor(-100000.).to(scores.device))
            emissions[:, t] = torch.max(masked, dim=1)[0]
        return emissions

class NcmClassification():
    def __init__(self):
        super(NcmClassification, self).__init__()


    def ncm_classifier_dot(self, reps, support_reps, support_labels, exemplar_means):
        n_tags = torch.max(support_labels) + 1

        feature = reps.view(-1, reps.shape[-1])  # (batch_size, feature_size)
        for j in range(feature.size(0)):  # Normalize
            feature.data[j] = feature.data[j] / feature.data[j].norm()
        #feature = feature.unsqueeze(1)  # (batch_size, 1, feature_size)

        means = torch.stack([exemplar_means[cls] for cls in range(n_tags)])  # (n_classes, feature_size)
    
        dists = -torch.matmul(feature, means.T)  # (batch_size, n_classes)
        _, pred_label = dists.min(1)
        pred_label = pred_label.view(reps.shape[0], reps.shape[1])
        return pred_label