import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import scipy
from torch.utils.data import DataLoader


from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_CFNER_params(parser):
    '''
        The parameters of model CFNER
    '''
    parser.add_argument("--CFNER_student_temperate", type=float, default=1, help="The temperature of student model")
    parser.add_argument("--CFNER_teacher_temperate", type=float, default=1, help="The temperature of teacher model")
    parser.add_argument("--CFNER_distill_weight", type=float, default=2, help="The weight of the distillation loss")
    parser.add_argument("--is_fix_trained_classifier", type=bool, default=True, help="Fix old classifier")
    parser.add_argument("--is_DCE", type=bool, default=True, help="use DCE")
    parser.add_argument("--is_ODCE", type=bool, default=True, help="use ODCE")
    parser.add_argument("--CFNER_top_k", type=float, default=3)
    parser.add_argument("--CFNER_adaptive_distill_weight", type=bool, default=True)
    parser.add_argument("--CFNER_adaptive_schedule", type=str, default="root")
    parser.add_argument("--CFNER_max_seq_length", type=float, default=128, help="The max length of the sentence")
    


class CFNER(BaseLearner):
    '''
        a unified causal framework to retrieve the causality from both newentity types and Other-Class
        Distilling Causal Effect from Miscellaneous Other-Class for Continual Named Entity Recognition(2022 EMNLP)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'CFNER')
        assert params.il_mode in ['CIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'CFNER')
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
        
        self.train_loader_list[task_id] = self.unshuffle_dataloader(self.train_loader_list[task_id])

        # Prepare teacher model
        if task_id>0:
            if self.wrap_teacher_model is not None:
                self.wrap_teacher_model.cpu()
                del self.wrap_teacher_model
            self.wrap_teacher_model = deepcopy(self.wrap_model)
            
            
        
        if task_id>0 and self.params.classifier!='None' and self.params.is_fix_trained_classifier:
            self.optimizer.param_groups[self.param_groups_mapping['classifier'][task_id-1]]['lr']=0
            
        if task_id>0 and (self.params.is_DCE or self.params.is_ODCE):    
            self.pos_matrix, self.O_pos_matrix, self.match_id, self.O_match_id = self.compute_O_match_id(task_id, self.train_loader_list, self.wrap_teacher_model)

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
        
        self.sample_id, self.O_sample_id, self.sentence_id = 0, 0, 0

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
        self.logits = logits
        label_idx = lm_input['label_idx_cil']
        # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
        label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
        if not self.params.is_replay:
            label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
        
        match_id_batch, O_match_id_batch = None, None
        # Use DCE
        if task_id>0 and self.params.is_DCE:
            batch_sent_ids = list(range(self.sentence_id,self.sentence_id+label_idx.shape[0]))
            # count the number of entities (not O) in the batch
            num_samples_batch = np.count_nonzero(np.isin(self.pos_matrix[:,0],batch_sent_ids))
            # get the reference feature and the match reference feature
            match_id_batch = self.match_id[self.sample_id*self.params.CFNER_top_k:(self.sample_id+num_samples_batch)*self.params.CFNER_top_k]
            # update count number
            self.sample_id += num_samples_batch

        # Use ODCE
        if task_id>0 and self.params.is_ODCE and len(self.O_match_id)>0:
            batch_sent_ids = list(range(self.sentence_id, self.sentence_id+label_idx.shape[0]))
            # count the number of O sampls in the batch
            num_O_sample_batch = np.count_nonzero(np.isin(self.O_pos_matrix[:,0],batch_sent_ids))
            # compute the O_pos_matrix_batch
            O_pos_matrix_batch = self.O_pos_matrix[np.isin(self.O_pos_matrix[:,0],batch_sent_ids)]
            O_pos_matrix_batch[:,0] = O_pos_matrix_batch[:,0]-self.sentence_id
            self.O_pos_matrix_batch = O_pos_matrix_batch
            # get the reference feature and the match reference feature
            O_match_id_batch = self.O_match_id[self.O_sample_id*self.params.CFNER_top_k:(self.O_sample_id+num_O_sample_batch)*self.params.CFNER_top_k]
            # update count number
            self.O_sample_id += num_O_sample_batch
            
        self.compute_logits_match(task_id, match_id_batch, O_match_id_batch, max_seq_length=self.params.CFNER_max_seq_length)
        
        
        if task_id == 0:
            # Training with Cross-Entropy Loss only

            total_loss = self.ce_loss(logits.view(-1, logits.shape[-1]), label_idx.view(-1))     
        else:
            # Training with Cross-Entropy Loss + Distillation Loss
            # Compute teacher's logits (only for task from 0 to task_id-1)
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                logits_teacher = torch.concatenate([classifier(extracted_feature_teacher) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                # logits_teacher = logits_teacher.reshape(-1,logits_teacher.shape[-1])
            ce_mask = torch.logical_and(label_idx!=-100,label_idx!=0)
            distill_mask = (label_idx==0)

            if torch.sum(ce_mask).item()>0:
                if self.params.is_DCE and self.logits_match!=None:
                    ce_loss = self.compute_DCE(logits, label_idx, ce_mask)
                else:
                    ce_loss = self.ce_loss(logits[ce_mask],label_idx[ce_mask])
            else:
                ce_loss = torch.tensor(0.).to(logits.device)
            
            if torch.sum(distill_mask).item()>0:
                if self.params.is_ODCE and self.O_logits_match!=None:
                    distill_loss = self.compute_ODCE(logits, logits_teacher, distill_mask)
                else:
                    distill_loss = self.distill_loss(F.log_softmax(logits[distill_mask]/self.params.CFNER_student_temperate,dim=-1)[:,:logits_teacher.shape[-1]],
                                                F.softmax(logits_teacher[distill_mask]/self.params.CFNER_teacher_temperate,dim=-1))
            else:
                distill_loss = torch.tensor(0.).to(logits.device)

            total_loss = ce_loss + self.params.CFNER_distill_weight*distill_loss

        
        self.sentence_id += label_idx.shape[0]

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
    
    def compute_O_match_id(self, task_id, train_loader_list, wrap_teacher_model):
        # Compute match samples for DCE
        dataloader_train = train_loader_list[task_id]
        
        new_classes_list = self.CL_dataset.continual_config['CUR_CLASS'][task_id]
        O_index = self.CL_dataset.continual_config['label2idx']["O"]
        # (1) ref feature model is fixed, get old features z0
        # 1.1 Compute the feature (flatten) and collect all data (not flatten)
        (refer_flatten_feat_train, refer_flatten_feat_O_train) \
            = self.compute_feature_by_dataloader(dataloader=dataloader_train,
                                            feature_model=wrap_teacher_model.model, 
                                            select_label_groups=[
                                                new_classes_list,
                                                [O_index],
                                            ],
                                            is_normalize=True)
        self.train_loader_list[task_id] = dataloader_train        
        # 1.2 flatten label list and compute the neighbor for each sample
        label_list = []
        for lm_input in self.train_loader_list[task_id]:
            label_idx = lm_input['label_idx_cil']
            for i in range(0, label_idx.shape[0]):
                label_list.append(label_idx[i])
        
        if self.params.is_DCE:
            flatten_label_train, pos_matrix = self.get_flatten_for_nested_list(
                                            label_list, 
                                            select_labels=new_classes_list,
                                            is_return_pos_matrix=True,
                                            max_seq_length=self.params.CFNER_max_seq_length)

            num_samples_all = len(flatten_label_train)
            assert refer_flatten_feat_train.shape[0] == num_samples_all, \
                    "refer_flatten_feat_train.shape[0]!=num_samples_all !!!"
            # compute the neighbor for each sample
            match_id = self.get_match_id(refer_flatten_feat_train, self.params.CFNER_top_k)
            # save the space
            del refer_flatten_feat_train

        if self.params.is_ODCE:
            _, O_pos_matrix = self.get_flatten_for_nested_list(
                                            label_list, 
                                            select_labels=[O_index],
                                            is_return_pos_matrix=True)

            ground_truth_O_pos_matrix_list = []
            old_class_list = []
            
            refer_flatten_feat_O_train, O_pos_matrix = self.select_O_samples(
                                            task_id,
                                            wrap_teacher_model,
                                            refer_flatten_feat_O_train, 
                                            O_pos_matrix,
                                            ground_truth_O_pos_matrix_list=ground_truth_O_pos_matrix_list,
                                            old_class_list = old_class_list)
            if len(O_pos_matrix)>0:
                num_O_samples_all = refer_flatten_feat_O_train.shape[0]
                # compute the neighbor for each sample
                O_match_id = self.get_match_id(refer_flatten_feat_O_train, self.params.CFNER_top_k)
                # save the space
                del refer_flatten_feat_O_train
            else:
                O_pos_matrix = []
                num_O_samples_all = 0
                O_match_id = []
        return pos_matrix, O_pos_matrix, match_id, O_match_id

        
    def compute_feature_by_dataloader(self, dataloader, feature_model, select_label_groups=[], is_normalize=False):
        '''
            Compute the feature of dataloader{(X, Y)}, X has dims (num_sentences, num_words)

            Params:
                - dataloader: torch.utils.data.DataLoader
                - feature_model: a model returns features (w/o FC layers)
                - select_label_groups: select the features of the chosen labels
                and return groups of features if "select_label_groups" is a nested list 
                (e.g.[1,2,3], or [[0],[1,2,3]])
                - is_normalize: if normalize the features
            Return:
                - features_matrix: a groups of such as [(num_samples, hidden_dims),...]
                according to the select_label_groups
        '''
        feature_model.eval()

        num_groups = len(select_label_groups)
        return_feature_groups = [[] for i in range(num_groups)]

        features_list = []
        with torch.no_grad():
            for lm_input in dataloader:
                extracted_feature = obtain_features(params=self.params, 
                                                model=feature_model, 
                                                lm_input=lm_input, 
                                                tokenizer=self.tokenizer)
                targets = lm_input['label_idx_cil']
                if num_groups>0:
                    for i in range(num_groups):
                        select_mask = torch.zeros_like(targets)
                        for select_label in select_label_groups[i]:
                            select_mask = torch.logical_or(select_mask, targets==select_label)
                        return_feature_groups[i].append(\
                            extracted_feature[select_mask].reshape(-1,extracted_feature.shape[-1]))
                else:
                    features_list.append(extracted_feature.reshape(-1,extracted_feature.shape[-1]))
        feature_model.train()

        if num_groups>0:
            for i in range(num_groups):
                return_feature_groups[i] = torch.cat(return_feature_groups[i], dim=0)
            if is_normalize:
                for i in range(num_groups):
                    return_feature_groups[i] = F.normalize(return_feature_groups[i], p=2, dim=-1)
            features_matrix = tuple(return_feature_groups)
        else:
            features_matrix = torch.cat(features_list, dim=0)
            if is_normalize:
                features_matrix = F.normalize(features_matrix, p=2, dim=-1)

        return features_matrix
    
    def get_flatten_for_nested_list(self, all_label_train, select_labels, is_return_pos_matrix=False, max_seq_length=512):
        '''
            Return a flatten version of the nested_list contains only select_labels,
            and a position matrix. 
            
            Params:
                - all_label_train: a nested list and each element is the token label
                - select_labels: a list indicates the select labels
                - is_return_pos_matrix: if return the pos matirx of each select element,
                e.g., [[1,4],[1,5],[2,1],[2,2],...]
                - max_seq_length: the longest length for each sentence
            Return:
                - flatten_label: a flatten version of the nested_list
                - pos_matrix: a "Numpy" matrix has dims (num_samples, 2)
                and it indicates the position (i-th sentence, j-th token) of each entity
        '''
        flatten_list = []
        pos_matrix = []
        for i,s in enumerate(all_label_train):
            if len(s)>max_seq_length:
                s=s[:max_seq_length]
            s = s.cpu()
            mask_4_sent = np.isin(s, select_labels)
            pos_4_sent = np.where(mask_4_sent)[0]
            flatten_list.extend(np.array(s)[mask_4_sent])
            if is_return_pos_matrix and len(pos_4_sent)>0:
                pos_matrix.append(np.array([[i,j] for j in pos_4_sent]))
        if is_return_pos_matrix:
            if len(pos_matrix)>0:
                pos_matrix = np.concatenate(pos_matrix, axis=0)
            else:
                pos_matrix = []
            return flatten_list, pos_matrix
        return flatten_list
    
    def select_O_samples(self, task_id, wrap_teacher_model, refer_flatten_feat_O_train, O_pos_matrix, sample_strategy='all', sample_ratio=1.0, ground_truth_O_pos_matrix_list=[], old_class_list=[]):
        assert refer_flatten_feat_O_train.shape[0]==O_pos_matrix.shape[0],\
            "refer_flatten_feat_O_train.shape[0]!=O_pos_matrix.shape[0] !!!"
        assert wrap_teacher_model!=None, "refer_model is none !!!"
        assert sample_ratio>0.0 and sample_ratio<=1.0, "invalid sample ratio %.4f!!!"%sample_ratio
        # get the logits of the refer model
        with torch.no_grad():
            wrap_teacher_model.eval()
            O_logits = []
            for O_feature in refer_flatten_feat_O_train.split(64):
                O_feature = O_feature.to(refer_flatten_feat_O_train.device)
                _O_logits = torch.concatenate([classifier(O_feature) for classifier in wrap_teacher_model.classifier_list[:task_id]],dim=-1)
                O_logits.append(_O_logits.cpu())        
        O_logits = torch.cat(O_logits, dim=0)
        O_predicts = torch.argmax(O_logits, dim=-1)
        
        if sample_strategy=='all':
            select_mask = torch.not_equal(O_predicts, self.CL_dataset.continual_config['label2idx']["O"])

        logger.info('Ratio of select samples %.2f%% (%d/%d).'%(\
                torch.sum(select_mask).item()/select_mask.shape[0]*100,
                torch.sum(select_mask).item(),
                select_mask.shape[0]
            )
        )

        return refer_flatten_feat_O_train[select_mask], O_pos_matrix[select_mask]
    
    
    def compute_logits_match(self, task_id, match_id_batch=None, O_match_id_batch=None, max_seq_length=512):    

        self.pad_token_id = -100
        dataloader_train = self.train_loader_list[task_id]
        if match_id_batch!=None and len(match_id_batch)==0:
            self.features_match = None
            self.logits_match = None
        elif match_id_batch!=None and len(match_id_batch)>0:
            assert self.pad_token_id!=None, "pad_token_id is none!"
            assert dataloader_train!=None, "dataloader_train is none!"
            assert self.pos_matrix.any(), "pos_matrix is none!"
            # compute the sentences related to the match samples
            match_id_batch = match_id_batch.cpu()
            match_pos_matrix_batch = torch.tensor(self.pos_matrix[match_id_batch]).view(-1,2)
            select_sentence_idx = match_pos_matrix_batch[:,0]
            unique_sentence_idx = torch.unique(select_sentence_idx)
            select_to_unique_map = [list(unique_sentence_idx).index(i) for i in select_sentence_idx]
            select_sentence_batch = []
            
            input_list = []
            for lm_input in self.train_loader_list[task_id]:
                input_idx = lm_input['input_ids']
                for i in range(0, input_idx.shape[0]):
                    input_list.append(input_idx[i])
            for idx in unique_sentence_idx:
                select_sentence_batch.append(input_list[idx].cpu())
            # pad the sentence batch
            length_lst = [len(s) for s in select_sentence_batch]
            max_length = max(length_lst)
            select_batch = torch.LongTensor(len(select_sentence_batch), max_length).fill_(self.pad_token_id)
            for i, s in enumerate(select_sentence_batch):
                select_batch[i,:len(s)] = torch.LongTensor(s)
            if select_batch.shape[1]>max_seq_length:
                select_batch = select_batch[:,:max_seq_length].clone()
            select_batch = select_batch.to(self.logits.device)
            # compute match feature

            with torch.no_grad():
                self.wrap_model.eval()
                tmp_features_match_lst = []
                slice_nums = self.logits.shape[0]
                for _select_batch in select_batch.split(slice_nums):
                    tmp_features = self.wrap_model.model.forward(**{
                            'input_ids':_select_batch},
                            output_hidden_states=True).hidden_states
                    tmp_features = tmp_features[-1]
                    tmp_features_match_lst.append(tmp_features)
                tmp_features_match = torch.cat(tmp_features_match_lst, dim=0)
                features_match = torch.FloatTensor(len(match_id_batch),tmp_features_match.shape[-1])
                for i, pos in enumerate(match_pos_matrix_batch):
                    assert len(pos)==2, "pos type is invalid"
                    pos_i = pos[0].item()
                    pos_j = pos[1].item()
                    features_match[i] = tmp_features_match[select_to_unique_map[i]][pos_j]
                self.features_match = features_match.to(self.logits.device)
                self.logits_match = torch.concatenate([classifier(self.features_match) for classifier in self.wrap_model.classifier_list[:task_id+1]],dim=-1)

                self.model.train()         

        if O_match_id_batch!=None and len(O_match_id_batch)==0:
            self.O_features_match = None
            self.O_logits_match = None
        elif O_match_id_batch!=None and len(O_match_id_batch)>0:
            assert self.pad_token_id!=None, "pad_token_id is none!"
            assert self.train_loader_list!=None, "dataloader_train is none!"
            assert self.O_pos_matrix.any(), "O_pos_matrix is none!"
            # compute the sentences related to the match samples
            O_match_id_batch = O_match_id_batch.cpu()
            match_O_pos_matrix_batch = torch.tensor(self.O_pos_matrix[O_match_id_batch])
            if len(self.O_pos_matrix[O_match_id_batch]) > 2:
                select_sentence_idx = match_O_pos_matrix_batch[:,0]
            else:
                select_sentence_idx = match_O_pos_matrix_batch
            unique_sentence_idx = torch.unique(select_sentence_idx)
            select_to_unique_map = [list(unique_sentence_idx).index(i) for i in select_sentence_idx]
            input_list = []
            for lm_input in self.train_loader_list[task_id]:
                input_idx = lm_input['input_ids']
                for i in range(0, input_idx.shape[0]):
                    input_list.append(input_idx[i].cpu())
            
            select_sentence_batch = []
            for idx in unique_sentence_idx:
                select_sentence_batch.append(input_list[idx])
            # pad the sentence batch
            length_lst = [len(s) for s in select_sentence_batch]
            max_length = max(length_lst)
            select_batch = torch.LongTensor(len(select_sentence_batch), max_length).fill_(self.pad_token_id).to(self.logits.device)
            for i, s in enumerate(select_sentence_batch):
                select_batch[i,:len(s)] = torch.LongTensor(s)
            if select_batch.shape[1]>max_seq_length:
                select_batch = select_batch[:,:max_seq_length].clone()
            select_batch = select_batch.to(self.logits.device)
            # compute match feature
            with torch.no_grad():
                self.wrap_model.eval()
                tmp_O_features_match_lst = []
                slice_nums = self.logits.shape[0]
                for _select_batch in select_batch.split(slice_nums):
                    tmp_O_features = self.wrap_model.model.forward(**{
                            'input_ids':_select_batch},
                            output_hidden_states=True).hidden_states
                    tmp_O_features = tmp_O_features[-1]
                    tmp_O_features_match_lst.append(tmp_O_features)
                tmp_O_features_match = torch.cat(tmp_O_features_match_lst, dim=0)
                O_features_match = torch.FloatTensor(len(O_match_id_batch),tmp_O_features_match.shape[-1])
                for i, pos in enumerate(match_O_pos_matrix_batch):
                    assert len(pos)==2, "pos type is invalid"
                    pos_i = pos[0].item()
                    pos_j = pos[1].item()
                    O_features_match[i] = tmp_O_features_match[select_to_unique_map[i]][pos_j]
                self.O_features_match = O_features_match.to(self.logits.device)
                self.O_logits_match = torch.concatenate([classifier(self.O_features_match) for classifier in self.wrap_model.classifier_list[:task_id+1]],dim=-1)
                self.wrap_model.train()
    
    def compute_DCE(self, logits, labels, ce_mask):
        '''
            DCE for labeled samples
        '''
        assert torch.sum(ce_mask.float()) == int(self.logits_match.shape[0]/self.params.CFNER_top_k) \
                and self.logits_match.shape[0]%self.params.CFNER_top_k == 0, \
                "length of ce_mask and the number of match samples are not equal!!!"
        # joint ce_loss
        logits_prob = F.softmax(logits.view(-1, logits.shape[-1]), dim=-1)
        logits_prob_match = F.softmax(self.logits_match, dim=-1)
        # print(logits_prob_match)
        logits_prob_match = torch.mean(logits_prob_match.reshape(-1, self.params.CFNER_top_k, logits_prob_match.size(-1)), dim=1)
        # print(logits_prob_match)
        # print(logits_prob[ce_mask.flatten()])
        logits_prob_joint = (logits_prob[ce_mask.flatten()]+logits_prob_match)/2

        ce_loss = F.nll_loss(torch.log(logits_prob_joint+1e-10), labels[ce_mask])

        return ce_loss

    def compute_ODCE(self, logits, refer_logits, distill_mask):
        '''
            DCE for O samples
        '''
        refer_dims = refer_logits.shape[-1]
        assert self.O_pos_matrix_batch.any(), "O_pos_matrix_batch is none"

        # get mask for defined O samples
        defined_O_mask = torch.zeros(refer_logits.shape[:2], dtype=torch.float32).to(logits.device)
        defined_O_mask[self.O_pos_matrix_batch[:,0], self.O_pos_matrix_batch[:,1]] = 1
        defined_O_mask = torch.logical_and(distill_mask, defined_O_mask)
        # print(self.O_pos_matrix_batch)
        # print(torch.sum(defined_O_mask.float()))
        # print()
        assert torch.sum(defined_O_mask.float()) == int(self.O_logits_match.shape[0]/self.params.CFNER_top_k) \
                and self.O_logits_match.shape[0]%self.params.CFNER_top_k == 0, \
                "length of defined_O_mask and the number of 'O' match samples are not equal!!!"
        
        # get average scores of the matched samples
        # truncate the refer_dims before softmax to mitigate the imblance 
        # between old and new classes 
        O_logits_prob_match = F.softmax(
                            self.O_logits_match[:,:refer_dims]/self.params.CFNER_student_temperate, 
                            dim=-1)
        O_logits_prob_match = torch.mean(O_logits_prob_match.view(-1, self.params.CFNER_top_k, O_logits_prob_match.shape[-1]), dim=1)
        # get scores of the original samples
        old_class_score_all = F.softmax(
                            logits/self.params.CFNER_student_temperate,
                            dim=-1)[:,:,:refer_dims]
        joint_old_class_score_all = old_class_score_all.clone()

    
        # TODO: KLDiv or CE ?
        # Compute KL divergence of distributions
        # 1.log(joint_distribution)
        joint_old_class_score_all[defined_O_mask] = (old_class_score_all[defined_O_mask]+O_logits_prob_match)/2
        joint_old_class_score = torch.log(joint_old_class_score_all[distill_mask]+1e-10).view(-1, refer_dims)
        # 2.ref_distribution
        refer_logits[defined_O_mask] /= 1e-10 # Equals to applying CE to defined O samples, others is KLDivLoss
        ref_old_class_score = F.softmax(
                            refer_logits[distill_mask]/self.params.CFNER_teacher_temperate, 
                            dim=-1).view(-1, refer_dims)
        # KL divergence
        distill_loss = nn.KLDivLoss(reduction='batchmean')(joint_old_class_score, ref_old_class_score)
        
        return distill_loss
    
    def get_match_id(self, flatten_feat_train, top_k, max_samples=5000):
        '''
            Compute the nearest samples id for each sample,

            Params:
                - flatten_feat_train: a matrix has dims (num_samples, hidden_dims)
                - top_k: for each sample, return the id of the top_k nearest samples
                - max_samples: number of maximum samples for computation.
                if it is set too large, "out of memory" may happen.
            Return:
                - match_id: a list has dims (num_samples*top_k) 
                and it represents the ids of the nearest samples of each sample.
        '''
        origin_device = flatten_feat_train.device
        num_samples_all = flatten_feat_train.shape[0]
        if num_samples_all>max_samples:
            # 2.1. calculate the L2 distance inside z0
            dist_z =  scipy.spatial.distance.cdist(flatten_feat_train.cpu(),
                                    flatten_feat_train.cpu()[:max_samples],
                                    'euclidean')
            dist_z = torch.tensor(dist_z)
            # 2.2. calculate distance mask: do not use itself
            mask_input = torch.clamp(torch.ones_like(dist_z).cpu()-torch.eye(num_samples_all, max_samples), min=0)
        else:
            # 2.1. calculate the L2 distance inside z0
            dist_z = self.pdist(flatten_feat_train.cpu(), squared=False)
            # 2.2. calculate distance mask: do not use itself
            mask_input = torch.clamp(torch.ones_like(dist_z).cpu()-torch.eye(num_samples_all), min=0)
        # 2.3 find the image meets label requirements with nearest old feature
        mask_input = mask_input.float() * dist_z
        mask_input[mask_input == 0] = float("inf")
        top_k = min(mask_input.shape[0], top_k)
        match_id = torch.flatten(torch.topk(mask_input, top_k, largest=False, dim=1)[1]).to(origin_device)

        return match_id
    
    def pdist(self, e, squared=False, eps=1e-12):
        '''
            Compute the L2 distance of all features

            Params:
                - e: a feature matrix has dims (num_samples, hidden_dims)
                - squared: if return the squared results of the distance
                - eps: the threshold to avoid negative distance
            Return:
                - res: a distance matrix has dims (num_samples, num_samples)
        '''
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
    
    def unshuffle_dataloader(self, train_data):
        unshuffle_train_dataloader = DataLoader(train_data.dataset, 
                                            batch_size=self.params.batch_size, 
                                            shuffle=False,
                                            drop_last=False)

        unshuffle_train_dataloader = self.accelerator.prepare(unshuffle_train_dataloader)
        
        return unshuffle_train_dataloader