import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
from seqeval.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_word_level_acc_with_classifier
from models.Base import BaseLearner


logger = logging.getLogger()

def get_SpanKL_params(parser):
    '''
        The parameters of model SpanKL
    '''
    parser.add_argument("--SpanKL_student_temperate", type=float, default=1, help="The temperature of student model")
    parser.add_argument("--SpanKL_teacher_temperate", type=float, default=1, help="The temperature of teacher model")
    parser.add_argument("--SpanKL_distill_weight", type=float, default=1, help="The weight of the distillation loss")
    parser.add_argument("--SpanKL_use_task_embed", type=bool, default=True, help="use sparse loss or not")
    parser.add_argument("--SpanKL_hidden_size", type=float, default=50, help="the hidden size of entity")
    


class SpanKL(BaseLearner):
    '''
         a Span-based model with Knowledge distillation
        
        A Neural Span-Based Continual Named Entity Recognition Model(2023 AAAI)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['Linear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'SpanKL')
        assert params.il_mode in ['CIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'SpanKL')
        assert params.classification_type == 'word-level', 'NotImplemented for classification_type %s'%(params.classification_type)
        assert not params.is_replay, 'NotImplemented for data replay!'

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        
        self.model, self.tokenizer = get_backbone(self.params)

    def build_classifier(self):
        feature_dim = self.model.module.config.hidden_size if hasattr(self.model,'module') else self.model.config.hidden_size
        self.output_dim_per_task_lst = [2 * self.params.SpanKL_hidden_size * (num_ents // 2) for num_ents in self.CL_dataset.continual_config['CUR_NUM_CLASS']]
        self.classifier_list = nn.ModuleList([torch.nn.Linear(feature_dim, output_dim_per_task) for output_dim_per_task in self.output_dim_per_task_lst])  # one task one independent layer
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.distill_loss = None

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

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
        
        # logits = logits.reshape(-1,logits.shape[-1])
        label_idx = lm_input['label_idx_cil']
        # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
        label_idx = label_idx.masked_fill(label_idx>=self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][task_id],0) 
        if not self.params.is_replay:
            label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0) 
        
        batch_size = label_idx.shape[0]
        seqlen = label_idx.shape[1]
        
        one_hot_label_idx = self.convert_to_one_hot_label(lm_input, label_idx).to(extracted_feature.device)
        
        mask = self.generate_mask(lm_input, batch_size, seqlen)

        logits = self.task_layer_forward(extracted_feature)
        span_logits = self.span_matrix_forward(logits)  # [bsz*num_spans, ent] torch.Size([32, 40, 40, 6])
        
        if task_id == 0:
            # Training with Binary Cross-Entropy Loss 
            # transform label_idx to one-hot label
            total_loss = self.take_loss(task_id, span_logits[mask], one_hot_label_idx[mask], bsz=batch_size) 
        else:
            # Training with Cross-Entropy Loss + Distillation Loss
            # Compute teacher's logits (only for task from 0 to task_id-1)
            with torch.no_grad():
                extracted_feature_teacher = obtain_features(params=self.params, 
                                                            model=wrap_teacher_model.model, 
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)

            logits_teacher = self.task_old_layer_forward(wrap_teacher_model, extracted_feature_teacher)
            span_logits_teacher = self.span_matrix_forward(logits_teacher)
            
            ce_mask = torch.logical_and(label_idx!=-100,label_idx!=0)
            distill_mask = (label_idx==0)
            
            if torch.sum(ce_mask).item()>0:
                ce_loss = self.take_loss(task_id, span_logits[mask], one_hot_label_idx[mask], bsz=batch_size) 
            else:
                ce_loss = torch.tensor(0.).to(logits.device)

            #get old class dims
            start, end = self.compute_begin_end(task_id - 1)
            if torch.sum(distill_mask).item()>0:
                span_logits = span_logits[mask]
                span_logits = span_logits[:, :end]
                span_logits = torch.stack([span_logits, -span_logits], dim=-1)#[6775, 16, 2]
                
                span_logits_teacher = span_logits_teacher[mask]
                span_logits_teacher = span_logits_teacher[:, :end]
                span_logits_teacher = torch.stack([span_logits_teacher, -span_logits_teacher], dim=-1)

                distill_loss = torch.nn.functional.kl_div(torch.nn.functional.logsigmoid(span_logits/self.params.SpanKL_student_temperate)+ 1e-10,
                                                torch.nn.functional.logsigmoid(span_logits_teacher/self.params.SpanKL_teacher_temperate)+ 1e-10, log_target=True)
                distill_loss = torch.sum(distill_loss, -1)  # kl definition
                distill_loss = torch.mean(distill_loss, -1)  # over ent
                distill_loss = torch.sum(distill_loss, -1)  # over spans
                distill_loss = distill_loss / batch_size
            else:
                distill_loss = torch.tensor(0.).to(logits.device)

            total_loss = ce_loss + self.params.SpanKL_distill_weight*distill_loss
            
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

        acc = self.evaluate_word_level_acc_with_bce_classifier(
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
    
    def span_matrix_forward(self, output):
        bsz, length = output.shape[:2]#torch.Size([1, 60, 3300])
        # decode the output from output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)
        ent_hid_lst = torch.split(output, self.output_dim_per_task_lst, dim=-1)
        start_hid_lst = [torch.chunk(t, 2, dim=-1)[0].reshape([bsz, length, self.params.SpanKL_hidden_size, -1]) for t in ent_hid_lst]  # list of [b,l,hid,ent](for one task)
        end_hid_lst = [torch.chunk(t, 2, dim=-1)[1].reshape([bsz, length, self.params.SpanKL_hidden_size, -1]) for t in ent_hid_lst]  # list of [b,l,hid,ent](for one task)
        start_hidden = torch.cat(start_hid_lst, dim=-1)  # b,l,h,e
        end_hidden = torch.cat(end_hid_lst, dim=-1)  # # b,l,h,e
        start_hidden = start_hidden.permute(0, 3, 1, 2)  # b,e,l,h
        end_hidden = end_hidden.permute(0, 3, 1, 2)  # b,e,l,h

        total_ent_size = start_hidden.shape[1]

        attention_scores = torch.matmul(start_hidden, end_hidden.transpose(-1, -2))  # [bat,num_ent,len,hid] * [bat,num_ent,hid,len] = [bat,num_ent,len,len]
        attention_scores = attention_scores / math.sqrt(self.params.SpanKL_hidden_size)
        span_ner_mat_tensor = attention_scores.permute(0, 2, 3, 1)  # b,l,l,e

        self.batch_span_tensor = span_ner_mat_tensor
        # print(span_ner_mat_tensor.shape)

        # len_mask = self.sequence_mask(seq_len)  # b,l
        # matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask pad为0
        # score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 下三角0 上三角和对角线1
        # span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask[..., None])  # 只取True或1组成列表
        # span_ner_pred_lst = span_ner_pred_lst.view(-1, total_ent_size)  # [*,ent]

        return span_ner_mat_tensor
    
    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        """ mask 句子非pad部分为 1"""
        if maxlen is None:
            maxlen = torch.max(lengths)
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask
    
    def task_layer_forward(self, encoder_output):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.CL_dataset.continual_config['NUM_TASK']):
            input_per_task = encoder_output
            output_per_task = self.classifier_list[task_id].to(encoder_output.device)(input_per_task)# torch.Size([32, 40, 100])
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]

        return output_per_task_lst
    
    def task_old_layer_forward(self, warp_model, encoder_output):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.CL_dataset.continual_config['NUM_TASK']):
            input_per_task = encoder_output
            output_per_task = warp_model.classifier_list[task_id].to(encoder_output.device)(input_per_task)# torch.Size([32, 40, 100])
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]

        return output_per_task_lst
    
    def take_loss(self, task_id, batch_predict, batch_target, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        ofs_s, ofs_e= self.compute_begin_end(task_id)
        # print("predict", batch_predict[:, ofs_s:ofs_e].shape)#[169, 3]
        # print(batch_target[:, ofs_s:ofs_e].shape)#[13,3]
        # print(batch_predict.shape)
        # print(batch_target.shape)
        # exit()
        loss = self.calc_loss(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])  # 只计算对应task的头

        if bsz is not None:
            loss = loss / bsz
        return loss
    
    def compute_begin_end(self, task_id):
        ofs_s = self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]//2 #[0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        ofs_e = self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]//2 + self.CL_dataset.continual_config['CUR_NUM_CLASS'][task_id]//2
        
        return ofs_s, ofs_e
    def calc_loss(self, batch_span_pred, batch_span_tgt):
        # pred  batch_span_pred [num_spans,ent]
        # label batch_span_tgt  [num_spans,ent]
        span_loss = self.bce_loss(batch_span_pred, batch_span_tgt)  # [*,ent] [*,ent](target已是onehot) -> [*,ent]
        # ipdb.set_trace()
        # span_loss = torch.sum(span_loss, -1)  # [*] over ent
        span_loss = torch.mean(span_loss, -1)  # [*] over ent
        # span_loss = torch.mean(span_loss)  
        span_loss = torch.sum(span_loss, -1)  # [] over num_spans
        # span_loss = torch.mean(span_loss, -1)  # [] over num_spans
        return span_loss  # []
    
    def convert_to_one_hot_label(self, lm_input, label_idx):
        entity_dict = {}
        # label_idx_clone = label_idx[lm_input['attention_mask']].clone()
        batch_size, seqlen = label_idx.shape
        seq_mask = (lm_input['attention_mask'] == 0)
        label_idx[seq_mask] = 0
        is_entity = False
        # print(label_idx)
        temp_entity = -1
        
        for i in range(0, batch_size):
            start = -1
            end = -1
            temp_entity = -1
            for j in range(0, seqlen):
                label = int(label_idx[i][j].item())
                if label%2 == 1:# Begin-entity
                    if label not in entity_dict:
                        entity_dict[label] = []
                    if is_entity:
                        entity_dict[label].append((i, start, end))
                        is_entity = False    
                        temp_entity = -1
                    temp_entity = label    
                    start = j
                    end = j+1
                    is_entity = True
                if label == 0:# end to O
                    end = j
                    if temp_entity > 0:
                        entity_dict[temp_entity].append((i, start, end))
                    is_entity = False
                    temp_entity = -1

        entity_size = (self.CL_dataset.continual_config["NUM_CLASS"] - 1) // 2 # (33-1) //2 = 16
        span_one_hot_label = torch.zeros((batch_size, seqlen, seqlen, entity_size), dtype=torch.float32)
        for k, v in entity_dict.items():
            k = (k - 1) // 2
            for i, start, end in v:
                span_one_hot_label[i][start][end][k] = 1
        
        # print(torch.sum(torch.sum(span_one_hot_label.view(-1, entity_size), dim=-1)))    
        return span_one_hot_label
    
    def generate_mask(self, lm_input, batch_size, seqlen):
        sentence_length = torch.sum(lm_input['attention_mask'], dim=-1)
        remove_pad_mask = torch.zeros((batch_size, seqlen, seqlen), dtype=torch.float32)
        for i, _seqlen in enumerate(sentence_length):
            remove_pad_mask[i, :_seqlen, :_seqlen] = 1
        
        up_trimatrix = torch.triu(remove_pad_mask, diagonal=0)
        
        mask = torch.logical_and(remove_pad_mask, up_trimatrix)
        
        return mask
        
    # ===========================================================================================
    
    
    def evaluate_word_level_acc_with_bce_classifier(self, model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
        '''
            Evaluate the accuracy with classifier for word-lvel classification tasks

            Return:
            - macro f1 (%)
        '''
        model.eval()
        new_idx2label = {-100: 'O'}
        for i in range(1, len(idx2label)):
            index = (i - 1) // 2 #0 1 2 3 4 5
            if index not in new_idx2label:
                new_idx2label[index] = idx2label[i]
                
                
        with torch.no_grad():
            # Only test once because the last test set contains all labels
            gold_lines = []
            pred_lines = []
            for lm_input in eval_data_loader: 

                extracted_features = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input,
                                                    tokenizer=tokenizer)
                

                label_idx = lm_input['label_idx_cil']
                logits = self.task_layer_forward(extracted_features)
                span_logits = self.span_matrix_forward(logits)  # [bsz*num_spans, ent] torch.Size([6775, 6])
                ofs_s, ofs_e= self.compute_begin_end(cur_task_id)
                span_logits = torch.sigmoid(span_logits)
                
                sequence_f1 = False
                if sequence_f1:#f1:12.57
                    span_logits = span_logits[:, :, :, :ofs_e]
                    max_values, max_indices = torch.max(span_logits, dim=-1, keepdim=True)
                    mask = torch.eq(span_logits, max_values).float()
                    span_logits= span_logits* mask
                    # print(span_logits.shape)
                    positions = torch.nonzero(span_logits > 0.5)
                    preds = torch.full_like(label_idx, -100)
                    for i,j,k,e in positions:
                        if k > j:
                            preds[i][j:k] = e
                span_f1 = True
                if span_f1:
                    batch_size = extracted_features.shape[0]
                    seqlen = extracted_features.shape[1]
                    mask = self.generate_mask(lm_input, batch_size, seqlen)
                    
                    one_hot_label = self.convert_to_one_hot_label(lm_input, label_idx)
                    one_hot_label = one_hot_label[mask]#include vector in all 0 [1424, 16]

                    label_max_values , label_max_indices = torch.max(one_hot_label, dim=-1, keepdim=True)#[1424, 1]
                    label_idx = torch.full_like(label_max_indices, -100)
                    label_positions = (label_max_values == 1)
                    label_idx[label_positions] = label_max_indices[label_positions]
                    
                    span_logits = span_logits[mask]
                    span_logits = span_logits[:, :ofs_e]
                    max_values, max_indices = torch.max(span_logits, dim=-1, keepdim=True)
                    positions = (max_values > 0.5)
                    preds = torch.full_like(max_indices, -100)
                    
                    preds[positions] = max_indices[positions]

                # Gather from different processes in DDP
                preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
                preds, label_idx = preds.cpu(), label_idx.cpu()
                if sequence_f1:
                    for _sent_gold, _sent_pred in zip(label_idx,preds):
                        gold_lines.append([idx2label[_gold.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])
                        pred_lines.append([new_idx2label[_pred.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])  
                if span_f1:
                    for _sent_gold, _sent_pred in zip(label_idx,preds):
                        if _sent_gold.item()!=-100:
                            gold_lines.append(new_idx2label[_sent_gold.item()])
                            pred_lines.append(new_idx2label[_sent_pred.item()])
                # print("gold_lines:", gold_lines)

            gold_lines = [gold_lines]
            pred_lines = [pred_lines]
            # micro f1 (average over all instances)
            mi_f1 = f1_score(gold_lines, pred_lines)*100
            # macro f1 (default, average over each class)
            ma_f1 = f1_score(gold_lines, pred_lines, average=None)*100
            
            print("ma_f1:", ma_f1)
            ma_f1 = ma_f1[:ofs_e].mean()
            # ma_f1 = f1_score(gold_lines, pred_lines, average='macro')*100

            logger.info('Evaluate: ma_f1 = %.2f, mi_f1=%.2f'%(ma_f1,mi_f1))
        
            model.train()

        return ma_f1
