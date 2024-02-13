import numpy as np
import logging
from copy import deepcopy
import torch
from torch.utils.data import ConcatDataset, DataLoader
from datasets import Dataset

from utils.metric import ResultSummary
from utils.backbone import get_backbone
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.evaluation import evaluate_sent_level_acc_with_generation
from models.Base import BaseLearner

logger = logging.getLogger()

def get_EWC_params(parser):
    '''
        The parameters of model EWC and WarmupAndFix
    '''
    parser.add_argument("--EWC_lambda", type=float, default=5000, help="The weight for the regularization term")


class EWC(BaseLearner):
    '''
        Training a model sequentially with l2 regularization
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['None'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'EWC')
        assert params.il_mode in ['IIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'EWC')
        assert params.classification_type in ['sentence-level'], 'NotImplemented for classification type %s'%(params.classification_type)


    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)
        self.old_model = None
        self.fisher = None

    def build_classifier(self):
        self.classifier_list = None

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)
        # Mapping ENCODER/CLASSIFIER and task_id to the index in the param_groups of the optimizer
        self.param_groups_mapping = {}
        self.param_groups_mapping['encoder'] = 0

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
    
    def accelerate_prepare(self):
        self.model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
        self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)

        
    def end_task(self, task_id):
        super().end_task(task_id)

        # Prepare teacher model
        if self.old_model is not None:
            self.old_model.cpu()
            del self.old_model
        self.old_model = deepcopy(self.model)

        if task_id>0: 
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()

        self.fisher=self.fisher_matrix_diag(self.train_loader_list[task_id])
        if task_id>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, 
            # therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*task_id)/(task_id+1)  
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

        # Compute loss
        # Training with Causal Language Modeling Loss
        if task_id==0:
            total_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                    'attention_mask':lm_input['attention_mask_with_ans'],
                                    'labels':lm_input['labels_with_ans']}).loss
        else:
            # classification loss
            causal_lm_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                    'attention_mask':lm_input['attention_mask_with_ans'],
                                    'labels':lm_input['labels_with_ans']}).loss
            
            # regularization loss
            regularization_loss = 0
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),
                                                  self.old_model.named_parameters()):
                regularization_loss+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

            total_loss = causal_lm_loss + self.params.EWC_lambda*regularization_loss

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
        
        model = self.model

        acc = evaluate_sent_level_acc_with_generation(
            model=model,
            eval_data_loader=data_loader[eval_task_id],
            tokenizer=self.tokenizer,
            accelerator=self.accelerator,
            params=self.params,
            idx2label=self.CL_dataset.continual_config['idx2label']
        )

        return  acc
    # ===========================================================================================

    # ======================== Other Model-Specific Functions ===================================
    def fisher_matrix_diag(self, dataloader):
        '''
            Compute Fisher Matrix for EWC
        '''
        # Compute
        self.model.train()
        self.model.cuda()

        # Init
        fisher={}
        for n,p in self.model.named_parameters():
            fisher[n]=0*p.data
        
        total_cnt = 0
        for lm_input in dataloader:
            
            batch_size = lm_input['input_ids_with_ans'].shape[0]
            total_cnt += batch_size
            # Forward and backward
            self.model.zero_grad()
            loss = self.model(**{'input_ids':lm_input['input_ids_with_ans'], 
                            'attention_mask':lm_input['attention_mask_with_ans'],
                            'labels':lm_input['labels_with_ans']}).loss
            loss.backward()
            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n]+=batch_size*p.grad.data.pow(2)
        # Mean
        for n,_ in self.model.named_parameters():
            fisher[n]=fisher[n]/total_cnt
            fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)

        return fisher
    