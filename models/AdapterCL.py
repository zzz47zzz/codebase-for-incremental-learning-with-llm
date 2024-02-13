import numpy as np
import logging
import torch
import torch.nn as nn
import transformers
# from transformers import AdapterConfig
from peft import LoraConfig
from copy import deepcopy

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_generate_ids
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from models.Base import BaseLearner

logger = logging.getLogger()

def get_AdapterCL_params(parser):
    '''
        The parameters of model AdapterCL
    '''
    pass
    # parser.add_argument("--PEFT_type", type=str, default='PromptTuning', choices=['PromptTuning','LoRA'], help="The peft type")
    
    # # For PromptTuning (more details can be found in this (url)[https://huggingface.co/docs/peft]
    # parser.add_argument("--PEFT_num_virtual_tokens", type=int, default=10, help="The number of tokens for prompt tuning")
    # parser.add_argument("--PEFT_prompt_tuning_init_text", type=str, default='auto', help="The initialization words for prompt tuning")
    # # For LoRA (more details can be found in this (url)[https://huggingface.co/docs/peft]
    # parser.add_argument("--PEFT_lora_r", type=int, default=4, help="The rank of lora")
    # parser.add_argument("--PEFT_lora_alpha", type=int, default=8, help="The scaling of lora")
    # parser.add_argument("--PEFT_lora_bias", type=str, default="none", help="The bias of lora")
    # parser.add_argument("--PEFT_lora_dropout", type=float, default=0.1, help="The dropout rate of lora")
    # parser.add_argument("--PEFT_lora_target_modules", type=str, nargs='+', default=None, help="The target_modules of lora")


class AdapterCL(BaseLearner):
    '''
        AdapterCL: AdapterCL learns each adapter for each task. During prediction in CIL, the model predict the 
        task id according to causal language modelling loss (PERPLEXITY), which requires N forward passes for each instance.
        This repository is implemented according to the [repository](https://github.com/andreamad8/ToDCL).

        - [Continual Learning in Task-Oriented Dialogue Systems](https://aclanthology.org/2021.emnlp-main.590/)

    '''
    def __init__(self, params, CL_dataset, accelerator):
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier == 'None', 'NotImplemented for classifier %s and model %s'%(params.classifier,'AdapterCL')
        assert params.backbone_type == 'generative', 'NotImplemented for backbone type %s'%(params.backbone_type)
        assert params.il_mode in ['CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'AdapterCL')
        assert params.classification_type == 'sentence-level', 'NotImplemented for classification type %s'%(params.classification_type)
        assert not params.is_replay, 'NotImplemented for data replay!'

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params, num_task=self.CL_dataset.continual_config['NUM_TASK'])
        num_task=self.CL_dataset.continual_config['NUM_TASK']
        for t_id in range(num_task):
            self.model.add_adapter('task-%d'%(t_id),config='pfeiffer')
            # peft_config = self.model.peft_config['default']
            # self.model.add_adapter(adapter_name='task-%d'%(t_id),peft_config=peft_config)
        if hasattr(self.model,'adapter_summary'):
            logger.info(self.model.adapter_summary())

    def build_classifier(self):
        self.ce_loss = nn.CrossEntropyLoss()

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, None)
        
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

        # add a new adapter
        self.model.set_active_adapters('task-%d'%(task_id))
        self.model.train_adapter('task-%d'%(task_id))
        # self.model.set_adapter('task-%d'%(task_id))
        
    def end_task(self, task_id):
        super().end_task(task_id)

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

        # For Distributed Data Parallel
        if hasattr(self.model,'module'):
            model = self.model.module
        else:
            model = self.model

        model.set_active_adapters('task-%d'%(task_id))
        # model.set_adapter('task-%d'%(task_id))
        # Compute loss
        # Training with Causal Language Modeling Loss
        if self.params.classifier == 'None':
            lm_input_labels = deepcopy(lm_input['input_ids_with_ans'])
            lm_input_labels.masked_fill_(lm_input['attention_mask_with_ans']==0,-100) # The label only masks padding to -100 according to AdapterCL

            total_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                    'attention_mask':lm_input['attention_mask_with_ans'],
                                    'labels':lm_input_labels}).loss
            
            # total_loss = model(**{'input_ids':lm_input['input_ids_with_ans'], 
            #                         'attention_mask':lm_input['attention_mask_with_ans'],
            #                         'labels':lm_input['labels_with_ans']}).loss
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
            
        # For evaluation
        if (self.params.evaluate_interval>0) and epoch_id%self.params.evaluate_interval==0:
            il_mode = self.params.il_mode
            acc = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            if self.accelerator.is_main_process:
                logger.info("%s, Current Task %d, Epoch %d, Step %d: Dev_acc=%.3f" % (
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
        if hasattr(self.model,'module'):
            model = self.model.module
        else:
            model = self.model

        if self.params.classifier == 'None':

            model.eval()
            acc_list_all = []
            idx2label = self.CL_dataset.continual_config['idx2label']
            
            with torch.no_grad():
                for lm_input in data_loader[eval_task_id]: 

                    label_idx = lm_input['label_idx_cil']
                    if il_mode == 'TIL':
                        model.set_active_adapters('task-%d'%(eval_task_id))
                        # model.set_adapter('task-%d'%(eval_task_id))
                        generate_ids = obtain_generate_ids(params=self.params, 
                                                            model=model,
                                                            lm_input=lm_input, 
                                                            tokenizer=self.tokenizer)
                    elif il_mode == 'CIL':
                        # Select classifier according to PPL
                        batch_size = lm_input['input_ids'].shape[0]
                        generate_ids_list = []
                        loss_list = -1*torch.ones((batch_size,cur_task_id+1))
                        for t_id in range(cur_task_id+1):
                            model.set_active_adapters('task-%d'%(t_id))
                            # model.set_adapter('task-%d'%(t_id))
                            lm_input_labels = deepcopy(lm_input['input_ids'])
                            lm_input_labels.masked_fill_(lm_input_labels==self.tokenizer.pad_token_id,-100)
                            lm_logits = model(**{'input_ids':lm_input['input_ids'], 
                                            'attention_mask':lm_input['attention_mask'],
                                            'labels':lm_input_labels}).logits
                            # Shift so that tokens < n predict n
                            for bs_i in range(batch_size):
                                shift_logits = lm_logits[bs_i, :-1, :].contiguous()
                                shift_labels = lm_input_labels[bs_i, 1:].contiguous()
                                # Flatten the tokens
                                lm_loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                loss_list[bs_i,t_id] = lm_loss.item()

                        pred_task_id_list = loss_list.argmin(dim=-1) # (bs,)
                        for bs_i in range(batch_size):
                            model.set_active_adapters('task-%d'%(pred_task_id_list[bs_i]))
                            # model.set_adapter('task-%d'%(pred_task_id_list[bs_i]))
                            _generate_ids = obtain_generate_ids(params=self.params, 
                                                                model=model,
                                                                lm_input={
                                                                    'input_ids':lm_input['input_ids'][bs_i].unsqueeze(0), 
                                                                    'attention_mask':lm_input['attention_mask'][bs_i].unsqueeze(0)}, 
                                                                tokenizer=self.tokenizer)[0]
                            generate_ids_list.append(_generate_ids)
                        max_len = max([len(lst) for lst in generate_ids_list])
                        generate_ids = self.tokenizer.pad_token_id*torch.ones((batch_size, max_len),dtype=torch.int64).to(model.device)
                        for bs_i in range(batch_size):
                            generate_ids[bs_i,:len(generate_ids_list[bs_i])] = generate_ids_list[bs_i]

                    else:
                        raise NotImplementedError()

                    # Gather from different processes in DDP
                    generate_ids = self.accelerator.pad_across_processes(
                                                        generate_ids, 
                                                        dim=1, 
                                                        pad_index=self.tokenizer.eos_token_id, 
                                                        pad_first=False)
                    generate_ids, label_idx = self.accelerator.gather_for_metrics((generate_ids, label_idx))

                    generate_seq = self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True)
                    generate_seq = [pred.strip(' \n').lower() for pred in generate_seq]

                    label_idx = label_idx.detach().cpu().numpy()
                    target_output = [idx2label[l] for l in label_idx]
                    target_output = [target.strip(' \n').lower() for target in target_output]

                    acc_list = [1 if pred==target else 0 for pred, target in zip(generate_seq,target_output)]
                    acc_list_all.extend(acc_list)
            
            acc = np.round(np.mean(acc_list_all)*100,3)
            model.train()

            return  acc
        
        else:
            raise NotImplementedError()
    # ===========================================================================================