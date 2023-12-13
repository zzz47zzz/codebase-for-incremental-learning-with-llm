import numpy as np
import logging
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import ConcatDataset, DataLoader
from datasets import Dataset

from utils.metric import ResultSummary
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.buffer import get_buffer
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_sent_level_acc_with_generation, evaluate_sent_level_acc_with_classifier
from models.Base import BaseLearner

logger = logging.getLogger()

def get_ProgPrompt_params(parser):
    '''
        The parameters of model ProgPrompt
    '''

    parser.add_argument("--ProgPrompt_num_tokens_per_task", type=float, default=5, help="The number of soft prompt token to be learned for each task")
    parser.add_argument("--ProgPrompt_use_mlp", type=bool, default=True, help="Use a residual two-layer MLP to encode soft prompt")

class ProgPrompt(BaseLearner):
    '''
        ProgPrompt: Progressive Prompt learns soft prompt for each task progressively.
        This implementation is according to this (repository)[https://github.com/arazd/ProgressivePrompts]

        - (Progressive Prompts: Continual Learning for Language Models)[https://openreview.net/forum?id=UJTgQBc91_]
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['None','Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'ProgPrompt')
        assert params.il_mode in ['TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'ProgPrompt')
        assert params.classification_type == 'sentence-level', 'NotImplemented for classification_type %s'%(params.classification_type)
        assert (params.backbone_type == 'discriminative' and params.backbone_extract_token == 'cls_token') or \
                (params.backbone_type == 'generative' and params.backbone_extract_token == 'last_token')

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        backbone_model, self.tokenizer = get_backbone(self.params)
        self.model = ProgPromptModel(backbone_model=backbone_model, 
                                     tokenizer=self.tokenizer, 
                                     num_task=self.CL_dataset.continual_config['NUM_TASK'], 
                                     params=self.params)

    def build_classifier(self):
        feature_dim = self.model.module.backbone_model.config.hidden_size if hasattr(self.model,'module') else self.model.backbone_model.config.hidden_size
        out_dim_list = self.CL_dataset.continual_config['CUR_NUM_CLASS']
        self.classifier_list = get_classifier(self.params, feature_dim, out_dim_list)
        if self.classifier_list is not None:
            self.ce_loss = nn.CrossEntropyLoss()

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
        if self.params.is_replay:
            self.buffer = get_buffer(self.params,self.CL_dataset.continual_config,self.accelerator)
    
    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, self.classifier_list)
        self.wrap_model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
        self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)

        # Freeze all models weight except for the word embeddings
        if self.params.backbone_type == 'discriminative' and 'bert' in self.params.backbone:
            embed_name = 'word_embeddings'
        elif self.params.backbone_type == 'generative':
            embed_name = 'embed_in'
        else:
            raise NotImplementedError()
        
        for name, param in self.model.backbone_model.named_parameters():
            if param.requires_grad == True and embed_name not in name:
                param.requires_grad = False
            if embed_name in name:
                param.requires_grad = True

        # Freeze all other mlps except the mlp of current task
        if self.params.ProgPrompt_use_mlp:
            num_task = self.CL_dataset.continual_config['NUM_TASK']
            for t_id in range(num_task):
                if t_id == task_id:
                    for name, param in self.model.mlp_list[t_id].named_parameters():
                        param.requires_grad = True
                else:
                    for name, param in self.model.mlp_list[t_id].named_parameters():
                        param.requires_grad = False

    def end_task(self, task_id):
        super().end_task(task_id)

        if self.params.is_replay:
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

        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            wrap_model = self.wrap_model.module
        else:
            wrap_model = self.wrap_model
        wrap_model.train()

    def observe_batch(self, task_id, epoch_id, lm_input):
        '''
            Observe a batch of data
        '''
        # Update step
        self.step += 1
        self.global_step += 1

        # Sample from buffer and combine old data with new data
        if task_id>0 and self.params.is_replay and self.params.Replay_batch_level:
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

            total_loss = model.forward(lm_input, task_id)

        elif self.params.classifier in ['Linear','CosineLinear']:
            
            extracted_feature = model.forward(lm_input, task_id)

            total_loss = torch.tensor(0.).to(extracted_feature.device)

            for t_id in range(task_id+1):

                class_idx_range_bg = (self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][t_id])
                class_idx_range_ed = (self.CL_dataset.continual_config['ACCUM_NUM_CLASS'][t_id])
                task_mask = torch.logical_and(
                    lm_input['label_idx_cil']>=class_idx_range_bg,
                    lm_input['label_idx_cil']<class_idx_range_ed
                )
                if task_mask.sum().item()==0:
                    continue

                logits = classifier_list[t_id](extracted_feature[task_mask])

                if self.params.classification_type == 'sentence-level':
                    label_idx = lm_input['label_idx_til'][task_mask]
                else:
                    raise NotImplementedError()
        
                total_loss += self.ce_loss(logits,label_idx)

        else:
            raise NotImplementedError()

        # Backward
        model.train()
        self.optimizer.zero_grad()        
        self.accelerator.backward(total_loss)

        # Set the gradient of softprompt from other tasks to zero
        origin_vocab_size = model.origin_vocab_size
        pos1 = origin_vocab_size + task_id*model.num_tokens_per_task
        pos2 = origin_vocab_size + (task_id+1)*model.num_tokens_per_task

        if self.params.backbone_type == 'discriminative' and 'bert' in self.params.backbone:
            model.backbone_model.base_model.embeddings.word_embeddings.weight.grad[:pos1] = 0
            model.backbone_model.base_model.embeddings.word_embeddings.weight.grad[pos2:] = 0
        elif self.params.backbone_type == 'generative':
            model.backbone_model.base_model.embed_in.weight.grad[:pos1] = 0
            model.backbone_model.base_model.embed_in.weight.grad[pos2:] = 0
        else:
            raise NotImplementedError()
        
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

        assert il_mode == 'TIL'
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

        if self.params.classifier == 'None':

            if self.params.backbone_type == 'generative':
                model.eval()
                acc_list_all = []
                idx2label = self.CL_dataset.continual_config['idx2label']
                
                with torch.no_grad():
                    for lm_input in data_loader[eval_task_id]: 

                        label_idx = lm_input['label_idx_cil']
                        generate_ids = model.generate(params=self.params, 
                                                    lm_input=lm_input, 
                                                    task_id=eval_task_id)

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
            else:
                raise NotImplementedError()

            return  acc
        
        elif self.params.classifier in ['Linear','CosineLinear']:

            if self.params.backbone_type == 'discriminative' and 'bert' in self.params.backbone:

                model.eval()
                acc_list_all = []
                        
                with torch.no_grad():
                    for lm_input in data_loader[eval_task_id]: 
                
                        extracted_features = model.forward(lm_input, eval_task_id)

                        logits = classifier_list[eval_task_id](extracted_features)
                        preds = logits.argmax(dim=-1)
                        label_idx = lm_input['label_idx_til']

                        # Gather from different processes in DDP
                        preds, label_idx = self.accelerator.gather_for_metrics((preds, label_idx))
                        label_idx = label_idx.detach().cpu().numpy()
                        acc_list = [1 if pred==target else 0 for pred, target in zip(preds,label_idx)]
                        acc_list_all.extend(acc_list)
                        acc = np.round(np.mean(acc_list_all)*100,3)
                
                model.train()

            return  acc

        else:
            raise NotImplementedError()
    # ===========================================================================================

class ProgPromptModel(nn.Module):
    def __init__(self, backbone_model, tokenizer, num_task, params) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.feature_dim = self.backbone_model.config.hidden_size 
        self.tokenizer = tokenizer
        self.params = params
        self.num_task = num_task
        self.num_tokens_per_task = self.params.ProgPrompt_num_tokens_per_task
        self.use_mlp = self.params.ProgPrompt_use_mlp
        if self.use_mlp:
            self.mlp_list = nn.ModuleList([ResMLP(self.feature_dim,800) for _ in range(self.num_task)])

        self.init_softprompt()
        
    def init_softprompt(self):

        if self.params.backbone_type == 'discriminative' and 'bert' in self.params.backbone:

            # (progressive) prompt tuning set-up
            # Hello world :) [pad] [pad] ... [pad] [PRE0_1] [PRE0_2] [PRE0_3] [PRE1_1] [PRE1_2] [PRE1_3]
            # 1  2  3  4  5  6      7        400    0        401      402       0        403      404

            self.origin_vocab_size = len(self.tokenizer)

            tokens_list = []
            for t_id in range(self.num_task):
                tokens_list.extend(['PRE%d_%d'%(t_id,_idx) for _idx in range(1,self.num_tokens_per_task+1)])

            special_tokens_dict = {'additional_special_tokens': tokens_list}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.backbone_model.base_model.resize_token_embeddings(len(self.tokenizer))
            embedding_matrix = self.backbone_model.base_model.embeddings.word_embeddings.weight

            embedding_matrix.requires_grad = False
            for t_id in range(self.num_task):
                for _idx in range(self.num_tokens_per_task):
                    if _idx == 0:
                        j = self.tokenizer.cls_token_id
                    else:
                        j = np.random.randint(0,self.origin_vocab_size)
                w = deepcopy(embedding_matrix[j].detach().cpu().numpy())
                embedding_matrix[self.origin_vocab_size+t_id*self.num_tokens_per_task+_idx] = torch.from_numpy(w)
                
            embedding_matrix.requires_grad = True

        elif self.params.backbone_type == 'generative':

            self.backbone_model_prepare_inputs_for_generation = self.backbone_model.prepare_inputs_for_generation

            # (progressive) prompt tuning set-up
            # [PRE0_1] [PRE0_2] [PRE0_3] [PRE1_1] [PRE1_2] [PRE1_3] <eos> <eos> ... <eos> Hello world :)   <eos>
            #    0        1        2        0        3       4       5     6         396   397   398  399   400

            self.origin_vocab_size = len(self.tokenizer)

            tokens_list = []
            for t_id in range(self.num_task):
                tokens_list.extend(['<PRE%d_%d>'%(t_id,_idx) for _idx in range(1,self.num_tokens_per_task+1)])

            special_tokens_dict = {'additional_special_tokens': tokens_list}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.backbone_model.base_model.resize_token_embeddings(len(self.tokenizer))
            embedding_matrix = self.backbone_model.base_model.embed_in.weight

            embedding_matrix.requires_grad = False
            for t_id in range(self.num_task):
                for _idx in range(self.num_tokens_per_task):
                    if _idx == 0:
                        j = self.tokenizer.bos_token_id
                    else:
                        j = np.random.randint(0,self.origin_vocab_size)
                w = deepcopy(embedding_matrix[j].detach().cpu().numpy())
                embedding_matrix[self.origin_vocab_size+t_id*self.num_tokens_per_task+_idx] = torch.from_numpy(w)
                
            embedding_matrix.requires_grad = True

        else:
            raise NotImplementedError()
        
    def forward(self, lm_input, task_id):

        if self.params.backbone_type == 'discriminative' and 'bert' in self.params.backbone:

            assert all(lm_input['input_ids'][:,0]==self.tokenizer.cls_token_id)
            bs, seq_len = lm_input['input_ids'].shape

            original_input_ids = lm_input['input_ids'][:,1:] # remove the original CLS token
            soft_prompt_input_ids = torch.tensor([list(range(self.origin_vocab_size,
                                                            self.origin_vocab_size+(task_id+1)*self.num_tokens_per_task))])
            soft_prompted_input_ids = torch.cat((original_input_ids,
                                                 soft_prompt_input_ids.repeat((bs,1)).to(original_input_ids.device)),dim=-1)
            
            original_attention_mask = lm_input['attention_mask'][:,1:] # remove the original CLS token
            soft_prompted_attention_mask = torch.cat((original_attention_mask,
                                torch.ones((bs,(task_id+1)*self.num_tokens_per_task)).to(original_attention_mask.device)),dim=-1)

            seq_len -= 1 
            position_ids = list(range(1,seq_len+1))
            for t_id in range(task_id+1):
                position_ids.append(0) # the CLS token for the t_id task
                position_ids.extend(list(range(seq_len+1+t_id*(self.num_tokens_per_task-1),
                                               seq_len+1+(t_id+1)*(self.num_tokens_per_task-1)))) 
            position_ids = torch.tensor(position_ids).repeat((bs,1)).to(lm_input['input_ids'].device)

            if not self.use_mlp:
                all_hidden_states = self.backbone_model.forward(**{
                                'input_ids':soft_prompted_input_ids,
                                'attention_mask':soft_prompted_attention_mask,
                                'position_ids':position_ids},
                                output_hidden_states=True).hidden_states
            else:
                bert_embeds = self.backbone_model.base_model.embeddings(**{
                    'input_ids':soft_prompted_input_ids,
                    'position_ids':position_ids
                })
                pos1 = seq_len+1 + self.num_tokens_per_task*t_id 
                pos2 = seq_len+1 + self.num_tokens_per_task*(t_id+1)
                bert_embeds[:, pos1:pos2, :] = self.mlp_list[task_id](bert_embeds[:, pos1:pos2, :].clone().to(self.backbone_model.device))

                extended_attention_mask: torch.Tensor = self.backbone_model.base_model.get_extended_attention_mask(
                    soft_prompted_attention_mask,
                    soft_prompted_input_ids.shape,
                    device=self.backbone_model.device
                )
                all_hidden_states = self.backbone_model.base_model.encoder(bert_embeds, 
                                                                           attention_mask=extended_attention_mask, 
                                                                           output_hidden_states=True).hidden_states 

            # cls token feature
            extracted_feature = all_hidden_states[-1][:,seq_len+1+task_id*self.num_tokens_per_task,:].contiguous() 

            return extracted_feature
        
        elif self.params.backbone_type == 'generative':
            
            bs, seq_len = lm_input['input_ids_with_ans'].shape

            original_input_ids = lm_input['input_ids_with_ans']
            soft_prompt_input_ids = torch.tensor([list(range(self.origin_vocab_size,
                                                            self.origin_vocab_size+(task_id+1)*self.num_tokens_per_task))])
            soft_prompted_input_ids = torch.cat((soft_prompt_input_ids.repeat((bs,1)).to(original_input_ids.device),
                                                 original_input_ids),dim=-1) # soft prompt first
            
            original_attention_mask = lm_input['attention_mask_with_ans']
            soft_prompted_attention_mask = torch.cat((torch.ones((bs,(task_id+1)*self.num_tokens_per_task)).to(original_attention_mask.device),
                                                      original_attention_mask),dim=-1)

            original_labels = lm_input['labels_with_ans']
            soft_prompted_labels = torch.cat((-100*torch.ones((bs,(task_id+1)*self.num_tokens_per_task),dtype=original_labels.dtype).to(original_labels.device),
                                                      original_labels),dim=-1)

            # NOTE: position_ids is ignored in gpt models
            # position_ids = []
            # for t_id in range(task_id+1):
            #     position_ids.append(0) # the CLS token for the t_id task
            #     position_ids.extend(list(range(1+t_id*(self.num_tokens_per_task-1),
            #                                    1+(t_id+1)*(self.num_tokens_per_task-1)))) 
            # position_ids.extend(list(range(1+(task_id+1)*(self.num_tokens_per_task-1),
            #                                1+(task_id+1)*(self.num_tokens_per_task-1)+seq_len)))
            # position_ids = torch.tensor(position_ids).repeat((bs,1)).to(lm_input['input_ids'].device)

            if not self.use_mlp:
                causal_lm_loss = self.backbone_model.forward(**{
                                'input_ids':soft_prompted_input_ids,
                                'attention_mask':soft_prompted_attention_mask,
                                'labels':soft_prompted_labels}).loss
            else:
                gpt_embeds = self.backbone_model.base_model.embed_in(soft_prompted_input_ids)
                pos1 = seq_len+1 + self.num_tokens_per_task*task_id 
                pos2 = seq_len+1 + self.num_tokens_per_task*(task_id+1)
                gpt_embeds[:, pos1:pos2, :] = self.mlp_list[task_id](gpt_embeds[:, pos1:pos2, :].clone().to(self.backbone_model.device))

                causal_lm_loss = self.backbone_model.forward(**{
                                'inputs_embeds':gpt_embeds,
                                'attention_mask':soft_prompted_attention_mask,
                                'labels':soft_prompted_labels}).loss 
                
            return causal_lm_loss

        else:
            raise NotImplementedError()

    def generate(self, params, lm_input, task_id):

        assert params.backbone_type == 'generative'   

        self.backbone_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

        try:
            self.generate_task_id = task_id
            input_len = lm_input['input_ids'].shape[1]
            generate_ids_all = self.backbone_model.generate(
                **{'input_ids':lm_input['input_ids'], 
                'attention_mask':lm_input['attention_mask']}, 
                max_new_tokens=params.backbone_max_new_token, 
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except:
            self.backbone_model.prepare_inputs_for_generation = self.backbone_model_prepare_inputs_for_generation
            self.generate_task_id = None
            raise
        else:
            self.backbone_model.prepare_inputs_for_generation = self.backbone_model_prepare_inputs_for_generation
            self.generate_task_id = None

            generate_ids = generate_ids_all[:,input_len:].contiguous()

            return generate_ids

    def prepare_inputs_for_generation(self, *args, **kwargs):

        model_kwargs = self.backbone_model_prepare_inputs_for_generation(*args, **kwargs)

        assert self.generate_task_id is not None
        task_id = self.generate_task_id

        assert model_kwargs.get('input_ids', None) is not None
        bs, seq_len = model_kwargs['input_ids'].shape

        if model_kwargs["past_key_values"] is not None:

            assert model_kwargs.get('attention_mask', None) is not None
            original_attention_mask = model_kwargs['attention_mask']
            soft_prompted_attention_mask = torch.cat((torch.ones((bs,(task_id+1)*self.num_tokens_per_task)).to(original_attention_mask.device),
                                                        original_attention_mask),dim=-1)
            model_kwargs['attention_mask'] = soft_prompted_attention_mask

            return model_kwargs

        if model_kwargs.get("position_ids", None) is not None:
            # Position ids are not supported for parameter efficient tuning. Ignoring position ids."
            model_kwargs["position_ids"] = None

        original_input_ids = model_kwargs['input_ids']
        soft_prompt_input_ids = torch.tensor([list(range(self.origin_vocab_size,
                                                        self.origin_vocab_size+(task_id+1)*self.num_tokens_per_task))])
        soft_prompted_input_ids = torch.cat((soft_prompt_input_ids.repeat((bs,1)).to(original_input_ids.device),
                                                original_input_ids),dim=-1) # soft prompt first
        
        assert model_kwargs.get('attention_mask', None) is not None
        original_attention_mask = model_kwargs['attention_mask']
        soft_prompted_attention_mask = torch.cat((torch.ones((bs,(task_id+1)*self.num_tokens_per_task)).to(original_attention_mask.device),
                                                    original_attention_mask),dim=-1)
        
        # We must use inputs_embeds instead of input_ids
        if self.use_mlp: 
            soft_prompted_inputs_embeds = self.backbone_model.base_model.embed_in(soft_prompted_input_ids)
            model_kwargs['input_ids'] = None
            model_kwargs['inputs_embeds'] = soft_prompted_inputs_embeds
            model_kwargs['attention_mask'] = soft_prompted_attention_mask
        else:
            model_kwargs['input_ids'] = soft_prompted_input_ids
            model_kwargs['inputs_embeds'] = None
            model_kwargs['attention_mask'] = soft_prompted_attention_mask
            
        return model_kwargs
        

class ResMLP(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, feature_dim),
        )

    def forward(self, inputs):
        return self.mlp(inputs) + inputs