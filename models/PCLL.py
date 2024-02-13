import numpy as np
import logging
import torch
import torch.nn as nn
from typing import List
from copy import deepcopy
from datasets import Dataset
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
from transformers import Conv1D
from transformers.activations import gelu

from utils.metric import ResultSummary
from utils.backbone import get_backbone
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader, preprocess_function_train_and_predict_generative_PCLL
from utils.wrapmodel import WrapModel
from utils.prompt import get_prompt_PCLL
from utils.evaluation import evaluate_sent_level_acc_with_generation
from models.Base import BaseLearner

logger = logging.getLogger()

def get_PCLL_params(parser):
    '''
        The parameters of model PCLL
    '''

    parser.add_argument("--PCLL_KD_lambda", type=float, default=0.50, help="The weight of the generation target in PCLL")
    parser.add_argument("--PCLL_weight_vae_loss", type=float, default=0, help="The weight of vae_loss (the model can be normally trained only when it is zero!?)")
    parser.add_argument("--PCLL_weight_gen_loss", type=float, default=0.50, help="The weight of generation_loss")
    parser.add_argument("--PCLL_alpha_z", type=float, default=0.1, help="Multiply alpha when adding the latent z embedding onto the original embeddings.")
    parser.add_argument("--PCLL_KD_gamma", type=float, default=0.20, help="The ratio of psesudo old samples w.r.t the training data of new task.")
    parser.add_argument("--PCLL_KD_topk", type=int, default=20, help="The top-k sampling for generating psesudo old samples.")
    parser.add_argument("--PCLL_KD_temperature", type=float, default=1.0, help="The temperature for knowledge distillation.")

class PCLL(BaseLearner):
    '''
        PCLL: PCLL is build upon LAMOL. Furthermore, PCLL utilizes the targets of varational autoencoders 
        and word-level knowledge distillation to train generative models. 
        The implementation is based on the official released code in the [repository](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/pcll).

        - [Prompt Conditioned VAE: Enhancing Generative Replay for Lifelong Learning in Task-Oriented Dialogue](https://aclanthology.org/2022.emnlp-main.766/)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)

        assert params.classifier in ['None'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'PCLL')
        assert params.il_mode in ['IIL','CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'PCLL')
        assert params.classification_type == 'sentence-level', 'NotImplemented for classification type %s'%(params.classification_type)
        assert params.backbone_type == 'generative', 'NotImplemented for backbone type %s'%(params.backbone_type)
        assert not params.is_replay, 'NotImplemented for is_replay = %s'%(params.is_replay)

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        backbone_model, self.tokenizer = get_backbone(self.params)
        self.model = PCLLModel(backbone_model, self.params)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def build_classifier(self):
        self.classifier_list = None

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

    def end_task(self, task_id):
        super().end_task(task_id)

    # ==============================================================================================

    # ================================= Epoch-Level Functions =======================================
    def train_epochs(self, task_id):
        '''
            Training the model with serveral epochs
        '''

        if task_id>0:
            train_dataset = self.train_loader_list[task_id].dataset
            pseudo_buf_dataset_list = self.generate_pseudo_buffer_samples(task_id=task_id,
                                                                     num_samples=int(len(train_dataset)*self.params.PCLL_KD_gamma))
            pseudo_train_loader = DataLoader(
                ConcatDataset(pseudo_buf_dataset_list),
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False
            )
            pseudo_train_loader = self.accelerator.prepare(pseudo_train_loader)
            pseudo_train_loader_iter = iter(pseudo_train_loader)
            num_batch_pseudo = len(pseudo_train_loader)
            pseudo_idx = 0
        
        cur_train_loader = self.train_loader_list[task_id]

        total_epochs = self.params.training_epochs

        self.beta_list = self.wrap_model.model.frange_cycle_linear(int(total_epochs*len(cur_train_loader)))

        for epoch_id in range(total_epochs):

            if self.accelerator.is_main_process:
                logger.info("------------------------ epoch %d ------------------------" %(epoch_id+1))

            self.begin_epoch(task_id, epoch_id)

            for lm_input in cur_train_loader:
                if task_id==0:
                    self.observe_batch(task_id, epoch_id, lm_input, buf_lm_input=None) 
                else:
                    if pseudo_idx==num_batch_pseudo:
                        pseudo_train_loader_iter = iter(pseudo_train_loader)
                        pseudo_idx = 0

                    buf_lm_input = next(pseudo_train_loader_iter) 
                    pseudo_idx += 1

                    self.observe_batch(task_id, epoch_id, lm_input, buf_lm_input=buf_lm_input) 

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
        self.wrap_model.train()

    def observe_batch(self, task_id, epoch_id, lm_input, buf_lm_input=None):
        '''
            Observe a batch of data
        '''
        # Update step
        self.step += 1
        self.global_step += 1

        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            model = self.wrap_model.module.model
        else:
            model = self.wrap_model.model
        if self.wrap_teacher_model is not None:
            if hasattr(self.wrap_teacher_model,'module'):
                teacher_model = self.wrap_teacher_model.module.model
            else:
                teacher_model = self.wrap_teacher_model.model

        # Compute loss
        # Training with Causal Language Modeling Loss
        
        beta = self.beta_list[self.step-1]

        vae_loss = model.compute_vae_loss(lm_input=lm_input,beta=beta)
        lm_loss = model.compute_lm_loss(lm_input=lm_input)

        total_loss = self.params.PCLL_weight_vae_loss*vae_loss + lm_loss

        if task_id > 0:

            distil_vae_loss = model.compute_vae_loss(lm_input=buf_lm_input,teacher_backbone_model=teacher_model.backbone_model,beta=beta)
            distil_lm_loss = model.compute_lm_loss(lm_input=buf_lm_input,teacher_backbone_model=teacher_model.backbone_model)

            total_loss += self.params.PCLL_weight_vae_loss*distil_vae_loss + distil_lm_loss

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
        if hasattr(self.wrap_model,'module'):
            backbone_model = self.wrap_model.module.model.backbone_model
        else:
            backbone_model = self.wrap_model.model.backbone_model

        if self.classifier_list is None:
            
            acc = evaluate_sent_level_acc_with_generation(
                model=backbone_model,
                eval_data_loader=data_loader[eval_task_id],
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config['idx2label']
            )
            # NOTE: When not using classifier, the SEQ model does not need (benefit from) task identity 

            return  acc

        else:
            raise NotImplementedError()
    # ===========================================================================================

    # ======================== Other Model-Specific Functions ===================================
    def generate_pseudo_buffer_samples(self, task_id: int, num_samples: int) -> List[Dataset]:
        '''
            Generate pseudo old samples with generative models

            Args:
                - task_id: the current task id
                - num_samples: the number of samples to be generated

            Return:
                pseudo_dataset_list
        '''
        input_column = 'input'
        target_column = 'target'

        gen_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        num_task = self.CL_dataset.continual_config['NUM_TASK']

        pseudo_dataset_list = []
        
        generate_batch_size = 8

        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            model = self.wrap_model.module.model
        else:
            model = self.wrap_model.model

        with torch.no_grad():
        
            for t_id in range(task_id):

                if self.params.il_mode == 'IIL':
                    pesudo_samples_dict = {
                        'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': [],
                        'instance_id': [], 'concept_id': [], 'relation_id': [], 
                    }
                else:
                        pesudo_samples_dict = {
                        'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': []
                    }

                cnt_num_samples = num_samples//task_id

                prompt_for_generation = get_prompt_PCLL(text='',
                                                        label='',
                                                        gen_token=gen_token,
                                                        eos_token='', # NOTE: we do not use eos_token in the prompt
                                                        task_id=None if self.params.il_mode=='CIL' else task_id)[0]

                while cnt_num_samples > 0:
                        
                    generate_num = generate_batch_size if cnt_num_samples>=generate_batch_size else cnt_num_samples

                    lm_input = self.tokenizer([prompt_for_generation for _ in range(generate_num)],
                                                padding='longest',
                                                return_tensors='pt')
                    lm_input = {k:v.to(model.backbone_model.device) for k,v in lm_input.items()}
                    max_input_len = np.max([len(lm_input['input_ids'][i]) for i in range(generate_num)])

                    # input_embed = model.backbone_model.base_model.embed_in(lm_input['input_ids'])
                    # prior_out=model.backbone_model(**{'input_ids':lm_input['input_ids'], 
                    #                 'attention_mask':lm_input['attention_mask']},
                    #                 output_hidden_states=True).hidden_states[-1]
                    # prior_emb, _ = model.avg_attn(prior_out)
                    # prior_mean, prior_logvar = model.prior_mean(prior_emb), model.prior_logvar(prior_emb)
                    # z = model.reparameterize(prior_mean, prior_logvar)
                    # latent_proj = model.latent_mlp(z) * self.params.PCLL_alpha_z
                    
                    # NOTE: We cannot pass the key word augment 'inputs_embeds' for gpt models 
                    generate_ids_all = model.backbone_model.generate(**{'input_ids':lm_input['input_ids'], 
                                                'attention_mask':lm_input['attention_mask']}, 
                                            max_new_tokens=self.params.max_seq_length-max_input_len, 
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            do_sample=True,
                                            top_k=self.params.PCLL_KD_topk,
                                            top_p=0.95) 
                    generate_ids = generate_ids_all[:,max_input_len:].contiguous()
                    generated_samples = self.tokenizer.batch_decode(generate_ids)

                    for _one_sample in generated_samples:
                        if _one_sample.count(' "? Answer: ')!=1:
                            continue
                        _question, _answer = _one_sample.split(' "? Answer: ')
                        _answer = _answer.replace(self.tokenizer.eos_token,'')
                        pesudo_samples_dict['input'].append(_question)
                        pesudo_samples_dict['target'].append(_answer)
                        pesudo_samples_dict['label_idx_cil'].append(-1)
                        pesudo_samples_dict['label_idx_til'].append(-1)
                        if self.params.il_mode == 'IIL':
                            pesudo_samples_dict['instance_id'].append(-1)
                            pesudo_samples_dict['concept_id'].append(-1)
                            pesudo_samples_dict['relation_id'].append(-1)
                        
                    cnt_num_samples -= generate_num
            
                if len(pesudo_samples_dict['input'])==0:
                    logger.info('No pseudo samples are generated in the correct format for task %d!'%(t_id+1))
                    continue
                pseudo_dataset = Dataset.from_dict(pesudo_samples_dict)
                pseudo_dataset = pseudo_dataset.map(preprocess_function_train_and_predict_generative_PCLL, 
                                                    batched=True, 
                                                    desc='Generate pseudo samples for task %d'%(t_id+1), 
                                                    batch_size=1000,
                                                    fn_kwargs={
                                                        'params':self.params,
                                                        'tokenizer':self.tokenizer,
                                                        'num_task':num_task,
                                                        'task_id':task_id,
                                                        'input_column':input_column,
                                                        'target_column':target_column,
                                                        'eos_token':eos_token,
                                                        'gen_token':gen_token,
                                                    })
                if self.params.il_mode == 'IIL':
                    pseudo_dataset.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask',
                        'label_idx_cil', 'label_idx_til',
                        'input_ids_prompt', 'attention_mask_prompt',
                        'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                        'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                        'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans',
                        'target', 'instance_id', 'concept_id', 'relation_id'
                    ])
                else:
                    pseudo_dataset.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask',
                        'label_idx_cil', 'label_idx_til',
                        'input_ids_prompt', 'attention_mask_prompt',
                        'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                        'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                        'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans'
                    ])

                pseudo_dataset_list.append(pseudo_dataset)

        return pseudo_dataset_list

    # ===========================================================================================

class PCLLModel(nn.Module):
    '''
        We use a wrapper for a backbone and a classifier 
        because Deepspeed only support one nn.Mudule() when calling accelerate.prepare().
    '''
    def __init__(self, model, params) -> None:
        super().__init__()
        self.backbone_model = model
        self.params = params

        self.feature_dim = self.backbone_model.module.config.hidden_size if hasattr(self.backbone_model,'module') else self.backbone_model.config.hidden_size

        self.prior_mean = Conv1D(self.feature_dim, self.feature_dim)
        self.prior_logvar = Conv1D(self.feature_dim, self.feature_dim)
        self.post_mean = Conv1D(self.feature_dim, self.feature_dim)
        self.post_logvar = Conv1D(self.feature_dim, self.feature_dim)
        self.avg_attn = AverageSelfAttention(self.feature_dim)
        self.latent_mlp = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    
    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)
    
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=0.9):
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 

    def vae_kl_loss(self, mean1, logvar1, mean2, logvar2): # [batch_size(64), hidden_size(768)]
        # print(mean1.size(),logvar1.size(),mean2.size(),logvar2.size(),flush=True)
        exponential = logvar1 - logvar2 - \
            torch.pow(mean1 - mean2, 2) / logvar2.exp() - \
            torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()
     
    def compute_lm_loss(self, lm_input, teacher_backbone_model=None):

        if teacher_backbone_model is None:
            qa_loss = self.backbone_model(**{'input_ids':lm_input['input_ids_prompt_input_anstoken_ans'], 
                                        'attention_mask':lm_input['attention_mask_prompt_input_anstoken_ans'],
                                        'labels':lm_input['labels_qa_prompt_input_anstoken_ans']}).loss

        else:
            with torch.no_grad():
                teacher_qa_logits = teacher_backbone_model(**{'input_ids':lm_input['input_ids_prompt_input_anstoken_ans'], 
                                    'attention_mask':lm_input['attention_mask_prompt_input_anstoken_ans']}).logits
                teacher_qa_logits_flatten = teacher_qa_logits[lm_input['labels_qa_prompt_input_anstoken_ans']!=-100]

            student_qa_output = self.backbone_model(**{'input_ids':lm_input['input_ids_prompt_input_anstoken_ans'], 
                                        'attention_mask':lm_input['attention_mask_prompt_input_anstoken_ans'],
                                        'labels':lm_input['labels_qa_prompt_input_anstoken_ans']})
            student_qa_logits = student_qa_output.logits
            student_qa_logits_flatten = student_qa_logits[lm_input['labels_qa_prompt_input_anstoken_ans']!=-100]
            qa_loss = student_qa_output.loss
            temp = float(self.params.PCLL_KD_temperature)
            qa_loss += self.kl_loss(F.log_softmax(student_qa_logits_flatten/temp,dim=-1),
                                        F.softmax(teacher_qa_logits_flatten/temp,dim=-1))*(temp**2)

        if teacher_backbone_model is None:
            generation_loss = self.backbone_model(**{'input_ids':lm_input['input_ids_prompt_input_anstoken_ans'], 
                                    'attention_mask':lm_input['attention_mask_prompt_input_anstoken_ans'],
                                    'labels':lm_input['labels_gen_prompt_input_anstoken_ans']}).loss
        else:
            with torch.no_grad():
                teacher_gen_logits = teacher_backbone_model(**{'input_ids':lm_input['input_ids_prompt_input_anstoken_ans'], 
                                    'attention_mask':lm_input['attention_mask_prompt_input_anstoken_ans']}).logits
                teacher_gen_logits_flatten = teacher_gen_logits[lm_input['labels_gen_prompt_input_anstoken_ans']!=-100]

            student_gen_output = self.backbone_model(**{'input_ids':lm_input['input_ids_prompt_input_anstoken_ans'], 
                                        'attention_mask':lm_input['attention_mask_prompt_input_anstoken_ans'],
                                        'labels':lm_input['labels_gen_prompt_input_anstoken_ans']})
            student_gen_logits = student_gen_output.logits
            student_gen_logits_flatten = student_gen_logits[lm_input['labels_gen_prompt_input_anstoken_ans']!=-100]
            generation_loss = student_gen_output.loss
            temp = float(self.params.PCLL_KD_temperature)
            generation_loss += self.kl_loss(F.log_softmax(student_gen_logits_flatten/temp,dim=-1),
                                        F.softmax(teacher_gen_logits_flatten/temp,dim=-1))*(temp**2)

        return qa_loss + self.params.PCLL_weight_gen_loss*generation_loss

    def compute_vae_loss(self, lm_input, teacher_backbone_model=None, beta=0):
        # * Get posterior: latent representation p(z|x,p)

        post_out = self.backbone_model(**{'input_ids':lm_input['input_ids_prompt_input'], 
                                 'attention_mask':lm_input['attention_mask_prompt_input']},
                                 output_hidden_states=True).hidden_states[-1]
        post_emb, _ = self.avg_attn(post_out, attention_mask=lm_input['attention_mask_prompt_input'])
        # print('post_emb',post_emb.shape,flush=True) # (64, 768)
        posterior_mean, posterior_logvar = self.post_mean(post_emb), self.post_logvar(post_emb)

        # * Get prior: 
        prior_out = self.backbone_model(**{'input_ids':lm_input['input_ids_prompt'], 
                                 'attention_mask':lm_input['attention_mask_prompt']},
                                 output_hidden_states=True).hidden_states[-1]
        prior_emb, _ = self.avg_attn(prior_out, attention_mask=lm_input['attention_mask_prompt'])
        prior_mean, prior_logvar = self.prior_mean(prior_emb), self.prior_logvar(prior_emb)

        latent_z = self.reparameterize(posterior_mean, posterior_logvar)
        latent_proj = self.params.PCLL_alpha_z * self.latent_mlp(latent_z)
        assert not torch.isnan(latent_z).any(), 'Training gets NAN z!'

        kl_loss = self.vae_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)
        input_embed = self.backbone_model.base_model.embed_in(lm_input['input_ids_prompt_input'])
        
        if teacher_backbone_model is None:
            reconstruct_loss = self.backbone_model(**{'inputs_embeds':input_embed+latent_proj, 
                                    'attention_mask':lm_input['attention_mask_prompt_input'],
                                    'labels':lm_input['labels_gen_prompt_input']}).loss
        else:
            with torch.no_grad():
                teacher_reconstruct_logits = teacher_backbone_model(**{'inputs_embeds':input_embed+latent_proj, 
                                    'attention_mask':lm_input['attention_mask_prompt_input']}).logits
                teacher_reconstruct_logits_flatten = teacher_reconstruct_logits[lm_input['labels_gen_prompt_input']!=-100]
                
            generation_output = self.backbone_model(**{'inputs_embeds':input_embed+latent_proj, 
                                    'attention_mask':lm_input['attention_mask_prompt_input'],
                                    'labels':lm_input['labels_gen_prompt_input']})
            
            reconstruct_loss = generation_output.loss
            reconstruct_logits = generation_output.logits
            reconstruct_logits_flatten = reconstruct_logits[lm_input['labels_gen_prompt_input']!=-100]

            temp = float(self.params.PCLL_KD_temperature)
            reconstruct_loss += self.kl_loss(F.log_softmax(reconstruct_logits_flatten/temp,dim=-1),
                                        F.softmax(teacher_reconstruct_logits_flatten/temp,dim=-1))*(temp**2)

        return reconstruct_loss + beta*0.01*kl_loss

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # print('weighted shape',weighted.shape, flush=True)
        if len(weighted.shape) == 2:
            representations = weighted.sum(0).unsqueeze(0)
        else:
            representations = weighted.sum(1).squeeze(1)
        # print('after weighted shape',weighted.shape, flush=True) 

        return representations, scores