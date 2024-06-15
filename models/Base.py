import os
import numpy as np
import logging
import torch
from abc import abstractmethod
from copy import deepcopy
from accelerate.utils import gather_object


from utils.evaluation import compute_backward_transfer, compute_average_acc, compute_average_inc_acc, compute_forgetting
from utils.probing import probing_on_all_task, save_all_features_labels

logger = logging.getLogger()

class BaseLearner(object):
    def __init__(self, params, CL_dataset, accelerator):
        # parameters
        self.params = params
        self.best_model_ckpt_name = None
        self.global_step=0
        self.CL_dataset = CL_dataset
        self.accelerator = accelerator

        # initialization
        self.model = None
        self.classifier = None
        self.optimizer = None
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = [], [], []
        self.build_metric()
        self.build_backbone()
        self.build_classifier()
        self.build_optimizer()
        self.build_dataloader()
        self.build_buffer()
        self.accelerate_prepare()
        
    # ================================= Prepare model, data and optimizer =======================================
    @abstractmethod
    def build_metric(self):
        pass

    @abstractmethod
    def build_backbone(self):
        pass

    @abstractmethod
    def build_classifier(self):
        pass

    @abstractmethod
    def build_optimizer(self):
        pass
        
    @abstractmethod
    def build_dataloader(self):
        pass

    @abstractmethod
    def build_buffer(self):
        pass

    @abstractmethod
    def accelerate_prepare(self):
        pass
    # ==============================================================================================

    # ================================= Task-Level Functions =======================================
    def incremental_training(self):

        num_task = self.CL_dataset.continual_config['NUM_TASK']

        # Learn Tasks Incrementally
        for task_id in range(num_task):
            if self.accelerator.is_main_process: 
                logger.info("============================================================================")   
                logger.info("Beggin training the task %d (total %d tasks)"%(task_id+1, num_task))     
                logger.info("============================================================================")
            self.begin_task(task_id)
            self.train_epochs(task_id)
            self.end_task(task_id)  

    def begin_task(self, task_id):
        
        self.best_score = -1
        self.step = 0

        # save features
        if task_id==0 and self.params.save_features_before_after_IL:
            save_all_features_labels(self.params, 
                                     'BeforeIL',
                                     self.train_loader_list, 
                                     self.model, 
                                     self.tokenizer, 
                                     self.accelerator)
        
    def end_task(self, task_id):

        # testing
        if self.accelerator.is_main_process:
            logger.info("Testing...")

        result_dict = self.evaluate_model(task_id=task_id) 
        il_mode = self.params.il_mode      
        for t_id, _acc in enumerate(result_dict['Test_Acc_List']):
            self.result_summary.update(task_id, t_id, _acc)
        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Result Summary Test After Task %d = \n%s'%(il_mode, 
                                                                        task_id,
                                                                        self.result_summary.print_format()))
        if il_mode == 'IIL' and self.params.method != 'ICL':
            for t_id, _acc in enumerate(result_dict['Train_Acc_List']):
                self.result_summary_train.update(task_id, t_id, _acc)
            if self.accelerator.is_main_process:
                logger.info('Mode = %s, Result Summary Train After Task %d = \n%s'%(il_mode, 
                                                                            task_id,
                                                                            self.result_summary_train.print_format()))

        # save ckpt
        if self.params.save_ckpt:
            torch.save(
                {
                    'model': deepcopy(self.model).cpu(),
                    'classifier_list': deepcopy(self.classifier_list).cpu(),
                }
                ,os.path.join(self.params.dump_path,'last_ckpt_task%d.pth'%(task_id)))
            
        # save features
        num_task = self.CL_dataset.continual_config['NUM_TASK']
        if task_id==num_task-1 and self.params.save_features_before_after_IL:
            save_all_features_labels(self.params, 
                                     'AfterIL',
                                     self.train_loader_list, 
                                     self.model, 
                                     self.tokenizer, 
                                     self.accelerator)

        # Save GPU memory
        torch.cuda.empty_cache()
    # ===========================================================================================


    # ================================= Epoch-Level Functions ====================================
    @abstractmethod
    def train_epochs(self, task_id):
        pass
    # ===========================================================================================


    # ================== Evaluation, Logging, Saving and Loading Functions ======================
    def finish_training(self):
        '''
            Finish training: print the result
        '''
        log_dict = {}
        il_mode = self.params.il_mode
        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Summary Test Acc = \n%s'%(il_mode,self.result_summary.print_format()))
        # Compute Forward and Backward Transfer according to Result Summary for the whole learning process
        bwt_acc = compute_backward_transfer(self.result_summary.get_value()) 
        fgt_acc = compute_forgetting(self.result_summary.get_value()) 
        aver_acc = compute_average_acc(self.result_summary.get_value()) 
        aver_inc_acc = compute_average_inc_acc(self.result_summary.get_value()) 
        log_dict['Test_Aver_ACC'] = aver_acc
        log_dict['Test_Bwt_ACC'] = bwt_acc
        log_dict['Test_Fgt_ACC'] = fgt_acc
        log_dict['Test_Aver_Inc_ACC'] = aver_inc_acc

        if il_mode == 'IIL' and self.params.method != 'ICL':
            if self.accelerator.is_main_process:
                logger.info('Mode = %s, Summary Train Acc = \n%s'%(il_mode,self.result_summary_train.print_format()))
            # Compute Forward and Backward Transfer according to Result Summary for the whole learning process
            bwt_acc = compute_backward_transfer(self.result_summary_train.get_value()) 
            fgt_acc = compute_forgetting(self.result_summary_train.get_value()) 
            aver_acc = compute_average_acc(self.result_summary_train.get_value()) 
            aver_inc_acc = compute_average_inc_acc(self.result_summary_train.get_value()) 
            log_dict['Train_Aver_ACC'] = aver_acc
            log_dict['Train_Bwt_ACC'] = bwt_acc
            log_dict['Train_Fgt_ACC'] = fgt_acc
            log_dict['Train_Aver_Inc_ACC'] = aver_inc_acc

        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Summary Result = \n%s'%(il_mode,log_dict))    
        self.accelerator.log(log_dict,step=self.global_step)
    
        # Delete checkpoints
        # if not self.params.save_ckpt:
        #     for file_name in os.listdir(self.params.dump_path):
        #         if file_name[-4:] == '.pth':
        #             os.remove(os.path.join(self.params.dump_path,file_name))

        # End Wandb
        self.accelerator.end_training()

    def evaluate_model(self, task_id: int) -> dict:
        '''
        Evaluate the model and log the result

        Args:
            - task_id: task_id records how many tasks the model have learned previously

        Return:
            - {

                'Test_Acc_List':[90.42, 92.19, ..., 87.54], # Observed result
                
                'LinearProb_List':[95.2, 94.7, ..., 91.3],          # Linear Probing Performance

                'CosineLinearProb_List':[95.2, 94.7, ..., 91.3],    # Cosine Linear Probing Performance

                'PrototypeProb_List':[95.2, 94.7, ..., 91.3],       # Prototype Probing Performance

                'CosinePrototypeProb_List':[95.2, 94.7, ..., 91.3], # Cosine Prototype Probing Performance

            }
        '''
        result_dict = {}
        log_dict = {}

        # NOTE: 
        # When using classifier, we can only evaluate the model on the tasks which has been learned
        # The reason is that the classifiers of unseen tasks is randomly initialized and the result is meaningless.
        # For example, it is meaningless to let a CIL model which has learned 50 classes to make predictions in 150 classes.
        cur_task_id = task_id
        il_mode = self.params.il_mode
        
        acc_list = self.evaluate_all_seen_task_tc(cur_task_id, 'test', il_mode)
        result_dict['Test_Acc_List'] = acc_list
        for t_id in range(cur_task_id+1):
            log_dict['Test_Acc_Task_%d'%(t_id)] = acc_list[t_id]
        log_dict['Test_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id+1]),3)
        if self.params.classifier=='None':
            log_dict['Test_Acc_Task_All'] = np.round(np.mean(acc_list),3)
        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Test Result = %s'%(il_mode, log_dict))
        
        # Additional Evaluation on training set for Instance Incremental Learning
        if il_mode == 'IIL' and self.params.method != 'ICL':
            acc_list = self.evaluate_all_seen_task_tc(cur_task_id, 'train', il_mode)
            result_dict['Train_Acc_List'] = acc_list
            for t_id in range(cur_task_id+1):
                log_dict['Train_Acc_Task_%d'%(t_id)]=acc_list[t_id]
            log_dict['Train_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id+1]),3)
            if self.params.classifier=='None':
                log_dict['Train_Acc_Task_All'] = np.round(np.mean(acc_list),3)
            if self.accelerator.is_main_process:
                logger.info('Mode = %s, Train Result = %s'%(il_mode, log_dict))

        self.accelerator.log(log_dict,step=self.global_step)

        if self.params.is_probing:
            prob_acc_dict = probing_on_all_task(
                params=self.params,
                task_id=task_id,
                CL_dataset=self.CL_dataset, 
                train_loader_list=self.train_loader_list, 
                test_loader_list=self.test_loader_list, 
                model=self.model, 
                tokenizer=self.tokenizer, 
                accelerator=self.accelerator
            )
            prob_log_dict = {}
            for prob_metric in prob_acc_dict.keys():
                result_dict['%s_List'%(prob_metric)] = prob_acc_dict[prob_metric]
                prob_log_dict['%s_Acc_Task_Seen'%(prob_metric)] = np.round(np.mean(prob_acc_dict[prob_metric][:cur_task_id+1]),3)
                prob_log_dict['%s_Acc_Task_All'%(prob_metric)] = np.round(np.mean(prob_acc_dict[prob_metric]),3)
            if self.accelerator.is_main_process:
                logger.info('Probing Result = %s'%(prob_log_dict))
            self.accelerator.log(prob_log_dict, step=self.global_step)

        return result_dict

    def evaluate_all_seen_task_tc(self,
                                cur_task_id: int, 
                                phase: str,
                                il_mode: str) -> list:
        '''
            Evaluate the model on all seen tasks

            Params: 
                - cur_task_id: cur_task_id records how many tasks the model have learned previously
                - phase: 'train','dev'or'test'
                - il_mode: 'CIL' or 'TIL'

            Return:
                - {
                    [90.42, 92.19, ..., 87.54], 
                }: representing the accuracy of tasks [0, 1, ..., cur_task_id].

        '''
        assert phase in ['train','test','dev']

        acc_list = []

        # Evaluate on all seen tasks
        if self.params.classification_type == 'sentence-level':

            save_dict_all = None

            for eval_t_id in range(cur_task_id+1):

                if il_mode=='IIL':
                    acc, save_dict = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)
                    if save_dict_all is not None:
                        for k,v in save_dict.items():
                            save_dict_all[k] = v
                    else:
                        save_dict_all = save_dict
                else:
                    acc = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)

                acc_list.append(acc)

            save_dict_all = [save_dict_all] # transform to list, otherwise gather_object() will not collect correctly

            gathered_save_dict_all = gather_object(save_dict_all)

            if self.accelerator.is_main_process:
                if gathered_save_dict_all is not None:
                    with open(os.path.join(self.params.dump_path,
                                        '%s_cur_task_%d_save_result.npy'%(phase,cur_task_id)),'wb') as f:
                        np.save(f,gathered_save_dict_all)

        # Evaluate on all seen tasks
        # NOTE: We only test once because the test set of task task_id contains all seen labels from task 0 - task task_id)
        elif self.params.classification_type == 'word-level':

            eval_t_id = cur_task_id

            acc = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)

            acc_list = [acc]*(cur_task_id+1)

        return acc_list 

    @abstractmethod
    def evaluate_current_task(self,
                                eval_task_id: int, 
                                cur_task_id: int, 
                                phase: str,
                                il_mode: str) -> float:
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
        pass
    # ===========================================================================================