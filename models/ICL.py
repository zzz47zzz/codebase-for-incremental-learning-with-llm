import logging

from utils.metric import ResultSummary
from utils.backbone import get_backbone
from utils.dataloader import get_dataloader
from utils.evaluation import evaluate_with_generation_ICL
from models.Base import BaseLearner

logger = logging.getLogger()

def get_ICL_params(parser):
    '''
        The parameters of model ICL
    '''
    parser.add_argument("--ICL_same_instance", default=False, type=bool, help="if using same instance pair for demonstration")
    parser.add_argument("--ICL_same_concept", default=False, type=bool, help="if using same concept for demonstration")
    parser.add_argument("--ICL_n_shot", default=1, type=int, help="the number of demonstration")


class ICL(BaseLearner):
    '''
        In-context Learning
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['None'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'SEQ')
        assert params.il_mode in ['IIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'SEQ')
        assert self.CL_dataset.continual_config['NUM_TASK']==1, 'Incontext learning only support one task!'

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)

    def build_classifier(self):
        self.classifier_list = None

    def build_optimizer(self):
        self.optimizer = None

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)
        assert len(self.train_loader_list)==1 and len(self.dev_loader_list)==1 and len(self.test_loader_list)==1

    def build_buffer(self):
        self.buffer = None
    
    def accelerate_prepare(self):
        self.wrap_model, *self.train_loader_list = self.accelerator.prepare(self.model, *self.train_loader_list)
        self.dev_loader_list = [self.accelerator.prepare(*self.dev_loader_list)]
        self.test_loader_list = [self.accelerator.prepare(*self.test_loader_list)]
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)
        
    def end_task(self, task_id):
        super().end_task(task_id)

    # ==============================================================================================

    # ================================= Epoch-Level Functions =======================================
    def train_epochs(self, task_id):
        '''
            Training the model with serveral epochs
        '''
        pass

    def begin_epoch(self, task_id, epoch_id):
        '''
            Start of each epoch
        '''
        self.loss_list = []

    def observe_batch(self, task_id, epoch_id, lm_input):
        '''
            Observe a batch of data
        '''
        # Update step
        pass

    def end_epoch(self, task_id, epoch_id):
        '''
            End of each epoch
        '''
        pass

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

        assert phase in ['test','dev']
        assert il_mode in ['IIL'], 'NotImplemented for il_mode %s'%(il_mode)
        train_data_loader = self.train_loader_list
        if phase=='dev':
            eval_data_loader = self.dev_loader_list
        else:
            eval_data_loader = self.test_loader_list

        # For Distributed Data Parallel
        if hasattr(self.model,'module'):
            model = self.model.module
        else:
            model = self.model

        acc = evaluate_with_generation_ICL(
            model=model,
            train_data_loader=train_data_loader[eval_task_id],
            eval_data_loader=eval_data_loader[eval_task_id],
            tokenizer=self.tokenizer,
            accelerator=self.accelerator,
            params=self.params
        )

        return  acc

    # ===========================================================================================
