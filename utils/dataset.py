import os
import logging
import json
from datasets import Dataset

logger = logging.getLogger()

DATASET2TYPE = {
    'concept_1k_task1': 'sentence-level',
    'concept_1k_task10': 'sentence-level',
    'topic3datasets_task5': 'sentence-level',
    'clinc150_task15':'sentence-level',
    'banking77_task7':'sentence-level',
    'fewrel_task8':'sentence-level',
    'tacred_task8':'sentence-level',
    'conll2003':'word-level',
    'fewnerd_task11_base6_inc6':'word-level',
    'i2b2_task5_base8_inc2':'word-level',
    'i2b2_task1_base16_inc0':'word-level', # Single task (Upper bound)
    'ontonotes5_task6_base8_inc2':'word-level',
    'ontonotes5_task1_base18_inc0':'word-level', # Single task (Upper bound)
}

DATASET2MAXLEN = {
    'concept_1k_task1': 32,
    'concept_1k_task10': 32,
    'topic3datasets_task5': 256,
    'clinc150_task15': 50,
    'banking77_task7': 64,
    'fewrel_task8': 100,
    'tacred_task8': 128,
    'fewnerd_task11_base6_inc6': 128,
    'i2b2_task5_base8_inc2': 128,
    'i2b2_task1_base16_inc0': 128,
    'ontonotes5_task6_base8_inc2': 128,
    'ontonotes5_task1_base18_inc0': 128,
}

def get_dataset(params):
    '''
        Obtain the dataset
    '''
    if params.classification_type == 'auto':
        assert params.dataset in DATASET2TYPE.keys(), 'Dataset %s is not pre-defined in DATASET2TYPE.'%(params.dataset)
        classification_type = DATASET2TYPE[params.dataset]
        setattr(params,'classification_type',classification_type)
    else:
        assert params.classification_type in ['sentence-level','word-level'], 'Invalid classification type %s.'%(params.classification_type)
        classification_type = params.classification_type 

    if params.max_seq_length == -1:
        assert params.dataset in DATASET2MAXLEN.keys(), 'Dataset %s is not pre-defined in DATASET2MAXLEN.'%(params.dataset)
        max_seq_length = DATASET2MAXLEN[params.dataset]
        setattr(params,'max_seq_length',max_seq_length)

    if classification_type == 'sentence-level':
        CL_dataset = Continual_Sentence_Level_Classification_Dataset(dataset = params.dataset) 
    elif classification_type == 'word-level':
        CL_dataset = Continual_Word_Level_Classification_Dataset(dataset = params.dataset) 
    else:
        raise NotImplementedError()

    return CL_dataset

class Continual_Dataset():
    '''
        An abstract class for Continual Learning dataset
    '''
    def __init__(self):
        
        self.continual_data = None
        self.continual_config = None

class Continual_Sentence_Level_Classification_Dataset(Continual_Dataset):
    '''
        Loading a Continual Sentence-Level Classification (e.g., Intent Classification) dataset
    '''
    def __init__(self, dataset):
        super().__init__()
        # Get class list
        assert isinstance(dataset,str), 'dataset should be a str'
        data_path = os.path.join('./dataset',dataset)
        with open(os.path.join(data_path,'continual_config.json')) as f:
            self.continual_config = json.load(f)
        with open(os.path.join(data_path,'continual_data.json')) as f:
            _continual_data = json.load(f)
            continual_data = {
                k: {
                    'train': {'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': []},
                    'test': {'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': []},
                    'dev': {'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': []}
                }
                for k in range(self.continual_config['NUM_TASK']) 
            }
            for k in _continual_data:
                continual_data[int(k)]['train'] = Dataset.from_dict(_continual_data[k]['train'])
                continual_data[int(k)]['dev'] = Dataset.from_dict(_continual_data[k]['dev'])
                continual_data[int(k)]['test'] = Dataset.from_dict(_continual_data[k]['test'])
            self.continual_data = continual_data

        # Data statistic
        logger.info("Train size %s; Dev size %s; Test size: %s;" % (
            [self.continual_data[i]['train'].shape[0] for i in range(self.continual_config['NUM_TASK'])], 
            [self.continual_data[i]['dev'].shape[0] for i in range(self.continual_config['NUM_TASK'])],
            [self.continual_data[i]['test'].shape[0] for i in range(self.continual_config['NUM_TASK'])]))
        if 'LABEL_LIST' in self.continual_config:
            logger.info('Label_list = %s'%str(self.continual_config['LABEL_LIST'])) 

class Continual_Word_Level_Classification_Dataset(Continual_Dataset):
    '''
        Loading a Continual Word-Level Classification (e.g., Named Emtity Recognition) dataset 
    '''
    def __init__(self, dataset):
        super().__init__()
        # Get class list
        assert isinstance(dataset,str), 'dataset should be a str'
        data_path = os.path.join('./dataset',dataset)
        with open(os.path.join(data_path,'continual_config.json')) as f:
            self.continual_config = json.load(f)
        with open(os.path.join(data_path,'continual_data.json')) as f:
            _continual_data = json.load(f)
            continual_data = {
                k: {
                    'train': {'input': [], 'label_idx_cil': []},
                    'test': {'input': [], 'label_idx_cil': []},
                    'dev': {'input': [], 'label_idx_cil': []}
                }
                for k in range(self.continual_config['NUM_TASK']) 
            }
            for k in _continual_data:
                continual_data[int(k)]['train'] = Dataset.from_dict(_continual_data[k]['train'])
                continual_data[int(k)]['dev'] = Dataset.from_dict(_continual_data[k]['dev'])
                continual_data[int(k)]['test'] = Dataset.from_dict(_continual_data[k]['test'])
            self.continual_data = continual_data

        # Data statistic
        logger.info("Train size %s; Dev size %s; Test size: %s;" % (
            [self.continual_data[i]['train'].shape[0] for i in range(self.continual_config['NUM_TASK'])], 
            [self.continual_data[i]['dev'].shape[0] for i in range(self.continual_config['NUM_TASK'])],
            [self.continual_data[i]['test'].shape[0] for i in range(self.continual_config['NUM_TASK'])]))
        logger.info('Label_list = %s'%str(self.continual_config['LABEL_LIST'])) 


if __name__ == "__main__":
    pass
