# Transform the format of clinc150
import os
import json
import torch
import numpy as np
import pandas as pd
import random
import pickle
import argparse
import re
from collections import Counter

support_dataset_list = ['concept_400','concept_1k',
                        'topic3datasets','clinc150','banking77','clinc150_numbertarget','clinc150_classidxplusone',
                        'fewrel','tacred','conll2003','i2b2','ontonotes5','fewnerd']

def get_params():
    parser = argparse.ArgumentParser(description="The setting for preprocessing datasets")

    parser.add_argument("--dataset", type=str, 
                        default='concept_1k', 
                        choices=support_dataset_list, 
                        help="The dataset selected to be processed")
    parser.add_argument("--seed", 
                        default=1, type=int, 
                        help="The random seed for generating datasets. Using the same seed ensures the same dataset are generated!")

    parser.add_argument("--num_sample_train_per_class", default=3000, type=int, 
                        help="(Only for topic3datasets!) The number of instance for each classes")
    parser.add_argument("--num_sample_test_per_class", default=2000, type=int, 
                        help="(Only for topic3datasets!) The number of instance for each classes")

    parser.add_argument("--base_task_entity", 
                        default=6, type=int, 
                        help="(Only for NER datasets!) The number of classes learned in the first task")
    parser.add_argument("--incremental_task_entity", default=6, type=int, 
                        help="(Only for NER datasets!) The number of classes learned in the each incremental task")
    parser.add_argument("--seen_all_labels", default=True, type=bool, 
                        help="(Only for NER datasets!) If preserve the entity label of other tasks (default=False) \
                        We set it true for probing study and we manually set the future labels to 'O' during training.")

    params = parser.parse_args()

    return params

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(params):
    '''
        Preprocess each dataset to the same format. 
        It saves two files continual_data.json and continual_config.json to the target folder automatically.
        The format of each file is as follows:

        - continual_data.json = {
        
            # task 0

            0: { 
                'train':{
                    'input':[], # The input sentence
                    'target':[], # The target output sentence (e.g., the class name)
                    'label_idx_cil': [], # The index for class-incremental learning (e.g., [0,4,13,6,17,21,5,...] ranging from 0 to num_class-1)
                    'label_idx_til': []  # The index for task-incremental learning (e.g., [0,1,3,2,3,0,2,...] ranging from 0 to num_class_each_task-1)
                },

                'dev':{
                    'input':[],
                    'target':[],
                    'label_idx_cil':[],
                    'label_idx_til':[]
                },

                'test':{
                    'input':[],
                    'target':[],
                    'label_idx_cil':[],
                    'label_idx_til':[]
                }

            },

            ...

            # task N

            N: { 
                ...
            }
        }

        - continual_config.json = {
        
            'NUM_TASK': 15,

            'NUM_CLASS': 150,

            'LABEL_LIST': []

            'CLASSNAME_LIST': []

            'CUR_NUM_CLASS': []

            'CUR_CLASS': []

            'ACCUM_NUM_CLASS': []

            'PRE_ACCUM_NUM_CLASS': []

            'label2idx': {},

            'idx2label': []

        }

    '''
    if params.dataset == 'concept_1k':
        preprocess_concept_1k()
    elif params.dataset == 'topic3datasets':
        preprocess_topic3datasets(num_sample_train_per_class=params.num_sample_train_per_class,
                                  num_sample_test_per_class=params.num_sample_test_per_class)
    elif params.dataset == 'clinc150':
        preprocess_clinc150()
    elif params.dataset == 'clinc150_numbertarget':
        preprocess_clinc150_numbertarget()
    elif params.dataset == 'clinc150_classidxplusone':
        preprocess_clinc150_classidxplusone()
    elif params.dataset == 'banking77':
        preprocess_banking77()
    elif params.dataset == 'fewrel':
        preprocess_fewrel()
    elif params.dataset == 'tacred':
        preprocess_tacred()
    elif params.dataset == 'fewnerd':
        preprocess_fewnerd(dataset=params.dataset,
                                   base_task_entity=params.base_task_entity,
                                   incremental_task_entity=params.incremental_task_entity,
                                   seen_all_labels=params.seen_all_labels)
    elif params.dataset in ['conll2003','i2b2','ontonotes5','fewnerd']:
        preprocess_conll2003format(dataset=params.dataset,
                                   base_task_entity=params.base_task_entity,
                                   incremental_task_entity=params.incremental_task_entity,
                                   seen_all_labels=params.seen_all_labels)
    else:
        raise NotImplementedError()

def preprocess_concept_1k(num_task = 10):
    '''
        Preprocess concept 1k in source folder src_path and save the formatted dataset in target folder tgt_path. 
    '''
    generated_dataset_name = 'concept_1k_task%d'%(num_task)

    src_path = './dataset/concept_1k'
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)
    
    NUM_TASK = num_task

    train_x, train_y, train_concept_id, train_relation_id, train_instance_id = [], [], [], [], []
    dev_x, dev_y, dev_concept_id, dev_relation_id, dev_instance_id = [], [], [], [], []
    test_x, test_y, test_concept_id, test_relation_id, test_instance_id = [], [], [], [], []

    concept_list = []
    relation_list = []
    instanceid_2_conceptid = []
    instanceid_2_relationid = []
    instanceid_2_target = []

    with open(os.path.join(src_path,'dataset.txt')) as f:
        cur_triplet = []
        instance = []
        instanceid = 0
        for i,line in enumerate(f.readlines()):
            line = line.strip()
            if len(line)==0:
                continue
            # Triplet
            if line[0]=='(' and line[-1]==')':
                # Save last instance
                if len(instance)==4:

                    if instance[1]==instance[3]:
                
                        instanceid_2_conceptid.append(concept_list.index(cur_triplet[0]))
                        instanceid_2_relationid.append(relation_list.index(cur_triplet[1]))
                        instanceid_2_target.append(cur_triplet[2])

                        train_x.append(instance[0])
                        train_y.append(instance[1])
                        train_concept_id.append(concept_list.index(cur_triplet[0]))
                        train_relation_id.append(relation_list.index(cur_triplet[1]))
                        train_instance_id.append(instanceid)

                        dev_x.append(instance[2])
                        dev_y.append(instance[3])
                        dev_concept_id.append(concept_list.index(cur_triplet[0]))
                        dev_relation_id.append(relation_list.index(cur_triplet[1]))
                        dev_instance_id.append(instanceid)

                        test_x.append(instance[2])
                        test_y.append(instance[3])
                        test_concept_id.append(concept_list.index(cur_triplet[0]))
                        test_relation_id.append(relation_list.index(cur_triplet[1]))
                        test_instance_id.append(instanceid)

                        instanceid += 1
                    else:
                        print('Error! %s'%(line))

                elif i>0:
                    print('Error! %s'%(line))
                instance = []

                # New instance
                cur_triplet = line[1:-1].split(',')
                if len(cur_triplet)<3:
                    print('Error! %s'%(line))
                    continue
                elif len(cur_triplet)>3:
                    print('Warning! %s'%(line))
                    cur_triplet[0] = cur_triplet[0].strip()
                    cur_triplet[1] = cur_triplet[1].strip()
                    cur_triplet[2:] = ','.join(cur_triplet[2:]).strip()
                else:
                    cur_triplet[0] = cur_triplet[0].strip()
                    cur_triplet[1] = cur_triplet[1].strip()
                    cur_triplet[2] = cur_triplet[2].strip()

                if cur_triplet[0] not in concept_list: 
                    concept_list.append(cur_triplet[0])
                if cur_triplet[1] not in relation_list: 
                    relation_list.append(cur_triplet[1])
                
            elif '1: ' in line or '2: ' in line:
                _, qa = line.split(': ')
                instance.append(qa)

        print('Train:',len(train_y))
        print('Dev:',len(dev_y))
        print('Test:',len(test_y))

        print('Concept List = ',concept_list)
        print('Relation List = ',relation_list)

    num_instance = len(train_x)
    num_concept = len(concept_list)
    num_concept_per_task = num_concept//NUM_TASK
    num_concept_base_task = num_concept - num_concept_per_task*(NUM_TASK-1)

    # # Save Concepts
    # with open(os.path.join(src_path,'cur_concept.txt'),'w') as f1:
    #     for _concept in concept_list:
    #         f1.write('%s\n'%(_concept))

    # Check concepts
    # with open(os.path.join(src_path,'concepts.txt'),'r') as f1:
    #     concepts_all = []
    #     for line in f1.readlines():
    #         _concept = line.strip()
    #         if _concept not in concepts_all:
    #             concepts_all.append(_concept)  

    #     unseen_concept2 = []
    #     for _concept in concepts_all:
    #         if _concept not in concept_list:
    #             unseen_concept2.append(_concept)
    #     print('Unseen Concepts 2 = %s'%(unseen_concept2))

    #     additional_concept2 = []
    #     for _concept in concept_list:
    #         if _concept in concepts_all:
    #             additional_concept2.append(_concept)
    #     print('Overlap Concepts 2 = %s'%(additional_concept2))     

    # # # dedup triplet + remove relation + split words
    # remove_relation_list = ['RelatedTo','Involves','HasContext','AssociatedWith',
    #                         'ConceptuallyRelatedTo','LinkedTo','InContext','ConnectedTo',
    #                         'IsRelatedTo','Involved']
    # with open(os.path.join(src_path,'save_add_dataset_dedup_spilt.txt'),'w') as f1:
    #     def split_on_uppercase(s, keep_contiguous=False):
    #         """

    #             Args:
    #                 s (str): string
    #                 keep_contiguous (bool): flag to indicate we want to
    #                                         keep contiguous uppercase chars together

    #             Returns:

    #         """
    #         string_length = len(s)
    #         is_lower_around = (lambda: s[i-1].islower() or
    #                         string_length > (i + 1) and s[i + 1].islower())

    #         start = 0
    #         parts = []
    #         for i in range(1, string_length):
    #             if s[i].isupper() and (not keep_contiguous or is_lower_around()):
    #                 parts.append(s[start: i].lower())
    #                 start = i
    #         if (not keep_contiguous or is_lower_around()):
    #             parts.append(s[start:].lower())
    #         else:
    #             parts.append(s[start:])
    #         return parts
    #     all_triplet = set()
    #     all_triplet_list = []
    #     for i,(x1,y1,x2,y2,target) in enumerate(zip(train_x,train_y,test_x,test_y,instanceid_2_target)):
    #         cur_triplet = (instanceid_2_conceptid[i],instanceid_2_relationid[i])
    #         if relation_list[instanceid_2_relationid[i]] in remove_relation_list:
    #             continue
    #         all_triplet_list.append(cur_triplet)
    #         if cur_triplet not in all_triplet:
    #             all_triplet.add(cur_triplet)
    #             f1.write('(%s, %s, %s)\n'%(concept_list[instanceid_2_conceptid[i]],
    #                                     relation_list[instanceid_2_relationid[i]],
    #                                     target))
    #             f1.write('Q1: %s\n'%(x1))
    #             y1 = ' '.join(split_on_uppercase(y1, True))
    #             # y1 = ' '.join([word.lower() for word in re.findall('^[a-z]+|[A-Z][^A-Z]*', y1)])
    #             f1.write('A1: %s\n'%(y1.strip()))
    #             f1.write('Q2: %s\n'%(x2))
    #             y2 = ' '.join(split_on_uppercase(y2, True))
    #             # y2 = ' '.join([word.lower() for word in re.findall('^[a-z]+|[A-Z][^A-Z]*', y2)])
    #             f1.write('Q2: %s\n'%(y2.strip()))

    #             # '- ' -> '-'
    #             # 'wi fi' -> 'WiFi'
    #             # '  ' -> ' '
    # count_dict = dict(Counter(instanceid_2_relationid))
    # sorted_keys = sorted(count_dict.keys(),key=lambda k:count_dict[k])
    # print([(relation_list[k],count_dict[k]) for k in sorted_keys])

    # Shuffle Order
    concept_order = list(range(num_concept))
    random.shuffle(concept_order)
    new_concept_list = []
    for i in range(num_concept):
        new_concept_list.append(concept_list[concept_order[i]])
    concept_list = new_concept_list
    for i in range(num_instance):
        train_concept_id[i] = concept_order.index(train_concept_id[i])
        dev_concept_id[i] = concept_order.index(dev_concept_id[i])
        test_concept_id[i] = concept_order.index(test_concept_id[i])
        instanceid_2_conceptid[i] = concept_order.index(instanceid_2_conceptid[i])

    t_id = 0
    CONCEPT_2_TASK = [t_id]*num_concept_base_task
    for t_id in range(1,NUM_TASK):
        CONCEPT_2_TASK.extend([t_id]*num_concept_per_task)

    CUR_NUM_CLASS = [num_concept_base_task] + [num_concept_per_task]*(NUM_TASK-1)  
    ACCUM_NUM_CLASS = np.cumsum(CUR_NUM_CLASS).tolist()
    CUR_CLASS = [list(range(ACCUM_NUM_CLASS[i-1],ACCUM_NUM_CLASS[i])) 
                 if i>0 else list(range(ACCUM_NUM_CLASS[i])) for i in range(len(ACCUM_NUM_CLASS))]
    PRE_ACCUM_NUM_CLASS = [0]+ACCUM_NUM_CLASS[:-1]

    continual_config = {
        'NUM_TASK': NUM_TASK,
        'NUM_CONCEPT': num_concept,
        'CONCEPT_LIST': concept_list,
        'RELATION_LIST': relation_list,
        'CONCEPT_2_TASK': CONCEPT_2_TASK,
        'NUM_CLASS': num_concept,
        'LABEL_LIST': concept_list,
        'CLASSNAME_LIST': concept_list,
        'CUR_NUM_CLASS': CUR_NUM_CLASS,
        'CUR_CLASS': CUR_CLASS,
        'ACCUM_NUM_CLASS': ACCUM_NUM_CLASS,
        'PRE_ACCUM_NUM_CLASS': PRE_ACCUM_NUM_CLASS,
        'label2idx': [],
        'idx2label': [],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'concept_id': [],
                'relation_id': [],
                'instance_id': [],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'concept_id': [],
                'relation_id': [],
                'instance_id': [],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'test':{
                'input':[],
                'target':[],
                'concept_id': [],
                'relation_id': [],
                'instance_id': [],
                'label_idx_cil': [],
                'label_idx_til': [],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _target, concept_id, relation_id, instance_id in zip(train_x, train_y, train_concept_id, train_relation_id, train_instance_id):
        t_id = CONCEPT_2_TASK[concept_id]
        continual_data[t_id]['train']['input'].append(_input)
        continual_data[t_id]['train']['target'].append(_target)
        continual_data[t_id]['train']['concept_id'].append(concept_id)
        continual_data[t_id]['train']['relation_id'].append(relation_id)
        continual_data[t_id]['train']['instance_id'].append(instance_id)
        continual_data[t_id]['train']['label_idx_cil'].append(concept_id) # For data replay
        continual_data[t_id]['train']['label_idx_til'].append(concept_id) # For data replay

    for _input, _target, concept_id, relation_id, instance_id in zip(dev_x, dev_y, dev_concept_id, dev_relation_id, dev_instance_id):
        t_id = CONCEPT_2_TASK[concept_id]
        continual_data[t_id]['dev']['input'].append(_input)
        continual_data[t_id]['dev']['target'].append(_target)
        continual_data[t_id]['dev']['concept_id'].append(concept_id)
        continual_data[t_id]['dev']['relation_id'].append(relation_id)
        continual_data[t_id]['dev']['instance_id'].append(instance_id)
        continual_data[t_id]['dev']['label_idx_cil'].append(concept_id) # For data replay
        continual_data[t_id]['dev']['label_idx_til'].append(concept_id) # For data replay

    for _input, _target, concept_id, relation_id, instance_id in zip(test_x, test_y, test_concept_id, test_relation_id, test_instance_id):
        t_id = CONCEPT_2_TASK[concept_id]
        continual_data[t_id]['test']['input'].append(_input)
        continual_data[t_id]['test']['target'].append(_target)
        continual_data[t_id]['test']['concept_id'].append(concept_id)
        continual_data[t_id]['test']['relation_id'].append(relation_id)
        continual_data[t_id]['test']['instance_id'].append(instance_id)
        continual_data[t_id]['test']['label_idx_cil'].append(concept_id) # For data replay
        continual_data[t_id]['test']['label_idx_til'].append(concept_id) # For data replay

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_topic3datasets(num_sample_train_per_class=1000,num_sample_test_per_class=1000):
    '''
        Preprocess agnews, yahoo, dbpedia in source folder src_path and save the formatted dataset in target folder tgt_path. 
    
        NOTE: We remove the overlapping classes and shuffle the class order randomly.
    '''
    generated_dataset_name = 'topic3datasets_task5'
    subdataset_list = ['agnews','dbpedia','yahoo']

    src_path = './dataset/topic3datasets'
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)

    subdatasets_label_list = {k:[] for k in subdataset_list}
    subdatasets_label_idx = {k:[] for k in subdataset_list}
    label2idx = {}
    global_cnt = 0

    for subdataset in subdataset_list:
        with open(os.path.join(os.path.join(src_path, subdataset),'classes.txt')) as f:
            _cnt = 0
            for line in f.readlines():
                _label = line.strip()
                # We remove 'Sports', 'Business & Finance' and 'Science & Mathematics' from yahoo (becuase agnews contains 'Sports', 'Business', 'Sci/Tech')
                if subdataset=='yahoo' and _label in ['Sports', 'Business & Finance','Science & Mathematics']:
                    _cnt+=1
                    continue
                # Change to semantic labels for generative models
                if _label == 'Sci/Tech':
                    _label = 'Science and Technology'
                elif _label == 'EducationalInstitution':
                    _label = 'Educational Institution'
                elif _label == 'OfficeHolder':
                    _label = 'Office Holder'
                elif _label == 'MeanOfTransportation':
                    _label = 'Mean of Transportation'
                elif _label == 'NaturalPlace':
                    _label = 'Natural Place'
                elif _label == 'WrittenWork':
                    _label = 'Written Work'
                elif '&' in _label:
                    _label = _label.replace('&','and')
                subdatasets_label_list[subdataset].append(_label)
                subdatasets_label_idx[subdataset].append(_cnt)
                _cnt+=1
                label2idx[_label] = global_cnt
                global_cnt+=1

    NUM_TASK = 5
    NUM_CLASS = int(np.sum([len(v) for v in subdatasets_label_list.values()]))

    idx2label = {v:k for k,v in label2idx.items()}
    train_x_all, train_y_all = [], []
    dev_x_all, dev_y_all = [], []
    test_x_all, test_y_all = [], []

    for subdataset in subdataset_list:

        train_x, train_y = [], []
        datapath = os.path.join(os.path.join(src_path, subdataset),'train.csv')
        df = pd.read_csv(datapath, header=None)
        for i, row in df.iterrows():
            _tmp_label_idx = int(row[0])-1
            if _tmp_label_idx not in subdatasets_label_idx[subdataset]:
                continue
            _label = subdatasets_label_list[subdataset][subdatasets_label_idx[subdataset].index(_tmp_label_idx)]
            if 'yahoo' in subdataset:
                input_text = '%s %s %s'%(row[1],row[2],row[3])
            else:
                input_text = '%s %s'%(row[1],row[2])
            if pd.isna(input_text) or len(input_text)==0:
                continue
            train_x.append(input_text)
            train_y.append(label2idx[_label])
        # Downsample
        if num_sample_train_per_class!=-1:
            # Shuffle First
            shuffle_idx = list(range(len(train_y)))
            random.shuffle(shuffle_idx)
            # Select samples for each class
            _select_idx_all = []
            for _label_idx in set(train_y):
                _class_sample_idx = np.where(np.array(train_y)==_label_idx)[0]
                random.shuffle(_class_sample_idx)
                _select_idx_all.extend(_class_sample_idx[:num_sample_train_per_class])
            train_x = [train_x[_i] for _i in _select_idx_all]
            train_y = [train_y[_i] for _i in _select_idx_all]
        train_x_all.extend(train_x)
        train_y_all.extend(train_y)
        print('Train:',Counter(train_y))

        dev_x, dev_y = [], []
        datapath = os.path.join(os.path.join(src_path, subdataset),'test.csv')
        df = pd.read_csv(datapath, header=None)
        for i, row in df.iterrows():
            _tmp_label_idx = int(row[0])-1
            if _tmp_label_idx not in subdatasets_label_idx[subdataset]:
                continue
            _label = subdatasets_label_list[subdataset][subdatasets_label_idx[subdataset].index(_tmp_label_idx)]
            if 'yahoo' in subdataset:
                input_text = '%s %s %s'%(row[1],row[2],row[3])
            else:
                input_text = '%s %s'%(row[1],row[2])
            if pd.isna(input_text) or len(input_text)==0:
                continue
            dev_x.append(input_text)
            dev_y.append(label2idx[_label])
        # Downsample
        if num_sample_test_per_class!=-1:
            # Shuffle First
            shuffle_idx = list(range(len(dev_y)))
            random.shuffle(shuffle_idx)
            # Select samples for each class
            _select_idx_all = []
            for _label_idx in set(dev_y):
                _class_sample_idx = np.where(np.array(dev_y)==_label_idx)[0]
                random.shuffle(_class_sample_idx)
                _select_idx_all.extend(_class_sample_idx[:num_sample_test_per_class])
            dev_x = [dev_x[_i] for _i in _select_idx_all]
            dev_y = [dev_y[_i] for _i in _select_idx_all]
        dev_x_all.extend(dev_x)
        dev_y_all.extend(dev_y)
        print('Dev:',Counter(dev_y))

        test_x, test_y = [], []
        datapath = os.path.join(os.path.join(src_path, subdataset),'test.csv')
        df = pd.read_csv(datapath, header=None)
        for i, row in df.iterrows():
            _tmp_label_idx = int(row[0])-1
            if _tmp_label_idx not in subdatasets_label_idx[subdataset]:
                continue
            _label = subdatasets_label_list[subdataset][subdatasets_label_idx[subdataset].index(_tmp_label_idx)]
            if 'yahoo' in subdataset:
                input_text = '%s %s %s'%(row[1],row[2],row[3])
            else:
                input_text = '%s %s'%(row[1],row[2])
            if pd.isna(input_text) or len(input_text)==0:
                continue
            test_x.append(input_text)
            test_y.append(label2idx[_label])
        # Downsample
        if num_sample_test_per_class!=-1:
            # Shuffle First
            shuffle_idx = list(range(len(test_y)))
            random.shuffle(shuffle_idx)
            # Select samples for each class
            _select_idx_all = []
            for _label_idx in set(test_y):
                _class_sample_idx = np.where(np.array(test_y)==_label_idx)[0]
                random.shuffle(_class_sample_idx)
                _select_idx_all.extend(_class_sample_idx[:num_sample_test_per_class])
            test_x = [test_x[_i] for _i in _select_idx_all]
            test_y = [test_y[_i] for _i in _select_idx_all]
        test_x_all.extend(test_x)
        test_y_all.extend(test_y)
        print('Test:',Counter(test_y))

    # Shuffle the Class Order
    shuffle_class_order = list(range(NUM_CLASS))
    random.shuffle(shuffle_class_order)
    train_y_all = [shuffle_class_order[_y] for _y in train_y_all]
    dev_y_all = [shuffle_class_order[_y] for _y in dev_y_all]
    test_y_all = [shuffle_class_order[_y] for _y in test_y_all]


    continual_config = {
        'NUM_TASK': NUM_TASK,
        'NUM_CLASS': NUM_CLASS,
        'LABEL_LIST': list(range(NUM_CLASS)),
        'CLASSNAME_LIST': [idx2label[label_idx_cil] for label_idx_cil in range(NUM_CLASS)],
        'CUR_NUM_CLASS': [NUM_CLASS//NUM_TASK]*NUM_TASK,
        'CUR_CLASS': [list(range(task_id*(NUM_CLASS//NUM_TASK),(task_id+1)*(NUM_CLASS//NUM_TASK))) for task_id in range(NUM_TASK)],
        'ACCUM_NUM_CLASS': np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist(),
        'PRE_ACCUM_NUM_CLASS': [c-NUM_CLASS//NUM_TASK for c in np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist()],
        'label2idx': label2idx,
        'idx2label': [idx2label[i] for i in range(NUM_CLASS)],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            },
            'test':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _label_idx in zip(train_x_all,train_y_all):
        _label_idx_cil = _label_idx
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['target'].append(idx2label[_label_idx])

    for _input, _label_idx in zip(dev_x_all,dev_y_all):
        _label_idx_cil = _label_idx
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['target'].append(idx2label[_label_idx])

    for _input, _label_idx in zip(test_x_all,test_y_all):
        _label_idx_cil = _label_idx
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['target'].append(idx2label[_label_idx])

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_clinc150():
    '''
        Preprocess clinc150 in source folder src_path and save the formatted dataset in target folder tgt_path. 
    '''
    generated_dataset_name = 'clinc150_task15'

    src_path = './dataset/clinc150'
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)
    
    NUM_TASK = 15
    NUM_CLASS = 150

    assert NUM_CLASS%NUM_TASK==0

    with open(os.path.join(src_path,'label_dict.json')) as f:
        label_dict = json.load(f)
        label_dict = {k.replace('_', ' '):v for k,v in label_dict.items()}
        idx2label = {v:k for k,v in label_dict.items()}

    with open(os.path.join(src_path,'data_full.json')) as f:
        all_data = json.load(f)

    train_x = [row[0] for row in all_data['train']]
    train_y = [row[1].replace('_', ' ') for row in all_data['train']]
    print('Train:',Counter(train_y))
    dev_x = [row[0] for row in all_data['val']]
    dev_y = [row[1].replace('_', ' ') for row in all_data['val']]
    print('Dev:',Counter(dev_y))
    test_x = [row[0] for row in all_data['test']]
    test_y = [row[1].replace('_', ' ') for row in all_data['test']]
    print('Test:',Counter(test_y))

    continual_config = {
        'NUM_TASK': 15,
        'NUM_CLASS': 150,
        'LABEL_LIST': list(range(NUM_CLASS)),
        'CLASSNAME_LIST': [idx2label[label_idx_cil] for label_idx_cil in range(NUM_CLASS)],
        'CUR_NUM_CLASS': [NUM_CLASS//NUM_TASK]*NUM_TASK,
        'CUR_CLASS': [list(range(task_id*(NUM_CLASS//NUM_TASK),(task_id+1)*(NUM_CLASS//NUM_TASK))) for task_id in range(NUM_TASK)],
        'ACCUM_NUM_CLASS': np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist(),
        'PRE_ACCUM_NUM_CLASS': [c-NUM_CLASS//NUM_TASK for c in np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist()],
        'label2idx': label_dict,
        'idx2label': [idx2label[i] for i in range(NUM_CLASS)],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            },
            'test':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _target in zip(train_x,train_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['target'].append(_target)

    for _input, _target in zip(dev_x,dev_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['target'].append(_target)

    for _input, _target in zip(test_x,test_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['target'].append(_target)

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_clinc150_numbertarget():
    '''
        Preprocess clinc150 in source folder src_path and save the formatted dataset in target folder tgt_path.
        The target is number instead of words. 
        For analysis purpose only!
    '''

    generated_dataset_name = 'clinc150_task15_numbertarget'

    src_path = './dataset/clinc150'
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)

    NUM_TASK = 15
    NUM_CLASS = 150

    assert NUM_CLASS%NUM_TASK==0

    with open(os.path.join(src_path,'label_dict.json')) as f:
        label_dict = json.load(f)
        label_dict = {k.replace('_', ' '):v for k,v in label_dict.items()}
        idx2label = {v:k for k,v in label_dict.items()}

    with open(os.path.join(src_path,'data_full.json')) as f:
        all_data = json.load(f)

    train_x = [row[0] for row in all_data['train']]
    train_y = [row[1].replace('_', ' ') for row in all_data['train']]
    print('Train:',Counter(train_y))
    dev_x = [row[0] for row in all_data['val']]
    dev_y = [row[1].replace('_', ' ') for row in all_data['val']]
    print('Dev:',Counter(dev_y))
    test_x = [row[0] for row in all_data['test']]
    test_y = [row[1].replace('_', ' ') for row in all_data['test']]
    print('Test:',Counter(test_y))

    _modified_label_dict = {str(i):i for i in range(NUM_CLASS)}
    _modified_idx2label = {i:str(i) for i in range(NUM_CLASS)}

    continual_config = {
        'NUM_TASK': 15,
        'NUM_CLASS': 150,
        'LABEL_LIST': list(range(NUM_CLASS)),
        'CLASSNAME_LIST': [_modified_idx2label[label_idx_cil] for label_idx_cil in range(NUM_CLASS)],
        'CUR_NUM_CLASS': [NUM_CLASS//NUM_TASK]*NUM_TASK,
        'CUR_CLASS': [list(range(task_id*(NUM_CLASS//NUM_TASK),(task_id+1)*(NUM_CLASS//NUM_TASK))) for task_id in range(NUM_TASK)],
        'ACCUM_NUM_CLASS': np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist(),
        'PRE_ACCUM_NUM_CLASS': [c-NUM_CLASS//NUM_TASK for c in np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist()],
        'label2idx': _modified_label_dict,
        'idx2label': [_modified_idx2label[i] for i in range(NUM_CLASS)],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            },
            'test':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _target in zip(train_x,train_y):
        _label_idx_cil = label_dict[_target]
        _modified_target = str(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['target'].append(_modified_target)

    for _input, _target in zip(dev_x,dev_y):
        _label_idx_cil = label_dict[_target]
        _modified_target = str(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['target'].append(_modified_target)

    for _input, _target in zip(test_x,test_y):
        _label_idx_cil = label_dict[_target]
        _modified_target = str(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['target'].append(_modified_target)

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_clinc150_classidxplusone():
    '''
        Preprocess clinc150 in source folder src_path and save the formatted dataset in target folder tgt_path.
        The ground-truth class label is replaced by the label whose index is one plus the label index of the original label.
        For analysis purpose only!
    '''
    generated_dataset_name = 'clinc150_task15_classidxplusone'

    src_path = './dataset/clinc150'
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)

    NUM_TASK = 15
    NUM_CLASS = 150

    assert NUM_CLASS%NUM_TASK==0

    with open(os.path.join(src_path,'label_dict.json')) as f:
        label_dict = json.load(f)
        label_dict = {k.replace('_', ' '):v for k,v in label_dict.items()}
        idx2label = {v:k for k,v in label_dict.items()}

    with open(os.path.join(src_path,'data_full.json')) as f:
        all_data = json.load(f)

    train_x = [row[0] for row in all_data['train']]
    train_y = [row[1].replace('_', ' ') for row in all_data['train']]
    print('Train:',Counter(train_y))
    dev_x = [row[0] for row in all_data['val']]
    dev_y = [row[1].replace('_', ' ') for row in all_data['val']]
    print('Dev:',Counter(dev_y))
    test_x = [row[0] for row in all_data['test']]
    test_y = [row[1].replace('_', ' ') for row in all_data['test']]
    print('Test:',Counter(test_y))

    continual_config = {
        'NUM_TASK': 15,
        'NUM_CLASS': 150,
        'LABEL_LIST': list(range(NUM_CLASS)),
        'CLASSNAME_LIST': [idx2label[label_idx_cil] for label_idx_cil in range(NUM_CLASS)],
        'CUR_NUM_CLASS': [NUM_CLASS//NUM_TASK]*NUM_TASK,
        'CUR_CLASS': [list(range(task_id*(NUM_CLASS//NUM_TASK),(task_id+1)*(NUM_CLASS//NUM_TASK))) for task_id in range(NUM_TASK)],
        'ACCUM_NUM_CLASS': np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist(),
        'PRE_ACCUM_NUM_CLASS': [c-NUM_CLASS//NUM_TASK for c in np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist()],
        'label2idx': label_dict,
        'idx2label': [idx2label[i] for i in range(NUM_CLASS)],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            },
            'test':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _target in zip(train_x,train_y):
        _label_idx_cil = label_dict[_target]
        _modified_label_idx = (_label_idx_cil+1)%NUM_CLASS
        _modified_target = idx2label[_modified_label_idx]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_cil'].append(_modified_label_idx)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_til'].append(_modified_label_idx%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['target'].append(_modified_target)

    for _input, _target in zip(dev_x,dev_y):
        _label_idx_cil = label_dict[_target]
        _modified_label_idx = (_label_idx_cil+1)%NUM_CLASS
        _modified_target = idx2label[_modified_label_idx]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_cil'].append(_modified_label_idx)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_til'].append(_modified_label_idx%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['target'].append(_modified_target)

    for _input, _target in zip(test_x,test_y):
        _label_idx_cil = label_dict[_target]
        _modified_label_idx = (_label_idx_cil+1)%NUM_CLASS
        _modified_target = idx2label[_modified_label_idx]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_cil'].append(_modified_label_idx)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_til'].append(_modified_label_idx%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['target'].append(_modified_target)

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_banking77():
    '''
        Preprocess banking77 in source folder src_path and save the formatted dataset in target folder tgt_path. 
    '''
    generated_dataset_name = 'banking77_task7'

    src_path = './dataset/banking77'
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)
    
    NUM_TASK = 7
    NUM_CLASS = 77

    assert NUM_CLASS%NUM_TASK==0

    with open(os.path.join(src_path,'categories.json')) as f:
        label_list = json.load(f)
        label_dict = {v.replace('_', ' '):k for k,v in enumerate(label_list)}
        idx2label = {k:v.replace('_', ' ') for k,v in enumerate(label_list)}

    train_df = pd.read_csv(os.path.join(src_path,'train.csv'))
    train_x = [row[0] for i, row in train_df.iterrows()]
    train_y = [row[1].replace('_', ' ') for i, row in train_df.iterrows()]
    print('Train:',Counter(train_y))
    dev_df = pd.read_csv(os.path.join(src_path,'test.csv'))
    dev_x = [row[0] for i, row in dev_df.iterrows()]
    dev_y = [row[1].replace('_', ' ') for i, row in dev_df.iterrows()]
    print('Dev:',Counter(dev_y))
    test_df = pd.read_csv(os.path.join(src_path,'test.csv'))
    test_x = [row[0] for i, row in test_df.iterrows()]
    test_y = [row[1].replace('_', ' ') for i, row in test_df.iterrows()]
    print('Test:',Counter(test_y))

    continual_config = {
        'NUM_TASK': 7,
        'NUM_CLASS': 77,
        'LABEL_LIST': list(range(NUM_CLASS)),
        'CLASSNAME_LIST': [idx2label[label_idx_cil] for label_idx_cil in range(NUM_CLASS)],
        'CUR_NUM_CLASS': [NUM_CLASS//NUM_TASK]*NUM_TASK,
        'CUR_CLASS': [list(range(task_id*(NUM_CLASS//NUM_TASK),(task_id+1)*(NUM_CLASS//NUM_TASK))) for task_id in range(NUM_TASK)],
        'ACCUM_NUM_CLASS': np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist(),
        'PRE_ACCUM_NUM_CLASS': [c-NUM_CLASS//NUM_TASK for c in np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist()],
        'label2idx': label_dict,
        'idx2label': [idx2label[i] for i in range(NUM_CLASS)],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            },
            'test':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _target in zip(train_x,train_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['target'].append(_target)

    for _input, _target in zip(dev_x,dev_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['target'].append(_target)

    for _input, _target in zip(test_x,test_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['target'].append(_target)

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))


def preprocess_fewrel():
    '''
        Preprocess fewrel in source folder src_path and save the formatted dataset in target folder tgt_path. 
    '''
    generated_dataset_name = 'fewrel_task8'

    src_path = './dataset/fewrel/FewRel-2021.pkl'
    tgt_path = './dataset/%s/'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)

    NUM_TASK = 8
    NUM_CLASS = 80

    assert NUM_CLASS%NUM_TASK==0

    with open(src_path, 'rb') as f:
        all_data = pickle.load(f)
        label_dict = {}
        train_x = []
        train_y = []
        for i in range(NUM_CLASS):
            if all_data[0][i][0]['semantic_label'] not in label_dict.keys():
                label_dict[all_data[0][i][0]['semantic_label']] = i
            for instance in all_data[0][i]:
                train_x.append(instance['text'])    
                train_y.append(instance['semantic_label'])
        idx2label = {v:k for k,v in label_dict.items()}
        print('Train:',Counter(train_y))

        dev_x = []
        dev_y = []
        for i in range(NUM_CLASS):
            for instance in all_data[1][i]:
                dev_x.append(instance['text'])    
                dev_y.append(instance['semantic_label'])
        print('Dev:',Counter(dev_y))

        test_x = []
        test_y = []
        for i in range(NUM_CLASS):
            for instance in all_data[2][i]:
                test_x.append(instance['text'])    
                test_y.append(instance['semantic_label'])
        print('Test:',Counter(test_y))

    continual_config = {
        'NUM_TASK': 8,
        'NUM_CLASS': 80,
        'LABEL_LIST': list(range(NUM_CLASS)),
        'CLASSNAME_LIST': [idx2label[label_idx] for label_idx in range(NUM_CLASS)],
        'CUR_NUM_CLASS': [NUM_CLASS//NUM_TASK]*NUM_TASK,
        'CUR_CLASS': [list(range(task_id*(NUM_CLASS//NUM_TASK),(task_id+1)*(NUM_CLASS//NUM_TASK))) for task_id in range(NUM_TASK)],
        'ACCUM_NUM_CLASS': np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist(),
        'PRE_ACCUM_NUM_CLASS': [c-NUM_CLASS//NUM_TASK for c in np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist()],
        'label2idx': label_dict,
        'idx2label': [idx2label[i] for i in range(NUM_CLASS)],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            },
            'test':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _target in zip(train_x,train_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['target'].append(_target)

    for _input, _target in zip(dev_x,dev_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['target'].append(_target)

    for _input, _target in zip(test_x,test_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['target'].append(_target)

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_tacred():
    '''
        Preprocess tacred in source folder src_path and save the formatted dataset in target folder tgt_path. 
    '''
    generated_dataset_name = 'tacred_task8'

    src_path = './dataset/tacred/TACRED-2021.pkl'
    tgt_path = './dataset/%s/'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)

    NUM_TASK = 8
    NUM_CLASS = 40

    assert NUM_CLASS%NUM_TASK==0

    # This mapping is from the repository https://github.com/shaoyijia/VAG.
    TACRED_SEMANTIC_LABEL = {
        'org:founded_by': 'organization related: founded by', 'per:employee_of': 'person related: employee of',
        'org:alternate_names': 'organization related: alternate names',
        'per:cities_of_residence': 'person related: city of residence',
        'per:children': 'person related: children', 'per:title': 'person related: title',
        'per:siblings': 'person related: siblings', 'per:religion': 'person related: religion',
        'per:age': 'person related: age', 'org:website': 'organization related: website',
        'per:stateorprovinces_of_residence': 'person related: state or provinces of residence',
        'org:member_of': 'organization related: member of',
        'org:top_members/employees': 'organization related: top members/employees',
        'per:countries_of_residence': 'person related: countries of residence',
        'org:city_of_headquarters': 'organization related: city of headquarters',
        'org:members': 'organization related: members',
        'org:country_of_headquarters': 'organization related: country of headquarters',
        'per:spouse': 'person related: spouse',
        'org:stateorprovince_of_headquarters': 'organization related: state or province of headquarters',
        'org:number_of_employees/members': 'organization related: number of employees/members',
        'org:parents': 'organization related: parents', 'org:subsidiaries': 'organization related: subsidiaries',
        'per:origin': 'person related: origin',
        'org:political/religious_affiliation': 'organization related: political/religious affiliation',
        'per:other_family': 'person related: other family',
        'per:stateorprovince_of_birth': 'person related: state or province of birth',
        'org:dissolved': 'organization related: dissolved', 'per:date_of_death': 'person related: date of death',
        'org:shareholders': 'organization related: shareholders', 'per:alternate_names': 'person related: alternate names',
        'per:parents': 'person related: parents', 'per:schools_attended': 'person related: schools attended',
        'per:cause_of_death': 'person related: cause of death', 'per:city_of_death': 'person related: city of death',
        'per:stateorprovince_of_death': 'person related: state or province of death',
        'org:founded': 'organization related: founded', 'per:country_of_birth': 'person related: country of birth',
        'per:date_of_birth': 'person related: date of birth', 'per:city_of_birth': 'person related: city of birth',
        'per:charges': 'person related: charges'
    }  # Expand the abbreviation in the original labels.

    with open(src_path, 'rb') as f:
        train_data, _, test_data = pickle.load(f)
        label_dict = {}
        train_x = []
        train_y = []
        for i in range(NUM_CLASS):
            for j, instance in enumerate(train_data[i]):
                if j==0:
                    label_dict[TACRED_SEMANTIC_LABEL[instance['semantic_label']]] = i
                train_x.append(instance['text'])    
                train_y.append(TACRED_SEMANTIC_LABEL[instance['semantic_label']])
        idx2label = {v:k for k,v in label_dict.items()}
        print('Train:',Counter(train_y))

        dev_x = []
        dev_y = []
        for i in range(NUM_CLASS):
            for instance in test_data[i]:
                dev_x.append(instance['text'])    
                dev_y.append(TACRED_SEMANTIC_LABEL[instance['semantic_label']])
        print('Dev:',Counter(dev_y))

        test_x = []
        test_y = []
        for i in range(NUM_CLASS):
            for instance in test_data[i]:
                test_x.append(instance['text'])    
                test_y.append(TACRED_SEMANTIC_LABEL[instance['semantic_label']])
        print('Test:',Counter(test_y))

    continual_config = {
        'NUM_TASK': 8,
        'NUM_CLASS': 40,
        'LABEL_LIST': list(range(NUM_CLASS)),
        'CLASSNAME_LIST': [idx2label[label_idx] for label_idx in range(NUM_CLASS)],
        'CUR_NUM_CLASS': [NUM_CLASS//NUM_TASK]*NUM_TASK,
        'CUR_CLASS': [list(range(task_id*(NUM_CLASS//NUM_TASK),(task_id+1)*(NUM_CLASS//NUM_TASK))) for task_id in range(NUM_TASK)],
        'ACCUM_NUM_CLASS': np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist(),
        'PRE_ACCUM_NUM_CLASS': [c-NUM_CLASS//NUM_TASK for c in np.cumsum([NUM_CLASS//NUM_TASK]*NUM_TASK).tolist()],
        'label2idx': label_dict,
        'idx2label': [idx2label[i] for i in range(NUM_CLASS)],
    }

    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'target':[],
                'label_idx_cil': [],
                'label_idx_til': [],
            },
            'dev':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            },
            'test':{
                'input':[],
                'target':[],
                'label_idx_cil':[],
                'label_idx_til':[],
            }
        }
        continual_data[task_id] = one_task_data

    for _input, _target in zip(train_x,train_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['train']['target'].append(_target)

    for _input, _target in zip(dev_x,dev_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['dev']['target'].append(_target)

    for _input, _target in zip(test_x,test_y):
        _label_idx_cil = label_dict[_target]
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_cil'].append(_label_idx_cil)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['label_idx_til'].append(_label_idx_cil%(NUM_CLASS//NUM_TASK))
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['input'].append(_input)
        continual_data[_label_idx_cil//(NUM_CLASS//NUM_TASK)]['test']['target'].append(_target)

    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_conll2003format(dataset,base_task_entity=1,incremental_task_entity=1,seen_all_labels=False):
    '''
        Preprocess conll2003 format dataset in source folder src_path and save the formatted dataset in target folder tgt_path. 
    
        NOTE: 
        - We use BIO tagging schema and the GreedySampling algorithm for Continual NER 
        ([Distilling Causal Effect from Miscellaneous Other-Class for Continual Named Entity Recognition](https://aclanthology.org/2022.emnlp-main.236) (Zheng et al., EMNLP 2022)).
        - We learn the entity according to the alphabetical order
    '''

    src_path = './dataset/%s'%(dataset)

    train_file = os.path.join(src_path,'train.txt')
    dev_file = os.path.join(src_path,'dev.txt')
    test_file = os.path.join(src_path,'test.txt')

    # Step 1: Obtain all entities information
    train_label_list = []
    with open(os.path.join(train_file), 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            try:
                token_list = line.strip().split(' ')
                assert len(token_list) == 2
            except Exception as e:
                print(e)
                token_list = line.strip().split('\t')
                assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
            word, tag = token_list
            train_label_list.append(tag.upper())
    train_entity_set = sorted([tag.split('-')[1] for tag in list(set(train_label_list)) if tag!='O' and 'B-' in tag])
    print(Counter(train_label_list))
    
    dev_label_list = []
    with open(os.path.join(dev_file), 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            try:
                token_list = line.strip().split(' ')
                assert len(token_list) == 2
            except Exception as e:
                print(e)
                token_list = line.strip().split('\t')
                assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
            word, tag = token_list
            dev_label_list.append(tag.upper())
    dev_entity_set = sorted([tag.split('-')[1] for tag in list(set(dev_label_list)) if tag!='O' and 'B-' in tag])
    print(Counter(dev_label_list))
    
    test_label_list = []
    with open(os.path.join(test_file), 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            try:
                token_list = line.strip().split(' ')
                assert len(token_list) == 2
            except Exception as e:
                print(e)
                token_list = line.strip().split('\t')
                assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
            word, tag = token_list
            test_label_list.append(tag.upper())
    test_entity_set = sorted([tag.split('-')[1] for tag in list(set(test_label_list)) if tag!='O' and 'B-' in tag])
    print(Counter(test_label_list))

    assert (train_entity_set == dev_entity_set) and (train_entity_set == test_entity_set)
    freq_sorted_entity_list = sorted(train_entity_set, key=lambda x:Counter(train_label_list)['B-'+x])
    alpha_sorted_entity_list = sorted(train_entity_set)

    # Step 2: Obtain config for continual learning
    NUM_CLASS = 1+2*len(alpha_sorted_entity_list)
    if incremental_task_entity>0:
        assert (len(alpha_sorted_entity_list)-base_task_entity)%incremental_task_entity==0
        NUM_TASK = 1+(len(alpha_sorted_entity_list)-base_task_entity)//incremental_task_entity
    else:
        NUM_TASK = 1
    LABEL_LIST = list(range(NUM_CLASS))
    label2idx = {'O':0}
    idx2label = ['O']
    for idx,tag in enumerate(alpha_sorted_entity_list):
        label2idx['B-'+tag] = 1+2*idx
        label2idx['I-'+tag] = 1+2*idx+1
        idx2label.append('B-'+tag)
        idx2label.append('I-'+tag)
    CLASSNAME_LIST = idx2label
    CUR_NUM_CLASS = [1+2*base_task_entity]
    for t_id in range(1,NUM_TASK):
        CUR_NUM_CLASS.append(2*incremental_task_entity)
    ACCUM_NUM_CLASS = np.cumsum(CUR_NUM_CLASS).tolist()
    PRE_ACCUM_NUM_CLASS = [0]
    PRE_ACCUM_NUM_CLASS.extend(ACCUM_NUM_CLASS[:-1])
    CUR_CLASS = [list(range(ACCUM_NUM_CLASS[0]))]
    for t_id in range(1,NUM_TASK):
        CUR_CLASS.append(list(range(ACCUM_NUM_CLASS[t_id-1],ACCUM_NUM_CLASS[t_id])))

    continual_config = {
        'NUM_TASK': NUM_TASK,
        'NUM_CLASS': NUM_CLASS,
        'LABEL_LIST': LABEL_LIST,
        'CLASSNAME_LIST': CLASSNAME_LIST,
        'CUR_NUM_CLASS': CUR_NUM_CLASS,
        'CUR_CLASS': CUR_CLASS,
        'ACCUM_NUM_CLASS': ACCUM_NUM_CLASS,
        'PRE_ACCUM_NUM_CLASS': PRE_ACCUM_NUM_CLASS,
        'label2idx': label2idx,
        'idx2label': idx2label,
    }

    generated_dataset_name = '%s_task%d_base%d_inc%d'%(dataset,NUM_TASK,base_task_entity,incremental_task_entity)
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)

    # Step 3: Obtain all input and labels
    def load_ner_data(load_file):
        all_x = []
        all_y = []
        one_sentence_x = []
        one_sentence_y = []
        with open(os.path.join(load_file), 'r') as f:
            for line in f.readlines():
                if line.strip() == '':
                    if len(one_sentence_x)>0:
                        all_x.append(one_sentence_x)
                        all_y.append(one_sentence_y)
                        one_sentence_x = []
                        one_sentence_y = []
                    continue
                try:
                    token_list = line.strip().split(' ')
                    assert len(token_list) == 2
                except Exception as e:
                    print(e)
                    token_list = line.strip().split('\t')
                    assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
                word, tag = token_list
                one_sentence_x.append(word)
                one_sentence_y.append(tag.upper())
            if len(one_sentence_x)>0:
                all_x.append(one_sentence_x)
                all_y.append(one_sentence_y)
                one_sentence_x = []
                one_sentence_y = []
        return all_x, all_y

    train_x, train_y = load_ner_data(train_file)
    dev_x, dev_y = load_ner_data(dev_file)
    test_x, test_y = load_ner_data(test_file)  

    # Step 4: Split the data into each tasks and adjust the label
    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'label_idx_cil': [],
            },
            'dev':{
                'input':[],
                'label_idx_cil':[],
            },
            'test':{
                'input':[],
                'label_idx_cil':[],
            }
        }
        continual_data[task_id] = one_task_data

    # Step 4.1: For training data, we use Greedy Sampling Algorithm and mask the label of other tasks to 'O'.
    num_all_sent = len(train_x)      
    num_entity_task_list = [base_task_entity]
    for t_id in range(1,NUM_TASK):
        num_entity_task_list.append(incremental_task_entity)
    # how many sentence should be added
    sent_cnt_task = [int(num_all_sent*num_entity_task_list[t_id]/np.sum(num_entity_task_list)) for t_id in range(NUM_TASK)]
    # mapping entity to the task id
    entity2task = {}
    for idx, e in enumerate(alpha_sorted_entity_list):
        entity2task[e] = 1+(idx-base_task_entity)//incremental_task_entity if idx>=base_task_entity else 0
    # what labels can be seen in each tasks
    task2labels_train = {} # Only entities for each task is preserved, while other entities are masked.
    task2labels_test = {} # The seen entity is accumulated
    for t_id in range(NUM_TASK):
        task2labels_train[t_id] = []
        task2labels_train[t_id].append('O')
        task2labels_test[t_id] = []
        task2labels_test[t_id].append('O')
        if t_id == 0:
            for e in alpha_sorted_entity_list[:np.cumsum(num_entity_task_list)[0]]:
                task2labels_train[t_id].append('B-'+e)
                task2labels_train[t_id].append('I-'+e)
            for e in alpha_sorted_entity_list[:np.cumsum(num_entity_task_list)[0]]:
                task2labels_test[t_id].append('B-'+e)
                task2labels_test[t_id].append('I-'+e)
        else:
            for e in alpha_sorted_entity_list[np.cumsum(num_entity_task_list)[t_id-1]:np.cumsum(num_entity_task_list)[t_id]]:
                task2labels_train[t_id].append('B-'+e)
                task2labels_train[t_id].append('I-'+e)
            for e in alpha_sorted_entity_list[:np.cumsum(num_entity_task_list)[t_id]]:
                task2labels_test[t_id].append('B-'+e)
                task2labels_test[t_id].append('I-'+e)

    for x_sent, y_sent in zip(train_x, train_y):
        is_break = False
        for entity in freq_sorted_entity_list:
            t_id = entity2task[entity]
            if 'B-'+entity in y_sent and sent_cnt_task[t_id]>0:
                continual_data[t_id]['train']['input'].append(x_sent)
                if seen_all_labels:
                    continual_data[t_id]['train']['label_idx_cil'].append(y_sent)
                else:
                    masked_y_sent = [tag if tag in task2labels_train[t_id] else 'O' for tag in y_sent]
                    continual_data[t_id]['train']['label_idx_cil'].append(masked_y_sent)
                sent_cnt_task[t_id] -= 1
                is_break = True
                break
        if not is_break:
            not_full_split = np.where(np.array(sent_cnt_task)>0)[0]
            if len(not_full_split)>0:
                t_id = np.random.choice(not_full_split)
                continual_data[t_id]['train']['input'].append(x_sent)
                if seen_all_labels:
                    continual_data[t_id]['train']['label_idx_cil'].append(y_sent)
                else:
                    masked_y_sent = [tag if tag in task2labels_train[t_id] else 'O' for tag in y_sent]
                    continual_data[t_id]['train']['label_idx_cil'].append(masked_y_sent)
                sent_cnt_task[t_id] -= 1

    # Step 4.2: For dev or test data, we mask the label of unseen tasks to 'O'.
    for t_id in range(NUM_TASK):
        for x_sent, y_sent in zip(dev_x, dev_y):
            masked_y_sent = [tag if tag in task2labels_test[t_id] else 'O' for tag in y_sent]
            if list(set(masked_y_sent)) != ['O']:
                continual_data[t_id]['dev']['input'].append(x_sent)
                continual_data[t_id]['dev']['label_idx_cil'].append(masked_y_sent)

    for t_id in range(NUM_TASK):
        for x_sent, y_sent in zip(test_x, test_y):
            masked_y_sent = [tag if tag in task2labels_test[t_id] else 'O' for tag in y_sent]
            if list(set(masked_y_sent)) != ['O']:
                continual_data[t_id]['test']['input'].append(x_sent)
                continual_data[t_id]['test']['label_idx_cil'].append(masked_y_sent)


    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))

def preprocess_fewnerd(dataset,base_task_entity=6,incremental_task_entity=6,seen_all_labels=False):
    '''
        Preprocess fewnerd dataset in source folder src_path and save the formatted dataset in target folder tgt_path. 
    
        NOTE: 
        - We use BIO tagging schema and the GreedySampling algorithm for Continual NER 
        ([Distilling Causal Effect from Miscellaneous Other-Class for Continual Named Entity Recognition](https://aclanthology.org/2022.emnlp-main.236) (Zheng et al., EMNLP 2022)).
        - The order of the entity is randomly shuffled.
    '''

    src_path = './dataset/%s'%(dataset)

    train_file = os.path.join(src_path,'train.txt')
    dev_file = os.path.join(src_path,'dev.txt')
    test_file = os.path.join(src_path,'test.txt')

    # Step 1: Obtain all entities information
    train_label_list = []
    with open(os.path.join(train_file), 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            try:
                token_list = line.strip().split('\t')
                assert len(token_list) == 2
            except Exception as e:
                print(e)
                token_list = line.strip().split(' ')
                assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
            word, tag = token_list
            train_label_list.append(tag.upper())
    train_entity_set = sorted([tag for tag in list(set(train_label_list)) if tag!='O'])
    print(Counter(train_label_list))
    
    dev_label_list = []
    with open(os.path.join(dev_file), 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            try:
                token_list = line.strip().split('\t')
                assert len(token_list) == 2
            except Exception as e:
                print(e)
                token_list = line.strip().split(' ')
                assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
            word, tag = token_list
            dev_label_list.append(tag.upper())
    dev_entity_set = sorted([tag for tag in list(set(dev_label_list)) if tag!='O'])
    print(Counter(dev_label_list))
    
    test_label_list = []
    with open(os.path.join(test_file), 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            try:
                token_list = line.strip().split('\t')
                assert len(token_list) == 2
            except Exception as e:
                print(e)
                token_list = line.strip().split(' ')
                assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
            word, tag = token_list
            test_label_list.append(tag.upper())
    test_entity_set = sorted([tag for tag in list(set(test_label_list)) if tag!='O'])
    print(Counter(test_label_list))

    assert (train_entity_set == dev_entity_set) and (train_entity_set == test_entity_set)
    freq_sorted_entity_list = sorted(train_entity_set, key=lambda x:Counter(train_label_list)[x])
    alpha_sorted_entity_list = sorted(train_entity_set)
    # shuffle the entity order
    random.shuffle(alpha_sorted_entity_list)

    # Step 2: Obtain config for continual learning
    NUM_CLASS = 1+2*len(alpha_sorted_entity_list)
    if incremental_task_entity>0:
        assert (len(alpha_sorted_entity_list)-base_task_entity)%incremental_task_entity==0
        NUM_TASK = 1+(len(alpha_sorted_entity_list)-base_task_entity)//incremental_task_entity
    else:
        NUM_TASK = 1
    LABEL_LIST = list(range(NUM_CLASS))
    label2idx = {'O':0}
    idx2label = ['O']
    for idx,tag in enumerate(alpha_sorted_entity_list):
        label2idx['B-'+tag] = 1+2*idx
        label2idx['I-'+tag] = 1+2*idx+1
        idx2label.append('B-'+tag)
        idx2label.append('I-'+tag)
    CLASSNAME_LIST = idx2label
    CUR_NUM_CLASS = [1+2*base_task_entity]
    for t_id in range(1,NUM_TASK):
        CUR_NUM_CLASS.append(2*incremental_task_entity)
    ACCUM_NUM_CLASS = np.cumsum(CUR_NUM_CLASS).tolist()
    PRE_ACCUM_NUM_CLASS = [0]
    PRE_ACCUM_NUM_CLASS.extend(ACCUM_NUM_CLASS[:-1])
    CUR_CLASS = [list(range(ACCUM_NUM_CLASS[0]))]
    for t_id in range(1,NUM_TASK):
        CUR_CLASS.append(list(range(ACCUM_NUM_CLASS[t_id-1],ACCUM_NUM_CLASS[t_id])))

    continual_config = {
        'NUM_TASK': NUM_TASK,
        'NUM_CLASS': NUM_CLASS,
        'LABEL_LIST': LABEL_LIST,
        'CLASSNAME_LIST': CLASSNAME_LIST,
        'CUR_NUM_CLASS': CUR_NUM_CLASS,
        'CUR_CLASS': CUR_CLASS,
        'ACCUM_NUM_CLASS': ACCUM_NUM_CLASS,
        'PRE_ACCUM_NUM_CLASS': PRE_ACCUM_NUM_CLASS,
        'label2idx': label2idx,
        'idx2label': idx2label,
    }

    generated_dataset_name = '%s_task%d_base%d_inc%d'%(dataset,NUM_TASK,base_task_entity,incremental_task_entity)
    tgt_path = './dataset/%s'%(generated_dataset_name)

    if not os.path.isdir(tgt_path):
        os.makedirs(tgt_path)

    # Step 3: Obtain all input and labels
    def load_ner_data(load_file):
        all_x = []
        all_y = []
        one_sentence_x = []
        one_sentence_y = []
        is_in_entity = False
        cur_entity_type = None
        with open(os.path.join(load_file), 'r') as f:
            for line in f.readlines():
                if line.strip() == '':
                    if len(one_sentence_x)>0:
                        all_x.append(one_sentence_x)
                        all_y.append(one_sentence_y)
                        one_sentence_x = []
                        one_sentence_y = []
                        is_in_entity = False
                        cur_entity_type = None
                    continue
                try:
                    token_list = line.strip().split('\t')
                    assert len(token_list) == 2
                except Exception as e:
                    print(e)
                    token_list = line.strip().split(' ')
                    assert len(token_list) == 2, 'Invalid line %s'%(line.strip())
                word, tag = token_list
                one_sentence_x.append(word)
                '''
                    [O] [O] [O] [B-PER] [I-PER]
                    His name is   Tim    Cook
                '''
                if tag.upper()=='O':
                    '''
                        [O] 
                        His 
                    '''
                    bio_tag = tag.upper()
                    is_in_entity = False
                    cur_entity_type = None
                elif tag.upper()!='O' and not is_in_entity:
                    '''
                        [B-PER]
                          Tim   
                    '''
                    bio_tag = 'B-'+tag.upper()
                    is_in_entity = True
                    cur_entity_type = tag.upper()
                elif tag.upper()!='O' and is_in_entity and tag.upper()==cur_entity_type:
                    '''
                        [B-PER] [I-PER]
                          Tim    Cook
                    '''
                    bio_tag = 'I-'+tag.upper()
                    is_in_entity = True
                    cur_entity_type = tag.upper()
                elif tag.upper()!='O' and is_in_entity and tag.upper()!=cur_entity_type:
                    '''
                        [B-PER] [B-GPE]
                          Tim    China
                    '''
                    bio_tag = 'B-'+tag.upper()
                    is_in_entity = True
                    cur_entity_type = tag.upper()
                one_sentence_y.append(bio_tag)
            if len(one_sentence_x)>0:
                all_x.append(one_sentence_x)
                all_y.append(one_sentence_y)
                one_sentence_x = []
                one_sentence_y = []
                is_in_entity = False
                cur_entity_type = None
        return all_x, all_y

    train_x, train_y = load_ner_data(train_file)
    dev_x, dev_y = load_ner_data(dev_file)
    test_x, test_y = load_ner_data(test_file)  

    # Step 4: Split the data into each tasks and adjust the label
    continual_data = {}
    for task_id in range(NUM_TASK):
        one_task_data = {
            'train':{
                'input':[],
                'label_idx_cil': [],
            },
            'dev':{
                'input':[],
                'label_idx_cil':[],
            },
            'test':{
                'input':[],
                'label_idx_cil':[],
            }
        }
        continual_data[task_id] = one_task_data

    # Step 4.1: For training data, we use Greedy Sampling Algorithm and mask the label of other tasks to 'O'.
    num_all_sent = len(train_x)      
    num_entity_task_list = [base_task_entity]
    for t_id in range(1,NUM_TASK):
        num_entity_task_list.append(incremental_task_entity)
    # how many sentence should be added
    sent_cnt_task = [int(num_all_sent*num_entity_task_list[t_id]/np.sum(num_entity_task_list)) for t_id in range(NUM_TASK)]
    # mapping entity to the task id
    entity2task = {}
    for idx, e in enumerate(alpha_sorted_entity_list):
        entity2task[e] = 1+(idx-base_task_entity)//incremental_task_entity if idx>=base_task_entity else 0
    # what labels can be seen in each tasks
    task2labels_train = {} # Only entities for each task is preserved, while other entities are masked.
    task2labels_test = {} # The seen entity is accumulated
    for t_id in range(NUM_TASK):
        task2labels_train[t_id] = []
        task2labels_train[t_id].append('O')
        task2labels_test[t_id] = []
        task2labels_test[t_id].append('O')
        if t_id == 0:
            for e in alpha_sorted_entity_list[:np.cumsum(num_entity_task_list)[0]]:
                task2labels_train[t_id].append('B-'+e)
                task2labels_train[t_id].append('I-'+e)
            for e in alpha_sorted_entity_list[:np.cumsum(num_entity_task_list)[0]]:
                task2labels_test[t_id].append('B-'+e)
                task2labels_test[t_id].append('I-'+e)
        else:
            for e in alpha_sorted_entity_list[np.cumsum(num_entity_task_list)[t_id-1]:np.cumsum(num_entity_task_list)[t_id]]:
                task2labels_train[t_id].append('B-'+e)
                task2labels_train[t_id].append('I-'+e)
            for e in alpha_sorted_entity_list[:np.cumsum(num_entity_task_list)[t_id]]:
                task2labels_test[t_id].append('B-'+e)
                task2labels_test[t_id].append('I-'+e)

    for x_sent, y_sent in zip(train_x, train_y):
        is_break = False
        for entity in freq_sorted_entity_list:
            t_id = entity2task[entity]
            if 'B-'+entity in y_sent and sent_cnt_task[t_id]>0:
                continual_data[t_id]['train']['input'].append(x_sent)
                if seen_all_labels:
                    continual_data[t_id]['train']['label_idx_cil'].append(y_sent)
                else:
                    masked_y_sent = [tag if tag in task2labels_train[t_id] else 'O' for tag in y_sent]
                    continual_data[t_id]['train']['label_idx_cil'].append(masked_y_sent)
                sent_cnt_task[t_id] -= 1
                is_break = True
                break
        if not is_break:
            not_full_split = np.where(np.array(sent_cnt_task)>0)[0]
            if len(not_full_split)>0:
                t_id = np.random.choice(not_full_split)
                continual_data[t_id]['train']['input'].append(x_sent)
                if seen_all_labels:
                    continual_data[t_id]['train']['label_idx_cil'].append(y_sent)
                else:
                    masked_y_sent = [tag if tag in task2labels_train[t_id] else 'O' for tag in y_sent]
                    continual_data[t_id]['train']['label_idx_cil'].append(masked_y_sent)
                sent_cnt_task[t_id] -= 1

    # Step 4.2: For dev or test data, we mask the label of unseen tasks to 'O'.
    for t_id in range(NUM_TASK):
        for x_sent, y_sent in zip(dev_x, dev_y):
            masked_y_sent = [tag if tag in task2labels_test[t_id] else 'O' for tag in y_sent]
            if list(set(masked_y_sent)) != ['O']:
                continual_data[t_id]['dev']['input'].append(x_sent)
                continual_data[t_id]['dev']['label_idx_cil'].append(masked_y_sent)

    for t_id in range(NUM_TASK):
        for x_sent, y_sent in zip(test_x, test_y):
            masked_y_sent = [tag if tag in task2labels_test[t_id] else 'O' for tag in y_sent]
            if list(set(masked_y_sent)) != ['O']:
                continual_data[t_id]['test']['input'].append(x_sent)
                continual_data[t_id]['test']['label_idx_cil'].append(masked_y_sent)


    with open(os.path.join(tgt_path,'continual_data.json'),'w') as f:
        json.dump(continual_data,f)

    with open(os.path.join(tgt_path,'continual_config.json'),'w') as f:
        json.dump(continual_config,f)

    print('Dataset %s is successful generated and saved into %s!'%(generated_dataset_name,tgt_path))


if __name__ == "__main__":
    params = get_params()
    set_seed(seed=params.seed)
    main(params)