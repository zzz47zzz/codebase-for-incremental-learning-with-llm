'''
This file is for evaluating the in-context learning performance of GPT3.5 and GPT4 on Concept-1K
'''

from openai import OpenAI
import json
import os
import random
import numpy as np

seed = 20240210
random.seed(seed)

client = OpenAI(api_key="")
role = "You're a helpful assistant."

# ============================================================================
prompt_1shot = "I will provide some knowledge as follows:\n \
Question: {0}\n Short Answer: {1}\n \
Please answer the following question according to the above knowledge:\n \
Question: {2}\n Short Answer: "

prompt_5shot = "I will provide some knowledge as follows:\n \
Question: {0}\n Short Answer: {1}\n \
Question: {2}\n Short Answer: {3}\n \
Question: {4}\n Short Answer: {5}\n \
Question: {6}\n Short Answer: {7}\n \
Question: {8}\n Short Answer: {9}\n \
Please answer the following question according to the above knowledge:\n \
Question: {10}\n Short Answer: "
# ============================================================================

n_subsample = 500

# model_name = 'gpt-3.5-turbo' # gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo 
# shot = 1
# method = 'same_concept' # same_instance, same_concept, rand

for model_name in ['gpt-3.5-turbo','gpt-4-turbo-preview']:
    for shot in [1,5]:
        for method in ['rand','same_concept']:
            if model_name == 'gpt-3.5-turbo' and shot == 1 and method=='rand':
                continue

            save_name = '%dshot_%s'%(shot,method)

            cur_dir = os.path.dirname(__file__)
            with open(os.path.join(cur_dir,'dataset/concept_1k_task1/continual_data.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            train_data = data['0']['train']
            test_data = data['0']['test']
            select_index = list(range(len(train_data['input'])))
            random.shuffle(select_index)
            select_index = select_index[:n_subsample]

            cnt = 0
            acc_list = []
            for test_sample_id in select_index:
                if shot == 1:
                    if method == 'same_instance':
                        prompted_test_input = prompt_1shot.format(
                            train_data['input'][test_sample_id],
                            train_data['target'][test_sample_id],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'same_concept':
                        same_concept_idx = np.where(np.array(train_data['concept_id'])==test_data['concept_id'][test_sample_id])[0]
                        random_index = np.random.choice(same_concept_idx,size=1)
                        prompted_test_input = prompt_1shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'rand':
                        random_index = np.random.choice(list(range(len(train_data['input']))),size=1)
                        prompted_test_input = prompt_1shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            test_data['input'][test_sample_id],
                        )
                    else:
                        raise NotImplementedError()
                elif shot==5:
                    if method == 'same_instance':
                        random_index = np.random.choice(list(range(len(train_data['input']))),size=4)
                        prompted_test_input = prompt_5shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            train_data['input'][random_index[1]],
                            train_data['target'][random_index[1]],
                            train_data['input'][random_index[2]],
                            train_data['target'][random_index[2]],
                            train_data['input'][random_index[3]],
                            train_data['target'][random_index[3]],
                            train_data['input'][test_sample_id],
                            train_data['target'][test_sample_id],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'same_concept':
                        same_concept_idx = np.where(np.array(train_data['concept_id'])==test_data['concept_id'][test_sample_id])[0]
                        random_index = np.random.choice(same_concept_idx,size=5)
                        prompted_test_input = prompt_5shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            train_data['input'][random_index[1]],
                            train_data['target'][random_index[1]],
                            train_data['input'][random_index[2]],
                            train_data['target'][random_index[2]],
                            train_data['input'][random_index[3]],
                            train_data['target'][random_index[3]],
                            train_data['input'][random_index[4]],
                            train_data['target'][random_index[4]],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'rand':
                        random_index = np.random.choice(list(range(len(train_data['input']))),size=5)
                        prompted_test_input = prompt_5shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            train_data['input'][random_index[1]],
                            train_data['target'][random_index[1]],
                            train_data['input'][random_index[2]],
                            train_data['target'][random_index[2]],
                            train_data['input'][random_index[3]],
                            train_data['target'][random_index[3]],
                            train_data['input'][random_index[4]],
                            train_data['target'][random_index[4]],
                            test_data['input'][test_sample_id],
                        )
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

                test_target = test_data['target'][test_sample_id]
                        
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": prompted_test_input}
                    ]
                )

                response = completion.choices[0].message.content
                response = response.strip().lower()
                test_target = test_target.strip().lower()
                acc_list.append(1.0 if response==test_target else 0.0)
                cnt += 1

                print('Progress %.2f%%[=%d/%d]: Test_ID=%s, Target=%s, Predict=%s, tmp_ACC=%.2f%%'%(
                    cnt/len(select_index)*100,
                    cnt,
                    len(select_index),
                    test_sample_id,
                    test_target,
                    response,
                    np.mean(acc_list)*100,
                ))

                with open(os.path.join(cur_dir,'./seed%d_%s_%s.txt'%(seed,model_name,save_name)), 'a') as file:
                    file.write('Progress %.2f%%[=%d/%d]: Test_ID=%s, Target=%s, Predict=%s, tmp_ACC=%.2f%%\n'%(
                    cnt/len(select_index)*100,
                    cnt,
                    len(select_index),
                    test_sample_id,
                    test_target,
                    response,
                    np.mean(acc_list)*100,
                ))


