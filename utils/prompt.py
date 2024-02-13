
Dataset2Prompt = {
    'concept_1k_task1': 'qa',
    'concept_1k_task10': 'qa',
    'topic3datasets_task5': 'default',
    'clinc150_task15': 'default',
    'clinc150_task15_numbertarget': 'default',
    'clinc150_task15_shuffle_classidx': 'default',
    'banking77_task7': 'default',
    'fewrel_task8': 'relation_classification_state',
    'tacred_task8': 'relation_classification_state',
}

Dataset2PromptTuningInitText = {
    'concept_1k_task1': 'Answer the question below:',
    'concept_1k_task10': 'Answer the question below:',
    'topic3datasets_task5': 'Classify the topic of this sentence:',
    'clinc150_task15': 'Classify the intent of this sentence:',
    'clinc150_task15_numbertarget': 'Classify the intent of this sentence:',
    'clinc150_task15_shuffle_classidx': 'Classify the intent of this sentence:',
    'banking77_task7': 'Classify the intent of this sentence:',
    'fewrel_task8': 'Classify the relationship in this sentence:',
    'tacred_task8': 'Classify the relationship in this sentence:',
}

def get_prompt(text, label, prompt_type, eos_token='', dataset='clinc150_task15'):

    if prompt_type=='auto':
        assert dataset in Dataset2Prompt.keys(), 'NotDefined for dataset %s when prompt_type==auto'%(dataset)
        prompt_type = Dataset2Prompt[dataset]

    if prompt_type=='none':
        if label is None:
            return text
        else:
            return text+' '+label

    if prompt_type=='default':
        if label is None:
            return f'Input sentence: {text}\n The label:'
        else:
            return f'Input sentence: {text}\n The label: {label}{eos_token}'
        
    if prompt_type=='qa':
        if label is None:
            return f'Question: {text}\n Short Answer:'
        else:
            return f'Question: {text}\n Short Answer: {label}{eos_token}'
        
    if prompt_type in ['relation_classification_qa','relation_classification_state','relation_classification_state_no_pos']:
        entity_first_bg, entity_first_ed = text.find('[E11]'), text.find('[E12]')
        entity_second_bg, entity_second_ed = text.find('[E21]'), text.find('[E22]')
        assert entity_first_bg!=-1 and entity_first_ed!=-1 and entity_second_bg!=-1 and entity_second_ed!=-1
        entity_first = text[entity_first_bg+len('[E11]'):entity_first_ed].strip()
        entity_second = text[entity_second_bg+len('[E21]'):entity_second_ed].strip()
        # remove the token [E11], [E12], [E21], [E22]
        text = text.replace(' [E11] ',' ')
        text = text.replace(' [E12] ',' ')
        text = text.replace(' [E21] ',' ')
        text = text.replace(' [E22] ',' ')
        text = text.replace('[E11] ','')
        text = text.replace('[E12] ','')
        text = text.replace('[E21] ','')
        text = text.replace('[E22] ','')
        text = text.replace(' [E11]','')
        text = text.replace(' [E12]','')
        text = text.replace(' [E21]','')
        text = text.replace(' [E22]','')

        if prompt_type == 'relation_classification_qa':

            if label is None:
                return f'Input sentence: {text}\n Question: What is the relationship between {entity_first} and {entity_second}?\n Answer:'
            else:
                return f'Input sentence: {text}\n Question: What is the relationship between {entity_first} and {entity_second}?\n Answer: {label}{eos_token}'
    
        if prompt_type == 'relation_classification_state':

            if label is None:
                return f'Input sentence: {text}\n The relationship between {entity_first} and {entity_second} is'
            else:
                return f'Input sentence: {text}\n The relationship between {entity_first} and {entity_second} is {label}{eos_token}'
    
        if prompt_type == 'relation_classification_state_no_pos':

            if label is None:
                return f'Input sentence: {text}\n The relationship:'
            else:
                return f'Input sentence: {text}\n The relationship: {label}{eos_token}'

    raise NotImplementedError('Not Implemented prompt_type %s'%(prompt_type))


def get_prompt_LAMOL(text, label, gen_token='__gen__', ans_token='__ans__', eos_token=''):
    # TODO: No space before and after answertoken in LAMOL
    if label is None:
        return f'{gen_token}{text}{ans_token}'
    else:
        return f'{gen_token}{text}{ans_token}{label}{eos_token}'

def get_prompt_PCLL(text, label, gen_token, eos_token, task_id=None):
    if task_id is None:
        return_list = []
        return_list.append(f'{gen_token}In the current task, which intent category best describes: "{eos_token}')
        return_list.append(f'{gen_token}In the current task, which intent category best describes: " {text}{eos_token}')
        return_list.append(f'{gen_token}In the current task, which intent category best describes: " {text} "? Answer:')
        return_list.append(f'{gen_token}In the current task, which intent category best describes: " {text} "? Answer: {label}{eos_token}')
        return return_list
    else:
        assert isinstance(task_id,int)
        return_list = []
        return_list.append(f'{gen_token}In the {task_id}-th task, which intent category best describes: "{eos_token}')
        return_list.append(f'{gen_token}In the {task_id}-th task, which intent category best describes: " {text}{eos_token}')
        return_list.append(f'{gen_token}In the {task_id}-th task, which intent category best describes: " {text} "? Answer:')
        return_list.append(f'{gen_token}In the {task_id}-th task, which intent category best describes: " {text} "? Answer: {label}{eos_token}')
        return return_list

def get_auto_prompt_tuning_init_text(dataset='clinc150_task15'):

    assert dataset in Dataset2PromptTuningInitText.keys(), 'NotDefined for dataset %s when params.PEFT_prompt_tuning_init_text==auto'%(dataset)
    prompt_tuning_init_text = Dataset2PromptTuningInitText[dataset]

    return prompt_tuning_init_text

def get_prompt_ICL(train_text_list, eval_text, prefix=''):
    '''
        Obtain prompt for inconext learning
    '''
    for train_text in train_text_list:
        prefix += f'{train_text} \n'
    prefix += f'{eval_text}'
    return prefix