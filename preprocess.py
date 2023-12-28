from datasets import load_dataset
import json
import numpy as np
from pathlib import Path
import re
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


with open('./paper_data/labels.json') as f:
    labels = json.load(f)
    qst2idx = {}
    idx2qst = {}
    question2autoannotation = {}
    for i, l in enumerate(labels.values()):
        qst2idx[l['question']] = i
        idx2qst[i] = l['question']
        if 'annotate_first' in l :
            question2autoannotation[l['question']] = l['annotate_first']
        else:
            question2autoannotation[l['question']] = False


def get_question_matrix(squad_dict):
    qst_matrix = np.zeros(shape=(len(squad_dict['data']), len(qst2idx)))
    for i, dct in enumerate(squad_dict['data']):
        policy = dct['paragraphs'][0]
        for qa in policy['qas']:
            question = qa['question']
            answers = qa['answers']
            question_idx = qst2idx[question]
            for a in answers['text']:
                qst_matrix[i, question_idx] += 1
    qst_weights = qst_matrix.sum(axis=0) / qst_matrix.sum()
    return qst_matrix, qst_weights


def autoannotate(squad_dict):
    '''
    The annotators received the instruction to annotate some labels only once (e.g. public name).
    Here, we get these annoattion and tag each occurence.
    '''
    autoannotate = {}
    autoannotations = 0
    for i, policy in enumerate(squad_dict['data']):
        autoannotate[i] = {}
        for paragraph in policy['paragraphs']:
            for qas in paragraph['qas']:
                question = qas['question']
                if question2autoannotation[question] == True:
                    if question not in autoannotate[i]:
                        autoannotate[i][question] = set()
                    for text in qas['answers']['text']:
                        autoannotate[i][question].add(text)
    for i, policy in enumerate(squad_dict['data']):
        for j, paragraph in enumerate(policy['paragraphs']):
            context = paragraph['context']
            for k, qas in enumerate(paragraph['qas']):
                question = qas['question']
                if question in autoannotate[i]:
                    for text in autoannotate[i][question]:
                         for occurence in re.finditer(text, context):
                            start = occurence.start()
                            if text[-1] == '.' and ('subsidiaries' in text or 'Groupon International Limited' in text or 'Airbnb Payments' in text or 'Ltd' in text or 'tv.' in text):
                                text = text[:-1]
                            if start not in squad_dict['data'][i]['paragraphs'][j]['qas'][k]['answers']['answer_start']:
                                squad_dict['data'][i]['paragraphs'][j]['qas'][k]['answers']['text'].append(text)
                                squad_dict['data'][i]['paragraphs'][j]['qas'][k]['answers']['answer_start'].append(start)
                                autoannotations += 1
    return autoannotations

def read_squad(path, annotations_count, mode='train'):
    '''
    Adapted from https://huggingface.co/transformers/v4.11.3/custom_datasets.html#qa-squad
    '''
    path_name = path.split('/')[-1].split('.')[0]
    target_file_name = f'{path_name}_formatted.json'
    target_path = Path(f"./data/{target_file_name}")
    path = Path(path)
    if target_path.is_file():
        raise Exception(f'{target_file_name} already exists.')
    with open(path, 'r', encoding='utf8') as f:
        squad_dict = json.load(f)
    # count the number of annotations
    check_count = 0
    for policy in squad_dict['data']:
        for paragraph in policy['paragraphs']:
            for qas in paragraph['qas']:
                for _ in qas['answers']['answer_start']:
                    check_count += 1
    # get weights of questions
    labels_matrix, labels_weights = get_question_matrix(squad_dict)
    # get validation set
    if mode == 'train':
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        for train_index, val_index in msss.split(list(range(len(squad_dict['data']))), labels_matrix): 
            train_idx = list(train_index)
            val_idx = list(val_index)
            break
            
    check_autoannotations = autoannotate(squad_dict)
    check_recount = 0
    check_special_char = 0
    ids_main = []
    contexts_main = []
    questions_main = []
    answers_main = []
    is_impossible_main = []
    ids_second = []
    contexts_second = []
    questions_second = []
    answers_second = []
    is_impossible_second = []
    for i, dct in enumerate(squad_dict['data']):
        if mode == 'train' and i in val_idx:
            ids = ids_second
            contexts = contexts_second
            questions = questions_second
            answers = answers_second
            is_impossible = is_impossible_second
        else:
            ids = ids_main
            contexts = contexts_main
            questions = questions_main
            answers = answers_main
            is_impossible = is_impossible_main
        title = dct['title']
        policy = dct['paragraphs'][0]
        context = policy['context']
        paragraph_offset = 0
        # split in paragraphs
        for j, paragraph in enumerate(context.split('\n')):
            has_question = False
            for k, qa in enumerate(policy['qas']):
                question = qa['question']
                answer = {'text': [], 'answer_start': []}
                for l, (text, answer_start) in enumerate(zip(qa['answers']['text'], qa['answers']['answer_start'])):
                    if paragraph_offset <= answer_start < answer_start + len(text) <= paragraph_offset + len(paragraph):
                        paragraph_start = answer_start - paragraph_offset
                        paragraph_end = paragraph_start + len(text)
                        try:
                            assert paragraph[paragraph_start:paragraph_end] == text # or '14 417 56 Gothenburg' in text   # special char
                        except:
                            check_special_char +=1
                        answer['text'].append(text)
                        answer['answer_start'].append(paragraph_start)
                check_recount += len(answer['text'])
                if len(answer['text']):
                    has_question = True
                    if mode == 'test':
                        ids.append(f'{i}-{j}-{k}')
                        contexts.append(paragraph)
                        questions.append(question)
                        answers.append(answer)
                        is_impossible.append(False)
                    else:
                        for l, (text, answer_start) in enumerate(zip(answer['text'], answer['answer_start'])):
                            ids.append(f'{i}-{j}-{k}-{l}')
                            contexts.append(paragraph)
                            questions.append(question)
                            answers.append({'text': [text], 'answer_start': [answer_start]})
                            is_impossible.append(False)
            if not has_question:
                # we generate a random impossible question
                label_idx = np.random.choice(list(range(len(labels_weights))), 1, p=labels_weights, replace=True)[0]
                ids.append(f'{i}-{j}-impossible')
                contexts.append(paragraph)
                questions.append(idx2qst[label_idx])
                answers.append({'text': [], 'answer_start': []})
                is_impossible.append(True)
            paragraph_offset += len(paragraph) + len('\n')
    assert annotations_count == check_count
    assert check_count + check_autoannotations == check_recount + check_special_char
    with open(target_path, 'w', encoding='utf8') as f:
        json.dump({'version': 'v1.0','data': {
            'id': ids_main,
            'context': contexts_main,
            'question': questions_main,
            'answers': answers_main,
        }}, f, ensure_ascii=False)
    if mode == 'train' and len(ids_second):
        with open('./data/GDPR120Q_validation_formatted.json', 'w', encoding='utf8') as f:
            json.dump({'version': 'v1.0','data': {
                'id': ids_second,
                'context': contexts_second,
                'question': questions_second,
                'answers': answers_second,
            }}, f, ensure_ascii=False)


read_squad('paper_data/GDPR120Q_train.json', annotations_count=31417)
read_squad('paper_data/GDPR120Q_test.json', mode='test', annotations_count=8417)
dataset = load_dataset('json', data_files={'train': './data/GDPR120Q_train_formatted.json', 'validation': './data/GDPR120Q_validation_formatted.json', 'test': './data/GDPR120Q_test_formatted.json'}, field='data')
