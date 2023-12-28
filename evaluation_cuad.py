import os
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from transformers import BertTokenizer
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IOU_THRESH = 0.5
J_TRESH = 0

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_json_path = "./data/GDPR120Q_test_formatted.json"

with open('./paper_data/labels.json') as f:
    labels = json.load(f)
    qst2label = {}
    label2qst = {}
    for l in labels.values():
        lst = l['label']
        qst = l['question']
        label2qst[lst[0]] = qst
        qst2label[qst] = lst


# dataset
with open(test_json_path, "r") as f2:
    ds = json.load(f2)['data']
    
qidBycat = {}
qidByQst = {}
for qid, qText in zip(ds['id'], ds['question']):
    label = qst2label[qText][0]
    if label not in qidBycat:
        qidBycat[label] = []
    qidBycat[label].append(qid)
    if qText not in qidByQst:
        qidByQst[qText] = []
    qidByQst[qText].append(qid)
    

def load_json(path):
    with open(path, "r") as f:
        dct = json.load(f)
    return dct


def get_preds(nbest_preds_dict, conf=None):
    results = {}
    for question_id in nbest_preds_dict:
        list_of_pred_dicts = nbest_preds_dict[question_id]
        preds = {}
        for pred_dict in list_of_pred_dicts:
            text = pred_dict["text"]
            prob = pred_dict["probability"]
            if not text == "":  # don't count empty string as a prediction
                preds[text] = prob
        preds_list = [pred for pred in preds.keys() if preds[pred] > conf]
        results[question_id] = preds_list
    return results


def get_answers(test_json_dict):
    results = {}
    data = test_json_dict["data"]
    for id, anws in zip(data['id'], data['answers']):
        answers = [text for text in anws['text']]
        results[id] = answers
    return results

    
def get_answers_hugging_face(test_json_dict):
    results = {}
    for id, anws in zip(test_json_dict['id'], test_json_dict['answers']):
        answers = [text for text in anws['text']]
        results[id] = answers
    return results


def get_jaccard(gt, pred):
    remove_tokens = [".", ",", ";", ":"]
    for token in remove_tokens:
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower()
    pred = pred.lower()
    gt = gt.replace("/", " ")
    pred = pred.replace("/", " ")
    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))
    intersection = gt_words.intersection(pred_words)
    union = gt_words.union(pred_words)
    jaccard = len(intersection) / len(union)
    return jaccard


def compute_precision_recall(gt_dict, preds_dict, category=None):
    tp, fp, fn = 0, 0, 0
    for key in gt_dict:
        if category and category not in key:
            continue
        
        answers = gt_dict[key]
        preds = preds_dict[key]
        # first check if answers is empty
        if len(answers) == 0:
            if len(preds) > 0:
                fp += len(preds)  # false positive for each one
        else:
            for ans in answers:
                assert len(ans) > 0
                # check if there is a match
                match_found = False
                for pred in preds:
                    j = get_jaccard(ans, pred)
                    is_match = j >= IOU_THRESH
                    if is_match:
                        match_found = True

                if match_found:
                    tp += 1
                else:
                    fn += 1
            # now also get any fps by looping through preds
            for pred in preds:
                # Check if there's a match. if so, don't count (don't want to double count based on the above)
                # but if there's no match, then this is a false positive.
                # (Note: we get the true positives in the above loop instead of this loop so that we don't double count
                # multiple predictions that are matched with the same answer.)
                match_found = False
                for ans in answers:
                    assert len(ans) > 0
                    j = get_jaccard(ans, pred)
                    is_match = j >= IOU_THRESH
                    if is_match:
                        match_found = True

                if not match_found:
                    fp += 1
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    return precision, recall


def process_precisions(precision):
    """
    Processes precisions to ensure that precision and recall don't both get worse
    Assumes the list precision is sorted in order of recalls
    """
    precision_best = precision[::-1]
    for i in range(1, len(precision_best)):
        precision_best[i] = max(precision_best[i-1], precision_best[i])
    precision = precision_best[::-1]
    return precision


def get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.9):
    """
    Assumes recalls are sorted in increasing order
    """
    processed_precisions = process_precisions(precisions)
    prec_at_recall = 0
    for prec, recall, conf in zip(processed_precisions, recalls, confs):
        if recall >= recall_thresh:
            prec_at_recall = prec
            break
    return prec_at_recall, conf


def get_precisions_recalls(pred_dict, gt_dict, category=None):
    precisions = [1]
    recalls = [0]
    confs = []
    f1s = []
    for conf in list(np.arange(0.99, 0, -0.01)) + [0.001, 0]:
        conf_thresh_pred_dict = get_preds(pred_dict, conf)
        prec, recall = compute_precision_recall(gt_dict, conf_thresh_pred_dict, category=category)
        f1 = (2 * (prec * recall)) / (prec + recall) if prec + recall > 0 else 0.0
        precisions.append(prec)
        recalls.append(recall)
        confs.append(conf)
        f1s.append(f1)
    return precisions, recalls, confs, f1s


def get_aupr(precisions, recalls):
    processed_precisions = process_precisions(precisions)
    aupr = metrics.auc(recalls, processed_precisions)
    if np.isnan(aupr):
        return 0
    return aupr


def get_results(model_path, gt_dict, pred_dict, verbose=False):
    assert sorted(list(pred_dict.keys())) == sorted(list(gt_dict.keys()))
    precisions, recalls, confs, f1s = get_precisions_recalls(pred_dict, gt_dict)
    prec_at_90_recall, _ = get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.9)
    prec_at_80_recall, _ = get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.8)
    aupr = get_aupr(precisions, recalls)

    if verbose:
        print("AUPR: {:.3f}, Precision at 80% Recall: {:.3f}, Precision at 90% Recall: {:.3f}".format(aupr, prec_at_80_recall, prec_at_90_recall))

    # now save results as a dataframe and return
    results = {"name": name, "aupr": aupr, "prec_at_80_recall": prec_at_80_recall, "prec_at_90_recall": prec_at_90_recall}
    return results, precisions, recalls, f1s, confs


def get_stats(path_dataset):
    with open(path_dataset, 'r') as f:
        squad_dict = json.load(f)
    stats = {'byPolicy': {}, 'questionOccurence': {}, 'len_token': []}
    for policy in squad_dict['data']:
        title = policy['title']
        policy_len_tokens = 0
        for paragraph in policy['paragraphs']:
            context = paragraph['context']
            tokens = len(tokenizer.tokenize(context))
            policy_len_tokens += tokens
            annotation_coverage = [False] * len(context)
            for qas in paragraph['qas']:
                question = qas['question']
                if question not in stats['questionOccurence']:
                    stats['questionOccurence'][question] = 0
                for s, t in zip(qas['answers']['answer_start'], qas['answers']['text']):
                    stats['questionOccurence'][question] += 1
                    for i in range(s, s+len(t)):
                        annotation_coverage[i] = True
        stats['len_token'].append(policy_len_tokens)
        stats['byPolicy'][title] = {
            'annotation_coverage': annotation_coverage,
            'tokens': tokens,
        }
    return stats


def plot_pr_curve(precisions_recalls, out_image, title="GDPR120Q Precision Recall Curve", colors=[], labels=[]):
    for i, (precisions, recalls) in enumerate(precisions_recalls):
        plt.step(recalls, precisions, color=colors[i], alpha=1)
    plt.gca().legend(labels + ['80% Recall'], loc='upper right')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.axvline(x=0.8, linestyle='--', color='gray')
    plt.title(title)
    plt.savefig(out_image, format='eps', bbox_inches='tight')
    plt.clf()


def get_05_index(confs):
    for i, c in enumerate(reversed(confs)):
        if c > 0.5:
            return i

def get_threshold_f1_at_recall(precisions, recalls, confs, f1s, target_recall):
    for prec, rec, conf, f1 in zip(precisions, recalls, confs, f1s):
        if rec >= target_recall:
            return conf, f1
    return None, None


longtable_start = r'''\documentclass{article}
\usepackage{supertabular}
\usepackage{array}
\usepackage{makecell}
\usepackage[T1]{fontenc}
\begin{document}
\tablefirsthead{
    \hline
    \textbf{Category} & \textbf{\#train} & \textbf{\#test} & \textbf{aupr} & \textbf{p80r} & \textbf{f1 (0.5)} & \textbf{F1 GPT} \\
    \hline
}
\tablehead{
    \hline
    \multicolumn{7}{l}{\small\sl continued from previous page}\\
    \hline
    \textbf{Category} & \textbf{\#train} & \textbf{\#test} & \textbf{aupr} & \textbf{p80r} & \textbf{f1 (0.5)} & \textbf{F1 GPT} \\
    \hline
}
\tabletail{
    \hline
    \multicolumn{7}{r}{\small\sl continued on next page}\\
    \hline
}
\tablelasttail{\hline}
\bottomcaption{The labels, questions and results for GDPR120Q. P80R stands for Precision at 80 Recall}
\begin{supertabular}{|m{0.3\textwidth}|m{0.075\textwidth}|m{0.075\textwidth}|m{0.075\textwidth}|m{0.075\textwidth}|m{0.075\textwidth}|m{0.075\textwidth}|}
'''

longtable_end = r'\end{supertabular}\end{document}'

if __name__ == "__main__":
    model_path = "./models/deberta-v3-large"
    save_dir = "./results_cuad/deberta-v3-large"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    gt_dict = load_json(test_json_path)
    gt_dict = get_answers(gt_dict)
    
    # preds
    predictions_path = os.path.join(model_path, "predict_nbest_predictions.json")
    name = model_path.split("/")[-1]
    pred_dict = load_json(predictions_path)
    
    # all cats
    results_main, precisions_main, recalls_main, f1s_main, confs_main = get_results(model_path, gt_dict, pred_dict, verbose=True) 
    
    save_path = os.path.join(save_dir, "{}.json".format(model_path.split("/")[-1]))
    with open(save_path, "w") as f:
        f.write("{}\n".format(results_main))
	
	
    # benchmarks
    # compare models
    models_comparison = [
        {
            'model_path': "./models/bert-base-uncased",
            'save_dir': "./results_cuad/bert-base-uncased-b",
            'color': '#283593',
            'label': 'GDPR120Q',
        },
        {
            'model_path': "./models/policy_qa_predict",
            'save_dir': "./results_cuad/policy_qa-b",
            'color': '#616161',
            'label': 'PolicyQA on GDPR120Q',
        },
        {
            'model_path': "./models/cuad-predict",
            'save_dir': "./results_cuad/roberta-base-cuad-b",
            'color': '#B71C1C',
            'label': 'CUAD on GDPR120Q',
        },
        
    ]
    models_comparison_plotted = ['./models/bert-base-uncased', './models/policy_qa_predict', './models/cuad-predict'] 
    precisions_recalls = []
    colors = []
    plot_labels = []
    precByModel = {}
    auprs = {}
    prec_at_80_recall = {}
    prec_at_90_recall = {}
    for m_dict in models_comparison:
        predictions_path = os.path.join(m_dict['model_path'], "predict_nbest_predictions.json")
        name = m_dict['model_path'].split("/")[-1]
        pred_dict = load_json(predictions_path)
        ground_truth = gt_dict
        if 'dataset' in m_dict:
            custom_dataset = load_dataset(m_dict['dataset'], split='test')
            custom_gt_dict = get_answers_hugging_face(custom_dataset)
            ground_truth = custom_gt_dict
        results, precisions, recalls, _, _ = get_results(m_dict['model_path'], ground_truth, pred_dict, verbose=True)
        if m_dict['model_path'] in models_comparison_plotted:
            precisions_recalls.append((precisions, recalls))
            colors.append(m_dict['color'])
            plot_labels.append(m_dict['label'])
        auprs[name] = results['aupr']
        prec_at_80_recall[name] = results['prec_at_80_recall']
        prec_at_90_recall[name] = results['prec_at_90_recall']
    # plot
    plot_pr_curve(precisions_recalls, './material/models_benchmark.eps', 'PR curve - task transferability', colors=colors, labels=plot_labels)
    
    
    # general stats
    stats = get_stats('paper_data/GDPR120Q_all.json')
    annotation_coverage = [boolean for c in stats['byPolicy'].values() for boolean in c['annotation_coverage']]
    annotation_coverage = sum([b for b in annotation_coverage if b is True]) / len(annotation_coverage)
    print('Annotation coverage', f'{annotation_coverage:.20f}')
    # train stats
    stats_train = get_stats('paper_data/GDPR120Q_train.json')
    qst_occurences_train = stats_train['questionOccurence']
    stats_test = get_stats('paper_data/GDPR120Q_test.json')
    qst_occurences_test = stats_test['questionOccurence']
    
    # plot policies length
    plt.bar([i for i in range(len(stats['len_token']))], sorted(stats['len_token']), color='#283593', width=0.4)
    plt.xlabel("Each bar represents one of the 120 policies")
    plt.ylabel("Number of tokens")
    plt.title("Lenght of the policies in terms of tokens (BERT tokenizer)")
    plt.xlim([0, 120])
    plt.savefig('./material/policies_len.eps', format='eps', bbox_inches='tight')
    plt.clf()
    
    # GPT
    with open('./GPT/gpt_answers.json') as f:
       gpt_pred_dict = json.load(f)
    
    prec_gpt, recall_gpt = compute_precision_recall(gt_dict, gpt_pred_dict)
    f1_gpt = (2 * (prec_gpt * recall_gpt)) / (prec_gpt + recall_gpt)
    
    # BERT pretrained
    results_pretrained, precisions_pretrained, recalls_pretrained, f1s_pretrained, confs_pretrained = get_results("./models/bert-base-pretraining", gt_dict, load_json('./models/bert-base-pretraining/predict_nbest_predictions.json'), verbose=False) 
    
    # plot f1 curve for comparing deberta and GPT
    plt.plot(confs_main, f1s_main, color='#283593', alpha=0.5)
    plt.plot(confs_main, f1s_pretrained, color='#B71C1C', alpha=0.5)
    plt.xlabel("Probability threshold")
    plt.ylabel("F1 score")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.axhline(y=f1_gpt, linestyle='--', color='#616161')
    plt.title("F1 score (micro) as a function of probability thresholds")
    plt.gca().legend(['DeBERTA (l.)', 'BERT (pre.)', f'GPT ({f1_gpt:.2f})'], loc='upper right')
    plt.savefig('./material/F1.eps', format='eps', bbox_inches='tight')
    plt.clf()
    
    # by cat
    auprByCat = {}
    for label in label2qst:
         _gt_dict = {qid: v for qid, v in gt_dict.items() if qid in qidBycat[label]}
         _pred_dict = {qid: v for qid, v in pred_dict.items() if qid in qidBycat[label]}
         results, precisions, recalls, f1s, confs = get_results(model_path, _gt_dict, _pred_dict, verbose=False)
         l = label
         if 'Transfer to' in l:
             l = 'Transfers to third countries'
         elif 'Automated' in l:
             l = 'Automated Decision-Making'
         auprByCat[l] = results['aupr']
    # auprs
    plt.barh(list(reversed(auprByCat.keys())), list(reversed(auprByCat.values())), color='#283593')
    plt.title("AUPR by category (DeBERTa l.)")
    plt.tight_layout()
    plt.savefig('./material/auprs_by_cat.eps', format='eps', bbox_inches='tight')
    plt.clf()
    
    total_train = 0
    total_test = 0
    
    file_content = []
    for question in [v['question'] for k, v in labels.items()]:
         if question not in qidByQst:
             continue
         question_latex = question.replace('\'', r'\textquotesingle')
         _gt_dict = {qid: v for qid, v in gt_dict.items() if qid in qidByQst[question]}
         _pred_dict = {qid: v for qid, v in pred_dict.items() if qid in qidByQst[question]}
         _gpt_pred_dict = {qid: v for qid, v in gpt_pred_dict.items() if qid in qidByQst[question]}
         _results, _, _, _f1s, _confs = get_results(model_path, _gt_dict, _pred_dict, verbose=False)
         aupr = _results['aupr']
         p80r = _results['prec_at_80_recall']
         _index_0_5 = get_05_index(_confs)
         _f1_at_0_5 = _f1s[_index_0_5]
         _prec_gpt, _recall_gpt = compute_precision_recall(_gt_dict, _gpt_pred_dict)
         _f1_gpt = (2 * (_prec_gpt * _recall_gpt)) / (_prec_gpt + _recall_gpt) if _prec_gpt + _recall_gpt > 0 else 0.0
         qst_occurence_train = stats_train['questionOccurence'][question] if question in stats_train['questionOccurence'] else 0
         qst_occurence_test = stats_test['questionOccurence'][question] if question in stats_test['questionOccurence'] else 0
         if qst_occurence_test >= 5:
             total_train += qst_occurence_train
             total_test += qst_occurence_test
             cats = qst2label[question]
             if 'Transfer' in cats[0]:
                 if 'Contractual clauses' in cats[-1]:
                     from textwrap import wrap
                     file_content.append(r'\multicolumn{7}{|l|}{\makecell{' + r' > \\'.join(cats[:-1]) + r' >\\' + r'\\'.join(list(wrap(cats[-1], 75))) + r'}}\\\hline')
                 else:
                     file_content.append(r'\multicolumn{7}{|l|}{\makecell{' + r' > \\'.join(cats) + r'}}\\\hline')
             else:
                 file_content.append(r'\multicolumn{7}{|l|}{' + ' > '.join(cats) + r'}\\\hline')
             file_content.append(f'{question_latex} & {qst_occurence_train} & {qst_occurence_test} & {aupr:.2f} & {p80r:.2f} & {_f1_at_0_5:.2f} & {_f1_gpt:.2f}' + r'\\\hline')
    with open('./material/longtable.tex', 'w') as fw:
        fw.write(longtable_start)
        for line in file_content:
            fw.write(f"{line}\n")
        fw.write(longtable_end)
    
    # compare data size using deberta
    models_size = ["./models/deberta-v3-base-1000", "./models/deberta-v3-base-10000", "./models/deberta-v3-base"]
    
    precisions_recalls = []
    
    x = [1000, 10000, 40000]
    y = []
    for m_path in models_size:
        predictions_path = os.path.join(m_path, "predict_nbest_predictions.json")
        name = model_path.split("/")[-1]
        pred_dict = load_json(predictions_path)
        results, _, _, _, _ = get_results(m_path, gt_dict, pred_dict, verbose=False)
        aupr = results['aupr']
        y.append(aupr)
    plt.plot(x, y, 'o-', color='#283593')
    plt.xlabel("Number of Training Annotations")
    plt.ylabel("AUPR")
    plt.title("GDPR120Q Performance vs. Dataset Size")
    plt.savefig("./material/GDPR120Q_performance_vs_dataset_size.eps", format='eps', bbox_inches='tight')
    plt.clf()
    
    # compare pretraining
    models_pretraining = [
        {
            'model_path': "./models/bert-base-uncased",
            'save_dir': "./results_cuad/bert-base-uncased",
            'color': '#616161',
            'label': 'BERT trained on GDPR120Q\'s 120 policies',
        },
        {
            'model_path': "./models/bert-base-pretraining",
            'save_dir': "./results_cuad/bert-base-pretraining",
            'color': '#B71C1C',
            'label': 'BERT pretrained on 130.000 policies',
        },
        {
            'model_path': "./models/bert-base-uncased-training-pretraining",
            'save_dir': "./results_cuad/bert-base-uncased-training-pretraining",
            'color': '#283593',
            'label': 'BERT pretrained + trained',
        }
    ]
    precisions_recalls = []
    for m_dict in models_pretraining:
        predictions_path = os.path.join(m_dict['model_path'], "predict_nbest_predictions.json")
        name = m_dict['model_path'].split("/")[-1]
        pred_dict = load_json(predictions_path)
        _, precisions, recalls, _, _ = get_results(m_dict['model_path'], gt_dict, pred_dict, verbose=False)
        precisions_recalls.append((precisions, recalls))
    plot_pr_curve(precisions_recalls, './material/pretraining.eps', 'PR curve - Impact of pretraining', colors=[m['color'] for m in (models_pretraining)], labels=[m['label'] for m in (models_pretraining)])
    
    # compare models
    models_comparison = [
        {
            'model_path': "./models/bert-base-uncased",
            'save_dir': "./results_cuad/bert-base-uncased",
            'color': '#616161',
            'label': 'BERT',
        },
        {
            'model_path': "./models/legal-bert-base-uncased",
            'save_dir': "./results_cuad/legal-bert-base-uncased",
            'color': '#B71C1C',
            'label': 'Legal-BERT',
        },
        {
            'model_path': "./models/roberta-base",
            'save_dir': "./results_cuad/roberta-base",
            'color': 'red',
            'label': 'RoBERTa',
        },
        {
            'model_path': "./models/legal-roberta-base",
            'save_dir': "./results_cuad/legal-roberta-base",
            'color': '#616161',
            'label': 'Legal-RoBERTa',
        },
        {
            'model_path': "./models/deberta-v3-base",
            'save_dir': "./results_cuad/deberta-v3-base",
            'color': '#283593',
            'label': 'DeBERTa',
        },
        {
            'model_path': "./models/deberta-v3-large",
            'save_dir': "./results_cuad/deberta-v3-large",
            'color': '#283593',
            'label': 'DeBERTa (large)',
        },
    ]
    models_comparison_plotted = ['./models/bert-base-uncased', './models/legal-bert-base-uncased', './models/deberta-v3-large'] 
    precisions_recalls = []
    colors = []
    plot_labels = []
    precByModel = {}
    auprs = {}
    prec_at_80_recall = {}
    prec_at_90_recall = {}
    for m_dict in models_comparison:
        predictions_path = os.path.join(m_dict['model_path'], "predict_nbest_predictions.json")
        name = m_dict['model_path'].split("/")[-1]
        pred_dict = load_json(predictions_path)
        results, precisions, recalls, _, _ = get_results(m_dict['model_path'], gt_dict, pred_dict, verbose=False)
        if m_dict['model_path'] in models_comparison_plotted:
            precisions_recalls.append((precisions, recalls))
            colors.append(m_dict['color'])
            plot_labels.append(m_dict['label'])
        auprs[name] = results['aupr']
        prec_at_80_recall[name] = results['prec_at_80_recall']
        prec_at_90_recall[name] = results['prec_at_90_recall']
    # plot
    plot_pr_curve(precisions_recalls, './material/models_comparison.eps', 'PR curve - models comparison', colors=colors, labels=plot_labels)
    # table comparison
    prec_table_start = r'''\documentclass{article}
    \usepackage{tabularx}
    \begin{document}
    \begin{table}[h]
    \centering
    \begin{tabular}{|c|c|c|c|}
    \hline
    \textbf{Model} & \textbf{AUPR} & \textbf{Prec. at 80\% rec.} & \textbf{Prec. at 90\% rec.} \\
    \hline
    '''
    prec_table_end = '''\hline
    \end{tabular}
    \caption{Precision at recall for different baseline models}
    \label{table:1}
    \end{table}
    \end{document}
    '''
    with open('./material/prec_at_rec.tex', 'w') as fw:
        fw.write(prec_table_start)
        for model_name in prec_at_80_recall:
            au = auprs[model_name]
            p8 = prec_at_80_recall[model_name]
            p9 = prec_at_90_recall[model_name]
            fw.write(f"{model_name} & {au:.3f} & {p8:.3f} & {p9:.3f} \\\\")
        fw.write(prec_table_end)
    
    print('#train, #test', total_train, total_test)
