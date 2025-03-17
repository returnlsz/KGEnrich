# 此脚本用于评估predictions的指标结果
import json
import re 
import string
from sklearn.metrics import precision_score
from datasets import load_dataset

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    # 把特殊符号都去掉"!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(dataset_name, cal_f1=True, topk = -1,llm=None):
    # 定义输入文件的路径
    qa_file = "/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/qa_datasets/webqsp_gpt4o-mini_direct_qa_2025-03-15_21-08-12.parquet"
    qa_file = "/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/qa_datasets/cwq_gpt4o-mini_direct_qa_2025-03-15_20-54-51.parquet"
    # qa_file = f"/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/qa_datasets/{dataset_name}_{llm}_qa.parquet"
    detailed_eval_file = f"/Users/jiangtong/KnowledgeEnrich/project/result/predictions_eval/{dataset_name}_{llm}_predictions.jsonl"

    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(detailed_eval_file, 'w') as f2:
        # 加载路径
        data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/qa_datasets'
        dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/cwq_gpt4o-mini_direct_qa_2025-03-16_18-12-41.parquet'})
        for data in dataset["test"]:
            id = data["id"]
            prediction = data['predictions']
            # 可能需要换成answer字段
            answer = data['answer']
        # for line in f:
        #     try:
        #         data = json.loads(line)
        #     except:
        #         print(line)
        #         continue
        #     id = data['id']
        #     prediction = data['predictions']
        #     # 可能需要换成answer字段
        #     answer = data['answer']
            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                else:
                    prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = ' '.join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({'id': id, 'predictions': prediction, 'answer': answer, 'acc': acc, 'hit': hit, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score}) + '\n')
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({'id': id, 'predictions': prediction, 'answer': answer, 'acc': acc, 'hit': hit}) + '\n')
    
    if len(f1_list) > 0:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(sum(recall_list) * 100 / len(recall_list))
    else:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list))
    print(result_str)


    result_name = f"{dataset_name}_{llm}_eval_result_top_{topk}.txt" if topk > 0 else f'{dataset_name}_{llm}_eval_result.txt'
    eval_result_path = "/Users/jiangtong/KnowledgeEnrich/project/result/result_eval/" + result_name
    with open(eval_result_path, 'w') as f:
        f.write(result_str)

def eval_result_main(dataset_name=None,cal_f1=True,top_k=-1,llm=None):
    eval_result(dataset_name,cal_f1,top_k,llm)
    