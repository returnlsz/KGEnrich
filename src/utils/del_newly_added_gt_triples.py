import sys
import os

from get_gt_triples_coverage import get_gt_triples_coverage_main

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from tqdm import tqdm

result_dict = {}

def del_newly_add_gt_triples(dataset):
    for sample in dataset["test"]:
        gt_triples = sample["gt_triples"]
        subgraph = sample["graph"]
        del_gt_triples = []
        for gt_triple in gt_triples:
            if len(gt_triple) != 3:
                del_gt_triples.append(gt_triple)
                continue
            else:
                for triple in subgraph:
                    if gt_triple[0] == triple[0] and gt_triple[1] == triple[1] and gt_triple[2] == triple[2]:
                        del_gt_triples.append(gt_triple)
                        break
        result_dict[sample["id"]] = del_gt_triples

def del_newly_add_gt_triples_main(samples):
    # 批量操作时，`samples` 是一个字典，字段是批量字段列表
    # 需要逐一处理批量中的每个样本
    ids = samples["id"]  # 批量样本的 ID 列表
    gt_triples = []  # 用于存储处理后的 gt_triples

    for i in ids:
        # 根据单个样本的 ID 从 result_dict 获取对应结果
        gt_triples.append(result_dict[i])  # 添加对应 gt_triples

    # 更新批量的 "gt_triples" 字段
    samples["gt_triples"] = gt_triples
    return samples

if __name__ == "__main__":

    dataset = load_dataset(
        "parquet", 
        data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_filter_gt_triples_datasets/webqsp_gpt4o-mini_sentence-transformers_750_llm_filter_gt_triples.parquet'}
    )
    get_gt_triples_coverage_main(dataset)

    del_newly_add_gt_triples(dataset)

    dataset["test"] = dataset["test"].map(
        del_newly_add_gt_triples_main,
        batched=True,  # 启用批量操作以加速
        batch_size=64,  # 设置合适的批量大小
        desc="正在删除new add GT triples"
    )
    get_gt_triples_coverage_main(dataset)

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{topk}_{initial_pruning}
    dataset["test"].to_parquet(f"preprocess_datasets/temp_datasets/webqsp_gpt4o-mini_sentence-transformers_750_llm_filter_gt_triples.parquet")