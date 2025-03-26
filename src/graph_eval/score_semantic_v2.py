# 计算一个子图的语义丰富性
from datasets import load_dataset
import torch

def caculate_semantic_score(dataset=None):
    all_score = 0
    for sample in dataset:
        subgraph = sample["graph"]



if __name__ == "__main__":
    # 加载路径
    # cwq数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    # 打印数据集的信息
    print(dataset)