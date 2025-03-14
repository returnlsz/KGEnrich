# 此脚本用于统计数据集中的graph字段的长度低于某个阈值k的数量
# input:
# subgraph
# output:
# 低于长度k的graph的样本数量

from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# 阈值k

# allk = [600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]
# allk = [3500,4000,4500,5000,5500,6000,6500,7000]
allk = [7500,8000,8500,9000,9500,10000]
num = 0

if __name__ == "__main__":
    # 加载路径
    # dataset_name = "cwq"
    dataset_name = "webqsp"
    # cwq数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    # 打印数据集的信息
    print(dataset)
    for k in allk:
        for sample in tqdm(dataset["test"],desc="统计中"):
            if len(sample["graph"]) < k:
                num = num + 1
        print(f"{dataset_name}数据集graph字段长度低于阈值{k}的样本数量有:{num}个")
        num = 0


