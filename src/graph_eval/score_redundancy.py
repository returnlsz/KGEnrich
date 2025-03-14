# 此脚本用于计算给定子图中的冗余比例,计算方式,检查每两个实体之间的relation,检查有无重复的relation,计算EM的个数
# score3=sum{for each triple{-2*sigmoid(EM numbers{for r in (h,r,t)})+2}}/same_h_and_r_triple_numbers
# input:h
# subgraph
# output:
# redundancy_score

from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
from collections import Counter
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def build_triple_structure(subgraph):
    """
    构建三元组数据结构。
    
    :param subgraph: list,子图,其中每个元素是一个三元组 [e1, r, e2]
                     e1 和 e2 是实体,r 是关系
    :return: dict,数据结构,key 是 (e1, e2),value 是一个 list,存储所有关系 r
    """
    triple_dict = {}

    # 遍历子图中的三元组
    for triple in subgraph:
        # 确保三元组的格式正确
        if len(triple) != 3 or not all(isinstance(x, str) for x in triple):
            raise ValueError(f"Invalid triple format: {triple}. Expected [e1, r, e2] with all elements as strings.")
        
        e1, r, e2 = triple  # 解包三元组
        
        # 使用 (e1, e2) 作为 key
        key = (e1, e2)
        
        # 如果 key 不存在,则创建一个新列表
        if key not in triple_dict:
            triple_dict[key] = []
        
        # 将关系 r 添加到列表中
        triple_dict[key].append(r)

    return triple_dict

def caculate_redundancy_score(subgraph=[]):
    # 用一个dict储存e1和e2之间的关系
    e2e_dict = build_triple_structure(subgraph)

    result = 0

    for entities,relations in e2e_dict.items():
        e1,e2 = entities
        # 计算重复的总数
        counter = Counter(relations)
        duplicate_count = sum(count - 1 for count in counter.values() if count > 1)
        result = result + (-2 * sigmoid(duplicate_count) + 2)
    # 这里不能直接除以len(subgraph),而是应该除以e2e_dict元素的个数

    return result / len(e2e_dict)


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
    # 测试用例
    subgraph = [["a", "b", "c"],["a", "b", "c"],["a", "b", "c"],["a", "b", "c"],["a", "b", "c"]]
    score = caculate_redundancy_score(subgraph)
    print("冗余性分数是:",score)
