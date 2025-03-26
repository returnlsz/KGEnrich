# 此脚本用于计算给定子图中的冗余比例,计算方式,检查每两个实体之间的relation,计算r之间的相似度并归一化
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

def calculate_upper_triangle_mean(relations,model=None):
    # 对所有relation进行嵌入
    embeddings = model.encode(relations, convert_to_tensor=False)
    
    # 计算所有嵌入之间的余弦相似度矩阵
    num_relations = len(relations)
    similarity_matrix = np.zeros((num_relations, num_relations))

    for i in range(num_relations):
        for j in range(num_relations):
            if i != j:  # 避免计算对角线
                similarity_matrix[i][j] = np.dot(embeddings[i], embeddings[j]) / \
                                          (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))

    # 提取上三角部分（不包含对角线）
    upper_triangle_indices = np.triu_indices(num_relations, k=1)
    upper_triangle_values = similarity_matrix[upper_triangle_indices]
    
    # 计算上三角部分的均值
    mean_value = np.mean(upper_triangle_values)
    
    return mean_value

# 计算一个sample的冗余性分数
def caculate_redundancy_score(dataset=None,model=None,graph_name="graph"):
    all_score = 0
    for sample in tqdm(dataset,desc="计算冗余分数中..."):
        subgraph = sample[graph_name]
        # 用一个dict储存e1和e2之间的关系
        e2e_dict = build_triple_structure(subgraph)

        result = 0

        for entities,relations in e2e_dict.items():
            if len(relations) < 2:
                continue
            # 计算重复的总数
            result += calculate_upper_triangle_mean(relations,model)
            
        # 这里不能直接除以len(subgraph),而是应该除以e2e_dict元素的个数

        all_score += result/len(e2e_dict)
    return all_score/len(dataset)


if __name__ == "__main__":
    # 加载嵌入模型
    device = torch.device("mps")
    model = SentenceTransformer("/Users/jiangtong/KnowledgeEnrich/project/sentence-transformers",device=device)

    # 加载路径
    # cwq数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    # dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})
    # dataset = load_dataset("parquet", data_files={'test': f'/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_three_channels_datasets/cwq_gpt4o-mini_sentence-transformers_750_300_llm_pruning_three_channels_2025-03-24_14-57-40.parquet'})
    dataset = load_dataset("parquet", data_files={'test': f'/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_three_channels_datasets/webqsp_gpt4o-mini_sentence-transformers_750_300_llm_pruning_three_channels_2025-03-24_14-57-18.parquet'})
    # 打印数据集的信息
    print(dataset)

    score = caculate_redundancy_score(dataset["test"],model,"pruned_graph")

    print("平均冗余性分数是:",score)

    # 测试用例  
    # subgraph = [["a", "b", "c"],["a", "b", "c"],["a", "b", "c"],["a", "b", "c"],["a", "b", "c"]]
    # score = caculate_redundancy_score(subgraph,model)
    # print(score)
