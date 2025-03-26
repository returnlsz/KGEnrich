# 此脚本用于评估子图中的三元组与user queries的相关性
# input:
# embedding_llm: e.g. sentence_transformers
# subgraph: A list,each element is list that length is 3
# user queries: A list,containing origin question, compound questions and unit questions
# output:
# relavant score

from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# 计算dataset中所有sample的均分
def caculate_relavance_score(dataset=None,embedding_llm="sentence-transformers",model=None,graph_name="graph"):
    all_score = 0
    for sample in tqdm(dataset,desc="计算相似度..."):
        subgraph = sample[graph_name]
        # 使用每个user query对subgraph中的所有三元组进行打分,即相似度计算,维护一个score_index记录每个三元组的得分,
        # 输出是最终平均得分

        corpus = []
        for triple in subgraph:
            corpus.append(" ".join(map(str, triple)))

        corpus_embeddings = model.encode(corpus)
        # 对嵌入向量进行归一化（余弦相似度需要将向量归一化到单位范数）
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        score_index = []
        user_queries = []
        user_queries.append(sample["question"])
        if "user_queries" in sample:
            for query in sample["user_queries"]:
                user_queries.append(query)

        for user_query in user_queries:
            user_query_embedding = model.encode(user_query)
            # 如果需要二维数组归一化
            if len(user_query_embedding.shape) == 2:
                user_query_embedding = user_query_embedding / np.linalg.norm(user_query_embedding, axis=1, keepdims=True)
            else:
                # 如果是 1D 数组，按方法 1 处理
                user_query_embedding = user_query_embedding / np.linalg.norm(user_query_embedding)
            # user_query_embedding = user_query_embedding / np.linalg.norm(user_query_embedding, axis=1, keepdims=True)

            for i, corpus_vec in enumerate(corpus_embeddings):
                # 计算点积（因为已经归一化了，点积即为余弦相似度）
                similarity = np.dot(user_query_embedding, corpus_vec)
                # 第一次添加
                if len(score_index) == i:
                    score_index.append((similarity))  # 将相似度存入列表
                else:
                    score_index[i] = max(score_index[i],similarity)
        # 返回平均最终得分
        all_score += sum(score_index)/len(score_index)
    return all_score/(len(dataset))

if __name__ == "__main__":
    device = torch.device("mps")
    model = SentenceTransformer("/Users/jiangtong/KnowledgeEnrich/project/sentence-transformers",device=device)

    # 加载路径
    # cwq数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    # dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})
    dataset = load_dataset("parquet", data_files={'test': f'/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_three_channels_datasets/webqsp_gpt4o-mini_sentence-transformers_750_300_llm_pruning_three_channels_2025-03-24_14-57-18.parquet'})
    # dataset = load_dataset("parquet", data_files={'test': f'/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_three_channels_datasets/cwq_gpt4o-mini_sentence-transformers_750_300_llm_pruning_three_channels_2025-03-24_14-57-40.parquet'})
    # 打印数据集的信息
    print(dataset)
    embedding_llm = "sentence-transformers"

    score = caculate_relavance_score(dataset["test"],embedding_llm,model,"pruned_graph")

    print("平均得分:",score)
    