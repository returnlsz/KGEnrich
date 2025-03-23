
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils.get_answer_entity_coverage import check_answer_in_graph_main
import torch
from datasets import load_dataset
import pandas as pd
from datasets import Dataset, DatasetDict

# step 0
# 从parquet文件中加载数据集，并把数据集组织成一个list-dict,dict的字段如下:
# id
# question
# answer
# q_entity
# a_entity
# graph
# choices


def llm_prune(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,llm_pruning_topk=300,task="llm_pruning"):
    # 全局环境变量设置
    ###############################################################################################################
    # this ensures that the current MacOS version is at least 12.3+
    print("the current MacOS version is at least 12.3+:",torch.backends.mps.is_available())
    # this ensures that the current PyTorch installation was built with MPS activated.
    print("the current PyTorch installation was built with MPS activated:",torch.backends.mps.is_built())
    device = torch.device("mps")
    print(f"Using device: {device}")  # 输出当前设备信息
    ###############################################################################################################
    
    
    ###############################################################################################################
    # 加载路径
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/question_decompose_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_question_decompose.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)
    
    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    write_data_dir = f"preprocess_datasets/llm_pruning_datasets/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_{llm_pruning_topk}_llm_pruning.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # 存放分解问题后的结果
    llm_pruning_dataset = None
    finished_id = []

    # 检查文件是否存在，如果不存在，则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在，创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在，已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        llm_pruning_dataset = DatasetDict({
            "test": Dataset.from_dict({
                "id": "",
                "question": "",
                "user_queries":[],
                "answer": [],
                "q_entity": [],
                "a_entity": [], 
                "graph": [],
                "pruned_graph": [],
                "choices": []
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成llm剪枝任务")
        llm_pruning_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        for sample in llm_pruning_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################

    ###############################################################################################################

    # step 4
    # 剪枝，根据每个user queries找到与其相似的top k个triple unit queries，需要融合子图（去重复），并将对应的三元组加入到pruned_graph中，字段如下:
    # id
    # question
    # user_queries
    # answer
    # q_entity
    # a_entity
    # graph
    # pruned_graph
    # choices

    # 最终输出文件的名称命名为{dataset_name}_{llm}_{embedding_model_name}_{faiss}_{topk}_llm_prune.parquet

    ###############################################################################################################
    # TODO:需要根据user unit query进行剪枝

    # 初始化 Sentence-BERT 模型
    model = SentenceTransformer("/Users/jiangtong/KnowledgeEnrich/project/sentence-transformers",device=device)

    print(f"以下为{dataset_name}数据集LLM剪枝前的覆盖率信息:")
    check_answer_in_graph_main(dataset=dataset,task="initial_pruning")

    filtered_graph = {}
    subgraph_total_length = 0
    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 100  # 设置批次大小

    for sample in tqdm(dataset["test"], desc="LLM剪枝"):
        corpus = []
        subgraph_total_length = subgraph_total_length + len(sample["graph"])
        # 已经剪枝过的就跳过
        if sample["id"] in finished_id:
            continue
        if llm_pruning_topk >= len(sample["graph"]):
            filtered_graph[sample["id"]] = sample["graph"]
            question_id = sample["id"]
            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "pruned_graph": sample["graph"]}
                batch_results.append(example_with_prediction)
        else:
            # 将每个元素转换为字符串后存放到 corpus
            for sublist in sample["graph"]:
                corpus.append(" ".join(map(str, sublist)))
            # 对候选集进行嵌入
            corpus_embeddings = model.encode(corpus)

            # 对嵌入向量进行归一化（余弦相似度需要将向量归一化到单位范数）
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

            # 构建 FAISS 索引（使用内积）
            embedding_dim = corpus_embeddings.shape[1]  # 嵌入向量的维度
            index = faiss.IndexFlatIP(embedding_dim)    # 使用内积
            index.add(corpus_embeddings)               # 添加嵌入向量到索引中

            # 对 sample["user_queries"] 中每个 query 计算得分
            total_scores = np.zeros(len(corpus))  # 用于存储每个 corpus 的总得分
            for query in sample["user_queries"]:
                query_embedding = model.encode([query])  # 对查询嵌入

                # 对查询向量进行归一化
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

                # 计算与 corpus 的相似度（得分）
                distances, indices = index.search(query_embedding, len(corpus))

                # 累积得分到 total_scores 中
                total_scores += distances[0]

            # 按总得分降序排序，并选取 top k,使用llm_pruning_topk
            top_k_indices = np.argsort(total_scores)[::-1][:llm_pruning_topk]

            # 筛选后的三元组列表
            triple_filtered_graph = [sample["graph"][idx] for idx in top_k_indices]

            filtered_graph[sample["id"]] = triple_filtered_graph

            question_id = sample["id"]
            processed_answer = triple_filtered_graph
            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "pruned_graph": processed_answer}
                batch_results.append(example_with_prediction)

            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(llm_pruning_dataset["test"]) == 0:  # 如果 llm_pruning_dataset["test"] 是空的
                    llm_pruning_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    llm_pruning_dataset["test"] = Dataset.from_dict({
                        key: llm_pruning_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in llm_pruning_dataset["test"].column_names
                    })

                # 写入到文件
                llm_pruning_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储

    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(llm_pruning_dataset["test"]) == 0:  # 如果 llm_pruning_dataset["test"] 是空的
            llm_pruning_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            llm_pruning_dataset["test"] = Dataset.from_dict({
                key: llm_pruning_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in llm_pruning_dataset["test"].column_names
            })
        llm_pruning_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的llm pruning任务!")
    

    # 检查覆盖后的子图的长度以及答案覆盖率
    pruned_subgraph_total_length = 0
    for sample in dataset["test"]:
        pruned_subgraph_total_length = pruned_subgraph_total_length + len(sample["graph"])

    print("数据集初步剪枝前三元组总数量:",subgraph_total_length)
    print("初步剪枝后三元组总数量:",pruned_subgraph_total_length)
    print("二者比率:",pruned_subgraph_total_length/subgraph_total_length)
    print(f"以下为{dataset_name}数据集剪枝后的覆盖率信息:")
    check_answer_in_graph_main(dataset=llm_pruning_dataset,task=task)

    ###############################################################################################################