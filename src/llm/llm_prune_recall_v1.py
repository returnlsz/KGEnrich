# 对于每个user query，以及triple list，我们关注的重点是（e1,r,?）和(?,r,e1)，即只关注头实体和关系或者尾实体和关系，还有(?,r/r',?)只关注关系。我们会有三个召回通道，分别是：
# 1.（e1,r,?）
# 2.(?,r,e1)
# 3.(?,r/r',?)
# 对于这三个召回通道，我们都需要召回其各自top k个最相关的三元组。这里k=100，所以每个query召回至多300个三元组。
# 1.然后对每个user query召回的三元组取交集，得到最终剪枝答案。为什么不采取针对每个user query对三元组的相似度分数进行相加？因为这三个通道召回的相似度分数可能存在较大差异。但是也存在user query和所有triple都相关的情况,比如LLM生成的What is {topic entity}就很盲目,导致召回的三元组很随机，在取交集的时候会存在问题。
# 2.或者对三个通道召回的相似度分数进行归一化然后加的三元组总得分上，这样差异就不存在了.然后直接取top k个即可
# task名为 llm_pruning_three_channels

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils.get_answer_entity_coverage import check_answer_in_graph_main
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm_asyncio

# def llm_prune_recall_v1_main(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,llm_pruning_topk=100,task="llm_pruning_three_channels",resume_path=None):
#     return asyncio.run(llm_prune_recall_v1(dataset_name,llm,initial_pruning_llm,initial_pruning_topk,llm_pruning_topk,task,resume_path))

subgraph_total_length = 0

def llm_prune_recall_v1(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,llm_pruning_topk=100,task="llm_pruning_three_channels",resume_path=None):
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
    # 加载路径,使用question decompose datastes
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/question_decompose_datasets'
    # 使用temp dataset
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/temp_datasets'
    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_question_decompose.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)
    
    # 获取当前时间
    current_time = datetime.now()

    # 将时间格式化为字符串（如 "2023-10-10_14-30-00"）,将这里替换成上次未完成的parquet名称可以继续
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/llm_pruning_three_channels_datasets/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_{llm_pruning_topk}_llm_pruning_three_channels_{time_str}.parquet"
    else:
        write_data_dir = resume_path

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    # write_data_dir = f"preprocess_datasets/llm_pruning_three_channels_datasets/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_{llm_pruning_topk}_llm_pruning_three_channels.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # 存放分解问题后的结果
    llm_pruning_three_channels_dataset = None
    finished_id = []

    # 检查文件是否存在，如果不存在，则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在，创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在，已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        llm_pruning_three_channels_dataset = DatasetDict({
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
        print(f"文件已存在：{write_data_dir},将从该文件继续完成llm three channels剪枝任务")
        llm_pruning_three_channels_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        for sample in llm_pruning_three_channels_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################

    # 初始化 Sentence-BERT 模型
    model = SentenceTransformer("/Users/jiangtong/KnowledgeEnrich/project/sentence-transformers",device=device)

    print(f"以下为{dataset_name}数据集LLM剪枝前的覆盖率信息:")
    check_answer_in_graph_main(dataset=dataset,task="initial_pruning")

    filtered_graph = {}
    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 30  # 设置批次大小

    # # 定义异步函数，处理单个请求
    # async def process_single_query(sample):
    #     global subgraph_total_length
    #     # 三个通道
    #     corpus1 = []
    #     corpus2 = []
    #     corpus3 = []
    #     subgraph_total_length = subgraph_total_length + len(sample["graph"])
        
    #     if llm_pruning_topk >= len(sample["graph"]):
    #         filtered_graph[sample["id"]] = sample["graph"]
    #         question_id = sample["id"]
    #         # 通过 question_id 快速检索对应的记录
    #         example = id_to_example_map.get(question_id)
    #         if example:
    #             # 为 example 添加预测结果
    #             example_with_prediction = {**example, "pruned_graph": sample["graph"]}
    #             batch_results.append(example_with_prediction)
    #     else:
    #         # 将每个元素转换为字符串后存放到 corpus
    #         # 通道1
    #         for sublist in sample["graph"]:
    #             corpus1.append(" ".join(map(str, sublist[:2])))
    #         corpus_embeddings1 = model.encode(corpus1)
    #         # 通道2
    #         for sublist in sample["graph"]:
    #             corpus2.append(" ".join(map(str, sublist[-2:])))
    #         corpus_embeddings2 = model.encode(corpus2)
    #         # 通道3
    #         for sublist in sample["graph"]:
    #             corpus3.append((map(str, sublist[1])))
    #         # 对候选集进行嵌入
    #         corpus_embeddings3 = model.encode(corpus3)

    #         # 对嵌入向量进行归一化（余弦相似度需要将向量归一化到单位范数）
    #         corpus_embeddings1 = corpus_embeddings1 / np.linalg.norm(corpus_embeddings1, axis=1, keepdims=True)
    #         corpus_embeddings2 = corpus_embeddings2 / np.linalg.norm(corpus_embeddings2, axis=1, keepdims=True)
    #         corpus_embeddings3 = corpus_embeddings3 / np.linalg.norm(corpus_embeddings3, axis=1, keepdims=True)

    #         # 构建 FAISS 索引（使用内积）
    #         embedding_dim = corpus_embeddings1.shape[1]  # 嵌入向量的维度
    #         index1 = faiss.IndexFlatIP(embedding_dim)    # 使用内积
    #         index1.add(corpus_embeddings1)               # 添加嵌入向量到索引中
    #         index2 = faiss.IndexFlatIP(embedding_dim)    # 使用内积
    #         index2.add(corpus_embeddings2)
    #         index3 = faiss.IndexFlatIP(embedding_dim)    # 使用内积
    #         index3.add(corpus_embeddings3)

    #         # 对 sample["user_queries"] 中每个 query 计算得分
    #         total_scores = np.zeros(len(corpus1))  # 用于存储每个 corpus 的总得分

    #         for query in sample["user_queries"]:
    #             query_embedding = model.encode([query])  # 对查询嵌入

    #             # 对查询向量进行归一化
    #             query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    #             # 通道1
    #             distances1, indices1 = index1.search(query_embedding, len(corpus1))
    #             # 通道2
    #             distances2, indices2 = index2.search(query_embedding, len(corpus2))
    #             # 通道1
    #             distances3, indices3 = index3.search(query_embedding, len(corpus3))

    #             # 对每个通道的得分进行归一化处理
    #             normalized_scores1 = (distances1 - distances1.min()) / (distances1.max() - distances1.min() + 1e-9)
    #             normalized_scores2 = (distances2 - distances2.min()) / (distances2.max() - distances2.min() + 1e-9)
    #             normalized_scores3 = (distances3 - distances3.min()) / (distances3.max() - distances3.min() + 1e-9)

    #             # 累积归一化得分到总得分
    #             total_scores += normalized_scores1
    #             total_scores += normalized_scores2
    #             total_scores += normalized_scores3

    #         # 按总得分降序排序，并选取 top k,使用llm_pruning_topk
    #         top_k_indices = np.argsort(total_scores)[::-1][:llm_pruning_topk]

    #         # 筛选后的三元组列表
    #         triple_filtered_graph = [sample["graph"][idx] for idx in top_k_indices]
    #         return sample["id"],triple_filtered_graph

    # # 定义异步函数，处理多个请求并按完成顺序处理结果
    # tasks = [
    #     process_single_query(sample)
    #     for sample in dataset["test"]
    #     if sample["id"] not in finished_id  # 已经剪枝过的就跳过
    # ]

    # # 使用 tqdm_asyncio 显示进度条
    # with tqdm_asyncio(desc=f"Call {llm} for llm pruning three channels", total=len(dataset["test"])) as pbar:
    #     for future in asyncio.as_completed(tasks):
    #         result = await future  # 等待某个任务完成
    #         question_id, triple_filtered_graph = result  # 从返回值解构
    #         pbar.update(1)  # 更新进度条

    for sample in tqdm(dataset["test"], desc="LLM three channels 剪枝"):
        global subgraph_total_length
        # 三个通道
        corpus1 = []
        corpus2 = []
        corpus3 = []
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
            # 通道1
            for sublist in sample["graph"]:
                corpus1.append(" ".join(map(str, sublist[:2])))
            corpus_embeddings1 = model.encode(corpus1)
            # 通道2
            for sublist in sample["graph"]:
                corpus2.append(" ".join(map(str, sublist[-2:])))
            corpus_embeddings2 = model.encode(corpus2)
            # 通道3
            for sublist in sample["graph"]:
                corpus3.append(" ".join(map(str, sublist[1])))
            # 对候选集进行嵌入
            corpus_embeddings3 = model.encode(corpus3)

            # 对嵌入向量进行归一化（余弦相似度需要将向量归一化到单位范数）
            corpus_embeddings1 = corpus_embeddings1 / np.linalg.norm(corpus_embeddings1, axis=1, keepdims=True)
            corpus_embeddings2 = corpus_embeddings2 / np.linalg.norm(corpus_embeddings2, axis=1, keepdims=True)
            corpus_embeddings3 = corpus_embeddings3 / np.linalg.norm(corpus_embeddings3, axis=1, keepdims=True)

            # 构建 FAISS 索引（使用内积）
            embedding_dim = corpus_embeddings1.shape[1]  # 嵌入向量的维度
            index1 = faiss.IndexFlatIP(embedding_dim)    # 使用内积
            index1.add(corpus_embeddings1)               # 添加嵌入向量到索引中
            index2 = faiss.IndexFlatIP(embedding_dim)    # 使用内积
            index2.add(corpus_embeddings2)
            index3 = faiss.IndexFlatIP(embedding_dim)    # 使用内积
            index3.add(corpus_embeddings3)

            # 对 sample["user_queries"] 中每个 query 计算得分
            total_scores = np.zeros(len(corpus1))  # 用于存储每个 corpus 的总得分

            for query in sample["user_queries"]:
                query_embedding = model.encode([query])  # 对查询嵌入

                # 对查询向量进行归一化
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

                # 通道1
                distances1, indices1 = index1.search(query_embedding, len(corpus1))
                # 通道2
                distances2, indices2 = index2.search(query_embedding, len(corpus2))
                # 通道1
                distances3, indices3 = index3.search(query_embedding, len(corpus3))

                # 对每个通道的得分进行归一化处理
                normalized_scores1 = (distances1[0] - distances1[0].min()) / (distances1[0].max() - distances1[0].min() + 1e-9)
                normalized_scores2 = (distances2[0] - distances2[0].min()) / (distances2[0].max() - distances2[0].min() + 1e-9)
                normalized_scores3 = (distances3[0] - distances3[0].min()) / (distances3[0].max() - distances3[0].min() + 1e-9)

                # 累积归一化得分到总得分
                total_scores += normalized_scores1
                total_scores += normalized_scores2
                total_scores += normalized_scores3

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
                if len(llm_pruning_three_channels_dataset["test"]) == 0:  # 如果 llm_pruning_three_channels_dataset["test"] 是空的
                    llm_pruning_three_channels_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    llm_pruning_three_channels_dataset["test"] = Dataset.from_dict({
                        key: llm_pruning_three_channels_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in llm_pruning_three_channels_dataset["test"].column_names
                    })

                # 写入到文件
                llm_pruning_three_channels_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储

    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(llm_pruning_three_channels_dataset["test"]) == 0:  # 如果 llm_pruning_three_channels_dataset["test"] 是空的
            llm_pruning_three_channels_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            llm_pruning_three_channels_dataset["test"] = Dataset.from_dict({
                key: llm_pruning_three_channels_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in llm_pruning_three_channels_dataset["test"].column_names
            })
        llm_pruning_three_channels_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的llm three channels pruning任务!")

    # 检查覆盖后的子图的长度以及答案覆盖率
    pruned_subgraph_total_length = 0
    for sample in dataset["test"]:
        pruned_subgraph_total_length = pruned_subgraph_total_length + len(sample["pruned_graph"])

    print("数据集初步剪枝前三元组总数量:",subgraph_total_length)
    print("初步剪枝后三元组总数量:",pruned_subgraph_total_length)
    print("二者比率:",pruned_subgraph_total_length/subgraph_total_length)
    print(f"以下为{dataset_name}数据集剪枝后的覆盖率信息:")
    check_answer_in_graph_main(dataset=llm_pruning_three_channels_dataset,task="llm_pruning")
