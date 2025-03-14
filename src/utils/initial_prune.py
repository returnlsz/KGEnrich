# 此文件用于对原数据集进行剪枝，输出剪枝后的数据集
# dataset path:
# /Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq
# /Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp
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

def initial_prune(dataset_name = "webqsp",initial_pruning_llm = "sentence-transformers",initial_pruning_topk = 100):

    # 全局环境变量设置
    ###############################################################################################################
    # this ensures that the current MacOS version is at least 12.3+
    print("the current MacOS version is at least 12.3+:",torch.backends.mps.is_available())
    # this ensures that the current PyTorch installation was built with MPS activated.
    print("the current PyTorch installation was built with MPS activated:",torch.backends.mps.is_built())
    device = torch.device("mps")
    print(f"Using device: {device}")  # 输出当前设备信息
    ###############################################################################################################

    # step 0
    # 从parquet文件中加载数据集，并把数据集组织成一个list-dict,dict的字段如下:
    # id
    # question
    # answer
    # q_entity
    # a_entity
    # graph
    # choices

    ###############################################################################################################
    # 加载路径
    # cwq数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    # 打印数据集的信息
    print(dataset)

    ###############################################################################################################

    # step 1
    # 需要先对graph进行预剪枝,否则后续进行三元组翻译的时候上下文会非常长,导致llm回答时间特别长,这里直接用embedding的方法

    # 最终输出文件的名称命名为{dataset_name}_{embedding_llm}_{top k}_{initial_pruning}

    ###############################################################################################################
    subgraph_total_length = 0
    # 初始化 Sentence-BERT 模型
    model = SentenceTransformer("/Users/jiangtong/KnowledgeEnrich/project/sentence-transformers",device=device)

    print(f"以下为{dataset_name}数据集初步剪枝前的覆盖率信息:")
    check_answer_in_graph_main(dataset=dataset)

    filtered_graph = {}
    for sample in tqdm(dataset["test"],desc="预剪枝"):
        corpus = []
        subgraph_total_length = subgraph_total_length + len(sample["graph"])
        if initial_pruning_topk >= len(sample["graph"]):
            filtered_graph[sample["id"]] = sample["graph"]
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

            # 输入一个查询
            query = sample["question"]
            query_embedding = model.encode([query])    # 对查询嵌入

            # 对查询向量进行归一化
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

            # 检索 top k
            k = initial_pruning_topk
            distances, indices = index.search(query_embedding, k)

            # 筛选后的三元组列表
            triple_filtered_graph = [sample["graph"][idx] for i, idx in enumerate(indices[0])]

            filtered_graph[sample["id"]] = triple_filtered_graph

    # 将 filtered_graph 转化为以 question_id 为键的字典
    filtered_graph_dict = {question_id: subgraph for question_id, subgraph in filtered_graph.items()}

    def pruned_cover_dataset(batch):
        # 遍历批量数据，将匹配到的 `graph` 字段更新为 `filtered_graph_dict` 中的值
        for i, sample_id in enumerate(batch["id"]):
            if sample_id in filtered_graph_dict:
                batch["graph"][i] = filtered_graph_dict[sample_id]
        return batch

    # 使用 Dataset.map 函数进行批量更新，并避免重复遍历
    dataset["test"] = dataset["test"].map(
        pruned_cover_dataset, 
        batched=True,  # 启用批量操作以加速
        batch_size=64,  # 设置合适的批量大小
        desc="预剪枝-覆盖原来的子图"
    )

    # 检查覆盖后的子图的长度以及答案覆盖率
    pruned_subgraph_total_length = 0
    for sample in dataset["test"]:
        pruned_subgraph_total_length = pruned_subgraph_total_length + len(sample["graph"])

    print("数据集初步剪枝前三元组总数量:",subgraph_total_length)
    print("初步剪枝后三元组总数量:",pruned_subgraph_total_length)
    print("二者比率:",pruned_subgraph_total_length/subgraph_total_length)
    print(f"以下为{dataset_name}数据集剪枝后的覆盖率信息:")
    check_answer_in_graph_main(dataset=dataset)

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{topk}_{initial_pruning}
    dataset["test"].to_parquet(f"preprocess_datasets/initial_pruning_datasets/{dataset_name}_{initial_pruning_llm}_{initial_pruning_topk}_initial_pruning.parquet")
    sys.exit()


# ###############################################################################################################


# # step 2:
# # 把question分解成user queries，并储存起来，字段如下:
# # id
# # question
# # user_queries
# # answer
# # q_entity
# # a_entity
# # graph
# # triple_unit_queries
# # choices

# # 最终输出文件的名称命名为{dataset_name}_{llm}_{user_queries}

# ###############################################################################################################

# # 用于提取 question 并生成 result_prompt
# def process_dataset(dataset):
#     # 存储结果
#     result_prompts = {}
    
#     # 遍历数据集中的每一行
#     for example in tqdm(dataset['test'],desc="Building question decompose prompts"):
#         # 提取 question 字段
#         question_id = example['id']
#         question = example['question']
#         mode = "question_decompose"
#         # 调用 PromptBuilder，传入 question
#         prompt_builder = PromptBuilder(question,mode)
        
#         # 获取生成的 result_prompt
#         result_prompts[question_id] = prompt_builder.build_prompt()

#         # print(result_prompts)
#         # return result_prompts
    
#     return result_prompts

# # 调用函数处理数据集
# question_decompose_result_prompts = process_dataset(dataset)
# llm_chat = llm_client() 
# mode = "question_decompose"
# # sub_queries = llm_chat.response(question_decompose_result_prompts[0],mode)
# # print(sub_queries)

# # 存放分解问题后的结果
# decompose_question_dataset = dataset

# # response预处理
# def extract_questions(sub_queries):
#     """
#     从给定的sub_queries文本中提取所有问题,并返回一个格式化后的问题列表。

#     :param sub_queries: 包含问题和子问题的树形结构的文本
#     :return: 格式化后的问题列表
#     """
#     # 移除所有的"-", "--", "---"和换行符
#     cleaned_text = sub_queries.replace("-", "").replace("--", "").replace("---", "").replace("\n", " ")
    
#     # 分割文本以提取所有问题
#     questions = [question.strip() for question in cleaned_text.split("?") if question.strip()]
    
#     return questions


# with tqdm(question_decompose_result_prompts.items(), desc="Call LLM for question decomposing and Mapping to datasets") as pbar:
#     for question_id, question in pbar:
#         mode = "question_decompose"
#         sub_queries = llm_chat.response(question, mode)

#         pbar.set_postfix(current_question_id=question_id)

#         # 对response进行预处理
#         question_list = extract_questions(sub_queries)

#         # 确保每条记录都初始化了 'user_queries' 列
#         if "user_queries" not in decompose_question_dataset["test"].column_names:
#             decompose_question_dataset["test"] = decompose_question_dataset["test"].add_column("user_queries", [None] * len(decompose_question_dataset["test"]))

#         # 定义一个函数用于修改特定 id 的数据
#         def add_user_queries(example):
#             if example["id"] == question_id:
#                 example["user_queries"] = question_list
#             return example
#         decompose_question_dataset["test"] = decompose_question_dataset["test"].map(add_user_queries)

# print("问题分解后的数据集样例:",decompose_question_dataset['test'][0])

# ###############################################################################################################

# # step 3.1
# # 需要先对graph进行预剪枝,否则后续进行三元组翻译的时候上下文会非常长,导致llm回答时间特别长,这里直接用embedding的方法

# # 最终输出文件的名称命名为{dataset_name}_{llm}_{initial_pruning}

# ###############################################################################################################
# # TODO:


# ###############################################################################################################


# # step 3.2
# # 把graph中的每个三元组翻译成query,并储存起来,字段如下：
# # id
# # question
# # user_queries
# # answer
# # q_entity
# # a_entity
# # graph
# # triple_unit_queries
# # choices

# # 最终输出文件的名称命名为{dataset_name}_{llm}_{triple_unit_queries}

# ###############################################################################################################

# # 建立三元组翻译的prompt
# triples_trans_prompt = {}
# with tqdm(decompose_question_dataset['test'], desc="Building triple transaction prompts") as pbar:
#     for each_sample in pbar:
#         pbar.set_postfix(current_question=each_sample["question"])
#         sub_graph = each_sample["graph"]
        
#         triple_text_list = []
#         for triple in sub_graph:
#             if isinstance(triple, list) and len(triple) == 3:  # 确保每个三元组是长度为3的列表
#                 triple_text_list.append(f"({triple[0]}, {triple[1]}, {triple[2]})")
        
#         # 用换行符连接每个三元组文本
#         sub_graph_text = ",".join(triple_text_list)

#         mode = "triples_trans"
#         # 调用 PromptBuilder，传入 question
#         prompt_builder = PromptBuilder(sub_graph_text,mode)
#         triples_trans_prompt[each_sample["id"]] = prompt_builder.build_prompt()

# # 调用llm翻译三元组
# llm_chat = llm_client() 
# mode = "triples_trans"
# triple_queries = {}
# with tqdm(triples_trans_prompt.items(),desc="Call LLM for triple transaction") as pbar:
#     for question_id,each_triples_trans_prompt in pbar:
#         pbar.set_postfix(current_question=question_id)
#         response = llm_chat.response(each_triples_trans_prompt,mode)

#         # 示例response文本
#         # response = """Natural Language Question: 
#         # (Beijing,located in,?):Which country does Beijing locate? (?,located in,China):What cities or places are located in China?
#         # (Eiffel Tower, located in, ?):In which city is the Eiffel Tower located? (?, located in, Paris):What landmarks or places are located in Paris?
#         # (Apple, founded by, ?):Who founded Apple? (?, founded by, Steve Jobs):Which companies or organizations were founded by Steve Jobs?
#         # (Python, created by, ?):Who created Python? (?, created by, Guido van Rossum):What programming languages or projects were created by Guido van Rossum?
#         # (Tesla, CEO of, ?):Who is the CEO of Tesla? (?, CEO of, Elon Musk):Which companies or organizations have Elon Musk as their CEO?
#         # """
#         def extract_triple_queries(response):
#             # 初始化最终结果列表
#             triple_queries = []
            
#             # 首先查找是否有"Natural Language Question: "
#             start_index = response.find("Natural Language Question: ")
#             if start_index != -1:
#                 # 如果存在，提取从"Natural Language Question: "开始的内容
#                 response = response[start_index + len("Natural Language Question: "):]
            
#             # 按行分割内容
#             lines = response.splitlines()
            
#             for line in lines:
#                 # 跳过空行
#                 line = line.strip()
#                 if not line:
#                     continue
                
#                 # 找到三元组部分和对应问题的分隔符 ":"
#                 if ":" in line:
#                     part = line.split(":", 2)
#                     if len(part) == 2:
#                         triple_part = part[0]
#                         question_part = part[1]
#                         question_part = question_part.strip()
#                         triple_queries.append(question_part)
#                     else:
#                         triple_part = part[1]
#                         question_part2 = part[2]
#                         question_part = triple_part.split("(", 1)[0]
#                         question_part = question_part.strip()
#                         question_part2 = question_part2.strip()
#                         triple_queries.append([question_part,question_part2])
#             return triple_queries
        
#         triple_queries[question_id] = extract_triple_queries(response)

# print("三元组翻译后的样例:",triple_queries)

# # 匹配到原数据集并新增字段triple_unit_queries
# triple_trans_dataset = decompose_question_dataset
#  # 确保每条记录都初始化了 'triple_unit_queries' 列
# if "triple_unit_queries" not in triple_trans_dataset["test"].column_names:
#     triple_trans_dataset["test"] = triple_trans_dataset["test"].add_column("triple_unit_queries", [None] * len(triple_trans_dataset["test"]))

# with tqdm(triple_queries.items(), desc="Mapping triple transaction to datastes") as pbar:
#     for question_id,each_question_triple_queries in pbar:
#         # 定义一个函数用于修改特定 id 的数据
#         def add_triple_queries(example):
#             if example["id"] == question_id:
#                 example["triple_unit_queries"] = each_question_triple_queries
#             return example
#         triple_trans_dataset["test"] = triple_trans_dataset["test"].map(add_triple_queries)

# print("三元组翻译后的数据集结构:",triple_trans_dataset)

# ###############################################################################################################

# # step 4
# # 剪枝，根据每个user queries找到与其相似的top k个triple unit queries，需要融合子图（去重复），并将对应的三元组加入到pruned_graph中，字段如下:
# # id
# # question
# # user_queries
# # answer
# # q_entity
# # a_entity
# # graph
# # pruned_graph
# # triple_unit_queries
# # choices

# # 最终输出文件的名称命名为{dataset_name}_{llm}_{embedding_model_name}_{faiss}_{topk}

# ###############################################################################################################
# # TODO:是否还需要根据user unit query进行剪枝,因为之前已经使用了origin query进行剪枝了
 
# # pruning_llm = "sentence-transformers"

# # with tqdm(triple_trans_dataset["test"], desc="Using user queries search topk triple unit queries and corresponding triple") as pbar:
# #     for sample in pbar:
# #         user_queries = sample["user_queries"]
# #         triple_unit_queries = sample["triple_unit_queries"]
# #         # 需要拆解triple_unit_queries构建一个index的索引,key-value:triple index-question



# ###############################################################################################################