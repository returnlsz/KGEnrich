# 此脚本用于对pruned_graph进行strutral enrich
# 输入:
# dataset_name
# llm
# 输出字段如下:

###############################################################################################################
# id
# question
# user_queries
# answer
# q_entity
# a_entity
# graph
# pruned_graph
# triple_unit_queries
# filter_triples(每个index对应直接储存相关user query的list)
# structral_enrich_triples
# choices
###############################################################################################################

###############################################################################################################
# 流程：
# 先从filter_triples提取出对应的子图，以及每个三元组对应的queries
# 将提取出的子图使用数据结构进行存储,并识别其中的一跳子图和两跳子图,这里的数据结构可以用邻接表
# 构建prompt
# llm生成回答
###############################################################################################################

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
import re
from collections import defaultdict
import pandas as pd
from datasets import Dataset, DatasetDict
from src.graph.graph import Graph

def llm_structral_enrich(dataset_name=None,llm=None):
    # 加载路径
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/filter_triple_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_filter_triple.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    write_data_dir = f"preprocess_datasets/structral_enrich_dataset{dataset_name}_{llm}_structral_enrich.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # 存放分解问题后的结果
    structral_enrich_dataset = None
    finished_id = []

    # 检查文件是否存在,如果不存在,则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在,创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在,已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        structral_enrich_dataset = DatasetDict({
                "test": Dataset.from_dict({
                "id": "",
                "question": "",
                "user_queries":[],
                "answer": [],
                "q_entity": [],
                "a_entity": [],
                "graph": [],
                "pruned_graph": [],
                "choices": [],
                "triple_unit_queries":[],
                "structral_enrich_triples":[],
                "filter_triples":[]
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成LLM structral enrich任务")
        structral_enrich_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        for sample in structral_enrich_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################
    # 对数据集中的每个问题,筛选出与之相关的三元组的index
    ###############################################################################################################
    mode = 'structral_enrich'
    structral_enrich_prompt = {}

    # 构建prompt
    with tqdm(dataset['test'], desc="Building structral_enrich prompts") as pbar:
        for each_sample in pbar:
            pbar.set_postfix(current_question=each_sample["question"])
            # 跳过已经存在的sample
            if each_sample["id"] in finished_id:
                continue

            question_id = each_sample["id"]
            filter_triples = each_sample["filter_triples"]
            pruned_graph = each_sample["pruned_graph"]
            triple_question_dict= {}

            # 图数据结构
            my_graph = Graph()

            def get_triple_query(filter_triples,pruned_graph):
                triple_question_dict = {}
                for ind,question_list in enumerate(filter_triples):
                    if len(question_list)>0:
                        # 维护图结构
                        my_graph.add_triplet(list(s,p,o))
                        # 维护triple-question字典结构
                        s,p,o = pruned_graph[ind]
                        if tuple(s,p,o) in triple_question_dict:
                            for question in question_list:
                                if question not in triple_question_dict[tuple(s,p,o)]:
                                    triple_question_dict[tuple(s,p,o)].append(question)
                        else:
                            triple_question_dict[tuple(s,p,o)] = []
                            triple_question_dict[tuple(s,p,o)] = question_list
                return triple_question_dict
            
            triple_question_dict = get_triple_query(filter_triples,pruned_graph)

            def assemble_prompt(triple_question_dict, my_graph, topics):
                # Helper function to format a triple into a string
                def format_triple(triple):
                    return f"({triple[0]},{triple[1]},{triple[2]})"

                # Helper function to format a two-hop path into a string
                def format_two_hop_path(two_hop_path):
                    return "->".join([format_triple(triple) for triple in two_hop_path])

                # Start building the prompt
                prompt = "Input:\n"

                # Add the triple-question mappings
                for triple, questions in triple_question_dict.items():
                    triple_str = format_triple(triple)
                    question_str = "-".join(questions)
                    prompt += f"{triple_str}-{question_str}\n"

                # Add 1-hop paths
                one_hop_paths = my_graph.get_one_hop_paths(topics)
                prompt += "1-hop:\n"
                for path in one_hop_paths:
                    prompt += f"{format_triple(path)}\n"

                # Add 2-hop paths
                two_hop_paths = my_graph.get_two_hop_paths(topics)
                prompt += "2-hop:\n"
                for path in two_hop_paths:
                    prompt += f"{format_two_hop_path(path)}\n"

                return prompt
            
            each_prompt = assemble_prompt(triple_question_dict, my_graph,None)
            # 调用 PromptBuilder，传入 question
            prompt_builder = PromptBuilder(each_prompt,mode)
            structral_enrich_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # 调用llm翻译三元组
    llm_chat = llm_client()

    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 20  # 设置批次大小

    with tqdm(structral_enrich_prompt.items(),desc=f"Call {llm} for triple transaction") as pbar:
        for question_id,each_structral_enrich_prompt in pbar:
            pbar.set_postfix(current_question=question_id)
            response = llm_chat.response(each_structral_enrich_prompt,mode)

            def extract_step_4_triples(response):
                # Find the part after "step 4:"
                step_4_content = response.split("step 4:")[1].strip()
                
                # Use a regular expression to extract all the triples
                # Match triples of the form (x, y, z)
                triples = re.findall(r'$ ([^,]+),\s*([^,]+),\s*([^)]+) $', step_4_content)
                
                # Convert the triples into a list of lists
                triple_list = [list(triple) for triple in triples]
                
                return triple_list
            if response == None:
                new_added_triples = []
            else:
                new_added_triples = extract_step_4_triples(response)
            
            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "structral_enrich_triples": new_added_triples}
                batch_results.append(example_with_prediction)

            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(structral_enrich_dataset["test"]) == 0:  # 如果 structral_enrich_dataset["test"] 是空的
                    structral_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    structral_enrich_dataset["test"] = Dataset.from_dict({
                        key: structral_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in structral_enrich_dataset["test"].column_names
                    })

                # 写入到文件
                structral_enrich_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储
    
    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(structral_enrich_dataset["test"]) == 0:  # 如果 structral_enrich_dataset["test"] 是空的
            structral_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            structral_enrich_dataset["test"] = Dataset.from_dict({
                key: structral_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in structral_enrich_dataset["test"].column_names
            })
        structral_enrich_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的structral enrich任务!")
    print("feature enrich后的样例:",structral_enrich_dataset["test"])
