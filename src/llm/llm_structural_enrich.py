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
# structural_enrich_triples
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
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

def llm_structural_enrich_main(dataset_name=None,llm=None,resume_path=None):
    return asyncio.run(llm_structural_enrich(dataset_name,llm,resume_path))

async def llm_structural_enrich(dataset_name=None,llm=None,resume_path=None):
    # 加载路径
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/filter_triple_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_filter_triple.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)

    # 获取当前时间
    current_time = datetime.now()

    # 将时间格式化为字符串（如 "2023-10-10_14-30-00"）,将这里替换成上次未完成的parquet名称可以继续
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/structural_enrich_dataset/{dataset_name}_{llm}_structural_enrich_{time_str}.parquet"
    else:
        write_data_dir = resume_path

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    # write_data_dir = f"preprocess_datasets/structural_enrich_dataset/{dataset_name}_{llm}_structural_enrich.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # 存放分解问题后的结果
    structural_enrich_dataset = None
    finished_id = []

    # 检查文件是否存在,如果不存在,则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在,创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在,已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        structural_enrich_dataset = DatasetDict({
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
                "structural_enrich_triples":[],
                "filter_triples":[]
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成LLM structural enrich任务")
        structural_enrich_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        # if len(structural_enrich_dataset["test"]) != 0:
        for sample in structural_enrich_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################
    # 对数据集中的每个问题,筛选出与之相关的三元组的index
    ###############################################################################################################
    mode = 'structural_enrich'
    structural_enrich_prompt = {}
    my_graph = {}

    # 构建prompt
    with tqdm(dataset['test'], desc="Building structural_enrich prompts") as pbar:
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
            this_graph = Graph()

            def get_triple_query(filter_triples,pruned_graph):
                triple_question_dict = {}
                for ind,question_list in enumerate(filter_triples):
                    if question_list != None and len(question_list) > 0:
                        # 维护图结构
                        s,p,o = pruned_graph[ind]
                        this_graph.add_triplet((s,p,o))
                        # 维护triple-question字典结构
                        if (s,p,o) in triple_question_dict:
                            for question in question_list:
                                if question not in triple_question_dict[(s,p,o)]:
                                    triple_question_dict[(s,p,o)].append(question)
                        else:
                            triple_question_dict[(s,p,o)] = []
                            triple_question_dict[(s,p,o)] = question_list
                return triple_question_dict
            
            triple_question_dict = get_triple_query(filter_triples,pruned_graph)
            if len(triple_question_dict) == 0:
                continue

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
                one_hop_paths = my_graph[question_id].get_one_hop_paths(topics)
                prompt += "1-hop:\n"
                for path in one_hop_paths:
                    prompt += f"{format_triple(path)}\n"

                # Add 2-hop paths
                two_hop_paths = my_graph[question_id].get_two_hop_paths(topics)
                prompt += "2-hop:\n"
                for path in two_hop_paths:
                    prompt += f"{format_two_hop_path(path)}\n"

                return prompt
            
            # 每个question对应一个graph
            my_graph[each_sample["id"]] = this_graph
            # 获取topics
            topics = each_sample["q_entity"]
            each_prompt = assemble_prompt(triple_question_dict, my_graph,topics)
            # 调用 PromptBuilder，传入 question
            prompt_builder = PromptBuilder(each_prompt,mode)
            structural_enrich_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # 调用llm翻译三元组
    llm_chat = llm_client()

    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 20  # 设置批次大小

    # 定义异步函数，处理单个请求
    async def process_single_query(llm_chat, question_id, each_triples_trans_prompt, mode):
        response = await llm_chat.response(each_triples_trans_prompt, mode)
        return question_id, response

    # 定义异步函数，处理多个请求并按完成顺序处理结果
    tasks = [
        process_single_query(llm_chat, question_id, each_structural_enrich_prompt, mode)
        for question_id, each_structural_enrich_prompt in structural_enrich_prompt.items()
    ]
    
    # 使用 tqdm_asyncio 显示进度条
    with tqdm_asyncio(desc=f"Call {llm} for structural_enrich", total=len(structural_enrich_prompt)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # 等待某个任务完成
            question_id, response = result  # 从返回值解构
            pbar.update(1)  # 更新进度条

    # with tqdm(structural_enrich_prompt.items(),desc=f"Call {llm} for structural_enrich") as pbar:
    #     for question_id,each_structural_enrich_prompt in pbar:
    #         pbar.set_postfix(current_question=question_id)
    #         response = llm_chat.response(each_structural_enrich_prompt,mode)

            def extract_step_4_triples(response):
                # Find the part after "step 4:"
                # 1. 提取 "Final output:" 后的内容
                final_output_match = re.search(r"Final output:\n(.*)", response, re.DOTALL)
                # step_4_content = response.split("step 4:")[1].strip()

                if not final_output_match:
                    step_4_content = response.strip()
                else:
                    step_4_content = final_output_match.group(1).strip()

                step_4_content = step_4_content.replace('{/thought}', '').replace('{thought}', '')
                step_4_content = step_4_content.replace('{demonstrations}', '').replace('{/demonstrations}', '')
                step_4_content = step_4_content.replace('```', '')

                # Use a regular expression to extract all the triples
                # Match triples of the form (x, y, z)
                # triples = re.findall(r'$ ([^,]+),\s*([^,]+),\s*([^)]+) $', step_4_content)
                triples = step_4_content.split("\n")
                cleaned_triples = [item for item in triples if item != '' and len(item) == 3]
                cleaned_triples = [
                    item.strip('()').split(',')  # 去掉括号并按 ", " 分隔字符串
                    for item in cleaned_triples
                ]
                return cleaned_triples
            
            if response == None:
                new_added_triples = []
            else:
                new_added_triples = extract_step_4_triples(response)
            
            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "structural_enrich_triples": new_added_triples}
                batch_results.append(example_with_prediction)

            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(structural_enrich_dataset["test"]) == 0:  # 如果 structural_enrich_dataset["test"] 是空的
                    structural_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    structural_enrich_dataset["test"] = Dataset.from_dict({
                        key: structural_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in structural_enrich_dataset["test"].column_names
                    })

                # 写入到文件
                structural_enrich_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储
    
    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(structural_enrich_dataset["test"]) == 0:  # 如果 structural_enrich_dataset["test"] 是空的
            structural_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            structural_enrich_dataset["test"] = Dataset.from_dict({
                key: structural_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in structural_enrich_dataset["test"].column_names
            })
        structural_enrich_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的structural enrich任务!")
    print("structural enrich后的样例:",structural_enrich_dataset["test"])
