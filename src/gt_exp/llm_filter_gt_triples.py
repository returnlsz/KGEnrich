# 此脚本用于用llm从子图中筛选出与推理所需的关键三元组,或补充关键三元组

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
import asyncio
from tqdm.asyncio import tqdm_asyncio

###############################################################################################################
# id
# question
# answer
# q_entity
# a_entity
# graph
# gt_triples
# choices
###############################################################################################################


def llm_filter_gt_triples_main(dataset_name=None,llm=None,initial_pruning_llm = "sentence-transformers",initial_pruning_topk = 750):
    return asyncio.run(llm_filter_gt_triples(dataset_name,llm,initial_pruning_llm,initial_pruning_topk))

async def llm_filter_gt_triples(dataset_name=None,llm=None,initial_pruning_llm = "sentence-transformers",initial_pruning_topk = 750):
    # 加载路径
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/initial_pruning_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{initial_pruning_llm}_{initial_pruning_topk}_initial_pruning.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    write_data_dir = f"preprocess_datasets/llm_filter_gt_triples_datasets/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_llm_filter_gt_triples.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # 存放分解问题后的结果
    llm_filter_gt_triples_dataset = None
    finished_id = []

    # 检查文件是否存在,如果不存在,则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在,创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在,已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        llm_filter_gt_triples_dataset = DatasetDict({
                "test": Dataset.from_dict({
                "id": "",
                "question": "",
                "answer": [],
                "q_entity": [],
                "a_entity": [],
                "graph": [],
                "choices": [],
                "gt_triples":[]
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成llm_filter_gt_triples任务")
        llm_filter_gt_triples_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        # if len(structural_enrich_dataset["test"]) != 0:
        for sample in llm_filter_gt_triples_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################
    # 构建promot
    ###############################################################################################################
    
    mode = 'filter_gt_triples'
    filter_gt_triples_prompt = {}

    # 构建prompt
    with tqdm(dataset['test'], desc="Building filter_gt_triples prompts") as pbar:
        for each_sample in pbar:
            pbar.set_postfix(current_question=each_sample["question"])
            # 跳过已经存在的sample
            if each_sample["id"] in finished_id:
                continue

            question_id = each_sample["id"]
            question = each_sample["question"]
            answers = each_sample["answer"]
            topics = each_sample["q_entity"]
            sub_graph = each_sample["graph"]

            def assemble_prompt(question,answers,topics,sub_graph):

                answers_str = f"[{','.join(answers)}]"
                # 将主题实体列表填充为字符串格式
                topics_str = f"[{','.join(topics)}]"
                # 将三元组填充为字符串格式
                triples_str = "\n".join([f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in sub_graph])
                
                # 填充模板
                result = f"""Input:
                    question:
                    {question}
                    answer:
                    {answers_str}
                    topic entity:
                    {topics_str}
                    triples:
                    {triples_str}"""
                        
                return result
            each_prompt = assemble_prompt(question,answers,topics,sub_graph)
            # 调用 PromptBuilder，传入 question
            prompt_builder = PromptBuilder(each_prompt,mode)
            filter_gt_triples_prompt[each_sample["id"]] = prompt_builder.build_prompt()
        
    ###############################################################################################################
    # 调用llm进行筛选
    ###############################################################################################################

    # 调用llm翻译三元组
    llm_chat = llm_client()

    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 20  # 设置批次大小

    # 定义异步函数，处理单个请求
    async def process_single_query(llm_chat, question_id, each_filter_gt_triples_prompt, mode):
        response = await llm_chat.response(each_filter_gt_triples_prompt, mode)
        return question_id, response

    # 定义异步函数，处理多个请求并按完成顺序处理结果
    tasks = [
        process_single_query(llm_chat, question_id, each_filter_gt_triples_prompt, mode)
        for question_id, each_filter_gt_triples_prompt in filter_gt_triples_prompt.items()
    ]
    
    # 使用 tqdm_asyncio 显示进度条
    with tqdm_asyncio(desc=f"Call {llm} for filter_gt_triples", total=len(filter_gt_triples_prompt)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # 等待某个任务完成
            question_id, response = result  # 从返回值解构
            pbar.update(1)  # 更新进度条

            def extract_triples(response):
                # # 定位 “### Related or newly added triples ###” 的起始位置
                # start_marker = "### Related or newly added triples ###"
                # start_index = response.find(start_marker)
                
                # # 若未找到起始标记，返回空列表
                # if start_index == -1:
                #     return []
                
                # # 提取从标记开始到最后的内容
                # content = response[start_index + len(start_marker):].strip()

                final_output_match = re.search(r"### Related or newly added triples ###\n(.*)", response, re.DOTALL)
                if not final_output_match:
                    content = response.strip()
                else:
                    content = final_output_match.group(1).strip()
                    
                content = content.replace('```', '')
                content = content.replace('{/thoughts & reason}', '').replace('{thoughts & reason}', '')
                content = content.replace('{demonstrations}', '').replace('{/demonstrations}', '')

                # 分割内容为每行，并提取三元组
                triples = []
                for line in content.split("\n"):
                    # 移除前后多余空格，跳过空行
                    line = line.strip()
                    if line:
                        line = line.replace('(', '').replace(')', '')
                        # 将三元组拆分成 (h, r, t)
                        triple = line.split(", ")
                        # if len(triple) == 3:  # 确保是长度为3的三元组
                        triples.append(triple)
                
                return triples
            
            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            # match_index是一个list,每个元素是一个三元组,每个三元组是一个长度为3的list,
            processed_answer = extract_triples(response)
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "gt_triples": processed_answer}
                batch_results.append(example_with_prediction)
            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(llm_filter_gt_triples_dataset["test"]) == 0:  # 如果 llm_filter_gt_triples_dataset["test"] 是空的
                    llm_filter_gt_triples_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    llm_filter_gt_triples_dataset["test"] = Dataset.from_dict({
                        key: llm_filter_gt_triples_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in llm_filter_gt_triples_dataset["test"].column_names
                    })

                # 写入到文件
                llm_filter_gt_triples_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储

    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(llm_filter_gt_triples_dataset["test"]) == 0:  # 如果 llm_filter_gt_triples_dataset["test"] 是空的
            llm_filter_gt_triples_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            llm_filter_gt_triples_dataset["test"] = Dataset.from_dict({
                key: llm_filter_gt_triples_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in llm_filter_gt_triples_dataset["test"].column_names
            })
        llm_filter_gt_triples_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的LLM筛选GT三元组任务!")
    print("三元组翻译后的样例:",llm_filter_gt_triples_dataset["test"])



