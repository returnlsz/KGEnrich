# 此脚本用于enrich子图:feature enrich
# input:
# dataset name
# llm
# output:
# 见以下字段
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
# feature_enrich_triples
# choices
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

def llm_feature_enrich(dataset_name=None,llm=None):
    # 加载路径
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/filter_triple_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_filter_triple.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    write_data_dir = f"preprocess_datasets/feature_enrich_datasets/{dataset_name}_{llm}_feature_enrich.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # 存放分解问题后的结果
    feature_enrich_dataset = None
    finished_id = []

    # 检查文件是否存在,如果不存在,则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在,创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在,已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        feature_enrich_dataset = DatasetDict({
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
                "feature_enrich_triples":[],
                "filter_triples":[]
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成LLM feature enrich任务")
        feature_enrich_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        for sample in feature_enrich_dataset["test"]:
            finished_id.append(sample["id"])
    
    ###############################################################################################################
    # 构建prompt
    ###############################################################################################################

    mode = "feature_enrich"
    feature_enrich_prompt = {}

    # 构建prompt
    with tqdm(dataset['test'], desc="Building feature enrich prompts") as pbar:
        for each_sample in pbar:
            # 已经存在的就跳过
            if each_sample["id"] in finished_id:
                continue
            question_id = each_sample["id"]
            filter_triples = each_sample["filter_triples"]
            pruned_graph = each_sample["pruned_graph"]

            # 提取filter_triples中相关实体,构造对应的数据结构:key(entity)-value:"user_queries":[question list],"triples":[triple list]
            # 函数get_related_questions_triples
            def get_related_questions_triples(filter_triples,pruned_graph):
                entity_question_triple_dict = {}
                for ind,question_list in enumerate(filter_triples):
                    if len(question_list) > 0:
                        corresponding_triple = pruned_graph[ind]
                        s,p,o = corresponding_triple
                        if s in entity_question_triple_dict:
                            entity_question_triple_dict[s]["triples"].append(corresponding_triple)
                            # 遍历 question_list 并只添加不存在的 user_query
                            for user_query in question_list:
                                if user_query not in entity_question_triple_dict[s]["user_queries"]:
                                    entity_question_triple_dict[s]["user_queries"].append(user_query)
                        else:
                            entity_question_triple_dict[s]["triples"] = []
                            entity_question_triple_dict[s]["user_queries"] = []
                        if o in entity_question_triple_dict:
                            entity_question_triple_dict[o]["triples"].append(corresponding_triple)
                            # 遍历 question_list 并只添加不存在的 user_query
                            for user_query in question_list:
                                if user_query not in entity_question_triple_dict[o]["user_queries"]:
                                    entity_question_triple_dict[o]["user_queries"].append(user_query)
                        else:
                            entity_question_triple_dict[o]["triples"] = []
                            entity_question_triple_dict[o]["user_queries"] = []
                return entity_question_triple_dict

            # 根据数据结构装填prompt
            entity_question_triple_dict = {}
            entity_question_triple_dict = get_related_questions_triples(filter_triples,pruned_graph)
            
            # 组装prompt
            def generate_prompt(entity_question_triple_dict):
                # 初始化结果字符串
                prompt_result = "Input:\nentity List:\n"
                
                # 遍历字典中的每个实体
                for entity, value in entity_question_triple_dict.items():
                    # 获取当前实体的 questions 和 triples
                    user_queries = value.get("user_queries", [])
                    triples = value.get("triples", [])
                    
                    # 如果列表为空，填写为 None
                    user_queries_str = "-".join(user_queries) if user_queries else "None"
                    triples_str = "-".join(["-".join(triple) for triple in triples]) if triples else "None"

                    # 填充模板
                    entity_block = f"[${entity}$ context]\nrelavent triple(s):{triples_str}\nrelavent user query(ies):{user_queries_str}\n[/${entity}$ context]\n"
                    prompt_result = prompt_result + entity_block
                return prompt_result
            each_prompt = generate_prompt(entity_question_triple_dict)
            # 调用 PromptBuilder，传入 question
            prompt_builder = PromptBuilder(each_prompt,mode)
            feature_enrich_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # 调用llm翻译三元组
    llm_chat = llm_client()

    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 20  # 设置批次大小

    with tqdm(feature_enrich_prompt.items(),desc=f"Call {llm} for triple transaction") as pbar:
        for question_id,each_feature_enrich_prompt in pbar:
            pbar.set_postfix(current_question=question_id)
            response = llm_chat.response(each_feature_enrich_prompt,mode)

            def extract_triples_from_response(response):
                """
                从 response 字符串中提取 {result} 和 {/result} 之间的内容，并解析为一个三元组列表。
                """
                # 使用正则表达式提取 {result} 和 {/result} 之间的内容
                match = re.search(r"\{result\}([\s\S]*?)\{/result\}", response)
                if not match:
                    # 如果没有匹配到 {result} 块，返回空列表
                    return []

                # 提取内容部分（即 {result} 和 {/result} 之间的内容）
                content = match.group(1).strip()

                # 解析内容，将每个三元组字符串转换为长度为 3 的列表
                triples = []
                for line in content.splitlines():
                    # 去掉多余的空白字符
                    line = line.strip()
                    # 跳过空行
                    if not line:
                        continue
                    # 确保行以括号开头和结尾（这是三元组的格式）
                    if line.startswith("(") and line.endswith(")"):
                        # 移除括号并按逗号分割
                        triple_parts = line[1:-1].split(", ")
                        if len(triple_parts) == 3:
                            triples.append(triple_parts)  # 确保三元组由三个部分组成

                return triples

            processed_answer = extract_triples_from_response(response)
            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "feature_enrich_triples": processed_answer}
                batch_results.append(example_with_prediction)

            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(feature_enrich_dataset["test"]) == 0:  # 如果 feature_enrich_dataset["test"] 是空的
                    feature_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    feature_enrich_dataset["test"] = Dataset.from_dict({
                        key: feature_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in feature_enrich_dataset["test"].column_names
                    })

                # 写入到文件
                feature_enrich_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储
                
    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(feature_enrich_dataset["test"]) == 0:  # 如果 feature_enrich_dataset["test"] 是空的
            feature_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            feature_enrich_dataset["test"] = Dataset.from_dict({
                key: feature_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in feature_enrich_dataset["test"].column_names
            })
        feature_enrich_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的feature enrich任务!")
    print("feature enrich后的样例:",feature_enrich_dataset["test"])

