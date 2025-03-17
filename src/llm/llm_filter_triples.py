# 此脚本用于筛选与问题强相关的三元组
# 输入:
# dataset,其中dataset需要含有graph以及user_queries以及triple_unit_queries字段,涉及字段
# graph子图,是一个list
# user_queries,是一个list
# triple_unit_queries,每个三元组对应的query,是一个list
# 输出:
# 输出与该问题相关的三元组的index

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


def llm_filter_triples(dataset_name=None,llm=None):
    # 加载路径
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/triple_trans_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_triple_trans.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    write_data_dir = f"preprocess_datasets/filter_triple_datasets/{dataset_name}_{llm}_filter_triple.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # 存放分解问题后的结果
    filter_triple_dataset = None
    finished_id = []

    # 检查文件是否存在,如果不存在,则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在,创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在,已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        filter_triple_dataset = DatasetDict({
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
                "filter_triples":[]
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成LLM筛选三元组任务")
        filter_triple_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        for sample in filter_triple_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################
    # 对数据集中的每个问题,筛选出与之相关的三元组的index
    ###############################################################################################################
    mode = 'filter_triples'
    filter_triples_prompt = {}

    # 构建prompt
    with tqdm(dataset['test'], desc="Building filter triples prompts") as pbar:
        for each_sample in pbar:
            pbar.set_postfix(current_question=each_sample["question"])
            # 跳过已经存在的sample
            if each_sample["id"] in finished_id:
                continue
            user_queries = each_sample["user_queries"]
            sub_graph = each_sample["graph"]
            triple_unit_queries = each_sample["triple_unit_queries"]
            
            # 以下是测试数据
            user_queries = [
                "What is the location that appointed Michelle Bachelet to a governmental position?",
                "Who is Michelle Bachelet?",
                "What governmental position was Michelle Bachelet appointed to?",
                "Where was Michelle Bachelet appointed to this position?",
                "What language is spoken in this location?"
            ]

            sub_graph = [
                ["Michelle Bachelet", "people.person.nationality", "Chile"],
                ["Chile", "language.human_language.countries_spoken_in", "Spanish Language"]
            ]

            triple_unit_queries = [
                ["What is Michelle Bachelet's nationality?", "Which people have Chilean nationality?"],
                ["What language is spoken in Chile?", "Which countries speak the Spanish Language?"]
            ]

            def assemble_prompt(user_queries, sub_graph, triple_unit_queries):
                # 初始化模板
                template = "Input:\nuser unit queries:\n"
                
                # 填充 user_queries
                user_queries_section = "\n".join(user_queries)
                template += user_queries_section + "\ntriple unit queries:\n"
                
                # 填充 triple_unit_queries
                triple_unit_queries_section = ""
                for idx, triple in enumerate(sub_graph):
                    triple_str = f"({triple[0]},{triple[1]},{triple[2]})"
                    if triple_unit_queries[idx]:  # 如果该三元组有对应的 query
                        queries = "<SEP>".join(triple_unit_queries[idx])
                        triple_unit_queries_section += f"{triple_str}<SEP>{queries}\n"
                    else:  # 如果该三元组没有对应的 query
                        triple_unit_queries_section += f"{triple_str}<SEP>\n"
                
                # 合并生成最终结果
                template += triple_unit_queries_section.strip()
                return template

            each_input = assemble_prompt(user_queries,sub_graph,triple_unit_queries)
            
            # 调用 PromptBuilder,传入 question
            prompt_builder = PromptBuilder(each_input,mode)
            filter_triples_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # 调用llm筛选三元组
    llm_chat = llm_client() 
    
    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 30  # 设置批次大小

    with tqdm(filter_triples_prompt.items(),desc="Call LLM for filter triple") as pbar:
        for question_id,each_filetr_triples_prompt in pbar:
            pbar.set_postfix(current_question=question_id)
            response = llm_chat.response(each_filetr_triples_prompt,mode)

            ###############################################################################################################
            # 回答格式
            # question<SEP>None
            # question<SEP>triple1<SEP>triple2
            ###############################################################################################################

            # 筛选Final output:后面的内容

            def extract_and_process_response(response):
                # 1. 提取 "Final output:" 后的内容
                final_output_match = re.search(r"Final output:\n(.*)", response, re.DOTALL)
                if not final_output_match:
                    return {}
                content = final_output_match.group(1).strip()
                
                # 2. 初始化结果字典
                triple_to_questions = defaultdict(list)
                
                # 3. 解析content
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        # question-部分
                        question, triples_str = line.split("<SEP>", 1)
                        question = question.strip()
                        
                        # triples部分
                        triples = []
                        if triples_str.strip() != "None":
                            triples = re.findall(r"$ (.*?) $", triples_str)
                            triples = [tuple(triple.split(", ")) for triple in triples]
                        
                        # 将问题添加到每个三元组的列表中
                        if triples:
                            for triple in triples:
                                triple_to_questions[triple].append(question)
                        else:
                            # 如果没有三元组,则添加空列表对应的key
                            triple_to_questions[()].append(question)
                
                # 4. 转换为普通dict并返回
                return dict(triple_to_questions)
            
            # 测试代码
            response = """
            Final output:
            Who was the champion of the 1931 World Series Championship?<SEP>(1931 World Series, sports.sports_team.championships, St. Louis Cardinals)
            What is the World Series Championship?<SEP>None
            Who won the World Series in 1931?<SEP>(1931 World Series, sports.sports_team.championships, St. Louis Cardinals)
            Where does this champion team play?<SEP>(St. Louis Cardinals, plays at, Busch Stadium)<SEP>(St. Louis Cardinals, has arena, Busch Stadium)
            What is the name of the stadium associated with this team?<SEP>(St. Louis Cardinals, sports.sports_team.arena_stadium, Busch Stadium)<SEP>(St. Louis Cardinals, has arena, Busch Stadium)
            Was this stadium their home stadium in 1931?<SEP>(St. Louis Cardinals, sports.sports_team.arena_stadium, Roger Dean Stadium)<SEP>(St. Louis Cardinals, home ground of, Busch Stadium)<SEP>(St. Louis Cardinals, home ground of, Roger Dean Stadium)
            """
            # 数据结构:key(tuple[s,p,o])-value(list[question]),s,p,o以及question均为str,list[question]可能为空[]
            triple_question_dict = extract_and_process_response(response)

            # 将triple_question_dict中的key与dataset中的pruned_graph字段进行匹配,找到匹配的index,然后到filter_triples中对应的index加上该question list
            # 函数write_triple_question
            def write_triple_question(triple_question_dict, example):
                """
                该函数将 triple_question_dict 中的三元组与 example["pruned_graph"] 的三元组进行匹配,
                并生成 match_index 和 no_match_dict。

                :param triple_question_dict: dict, 数据结构为 key(tuple[s, p, o]) - value(list[question])
                :param example: dict, 包含 "pruned_graph" 键,其值为 list,每个元素是一个长度为 3 的 list 表示三元组
                :return: match_index: list,与 example["pruned_graph"] 等长
                        no_match_dict: dict, key(tuple[s, p, o]) - value(list[question])
                """
                # 获取 pruned_graph
                pruned_graph = example["pruned_graph"]

                # 初始化 match_index,长度等于 pruned_graph,初始值全为 None
                match_index = [None] * len(pruned_graph)

                # 初始化 no_match_dict
                no_match_dict = {}

                # 遍历 triple_question_dict 中的每个三元组及其对应的 question 列表
                for triple, questions in triple_question_dict.items():
                    # 初始化匹配标志
                    matched = False

                    # 遍历 pruned_graph 及其索引,查找匹配的三元组
                    for index, graph_triple in enumerate(pruned_graph):
                        # 如果三元组匹配
                        if triple == tuple(graph_triple):  # 将 graph_triple 转为 tuple 以便比较
                            # 在 match_index 对应位置记录 questions
                            match_index[index] = questions
                            matched = True
                            break  # 匹配成功则停止当前三元组的匹配

                    # 如果没有匹配到任何三元组,将其加入 no_match_dict
                    if not matched:
                        no_match_dict[triple] = questions

                # 返回 match_index 和 no_match_dict
                return match_index, no_match_dict

            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            no_match_dict = {}
            # match_index是一个list,每个元素是一个三元组,每个三元组是一个长度为3的list,
            processed_answer,no_match_dict = write_triple_question(triple_question_dict,example)
            
            ###############################################################################################################
            # 假设一定能够匹配上,先不做异常处理
            if len(no_match_dict) > 0:
                print("存在没有匹配上的三元组,请注意!!!")
                pass
            ###############################################################################################################

            
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "filter_triples": processed_answer}
                batch_results.append(example_with_prediction)
            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(filter_triple_dataset["test"]) == 0:  # 如果 filter_triple_dataset["test"] 是空的
                    filter_triple_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    filter_triple_dataset["test"] = Dataset.from_dict({
                        key: filter_triple_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in filter_triple_dataset["test"].column_names
                    })

                # 写入到文件
                filter_triple_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储
                
    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(filter_triple_dataset["test"]) == 0:  # 如果 filter_triple_dataset["test"] 是空的
            filter_triple_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            filter_triple_dataset["test"] = Dataset.from_dict({
                key: filter_triple_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in filter_triple_dataset["test"].column_names
            })
        filter_triple_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的LLM筛选三元组任务!")
    print("三元组翻译后的样例:",filter_triple_dataset["test"])

            
            



