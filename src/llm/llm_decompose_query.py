import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm

def llm_decompose_query(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=100):
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
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/initial_pruning_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{initial_pruning_llm}_{initial_pruning_topk}_initial_pruning.parquet.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(dataset)

    ###############################################################################################################

    # step 2:
    # 把question分解成user queries，并储存起来，字段如下:
    # id
    # question
    # user_queries
    # answer
    # q_entity
    # a_entity
    # graph
    # triple_unit_queries
    # choices

    # 最终输出文件的名称命名为{dataset_name}_{llm}_{question_decompose}

    ###############################################################################################################

    # 用于提取 question 并生成 result_prompt
    def process_dataset(dataset):
        # 存储结果
        result_prompts = {}
        
        # 遍历数据集中的每一行
        for example in tqdm(dataset['test'],desc="Building question decompose prompts"):
            # 提取 question 字段
            question_id = example['id']
            question = example['question']
            mode = "question_decompose"
            # 调用 PromptBuilder，传入 question
            prompt_builder = PromptBuilder(question,mode)
            
            # 获取生成的 result_prompt
            result_prompts[question_id] = prompt_builder.build_prompt()

            # print(result_prompts)
            # return result_prompts
        
        return result_prompts

    # 调用函数处理数据集
    question_decompose_result_prompts = process_dataset(dataset)
    llm_chat = llm_client() 
    mode = "question_decompose"
    # sub_queries = llm_chat.response(question_decompose_result_prompts[0],mode)
    # print(sub_queries)

    # 存放分解问题后的结果
    decompose_question_dataset = dataset

    # response预处理
    def extract_questions(sub_queries):
        """
        从给定的sub_queries文本中提取所有问题,并返回一个格式化后的问题列表。

        :param sub_queries: 包含问题和子问题的树形结构的文本
        :return: 格式化后的问题列表
        """
        # 移除所有的"-", "--", "---"和换行符
        cleaned_text = sub_queries.replace("-", "").replace("--", "").replace("---", "").replace("\n", " ")
        
        # 分割文本以提取所有问题
        questions = [question.strip() for question in cleaned_text.split("?") if question.strip()]
        
        return questions


    with tqdm(question_decompose_result_prompts.items(), desc="Call LLM for question decomposing and Mapping to datasets") as pbar:
        for question_id, question in pbar:
            mode = "question_decompose"
            sub_queries = llm_chat.response(question, mode)

            pbar.set_postfix(current_question_id=question_id)

            # 对response进行预处理
            question_list = extract_questions(sub_queries)

            # 确保每条记录都初始化了 'user_queries' 列
            if "user_queries" not in decompose_question_dataset["test"].column_names:
                decompose_question_dataset["test"] = decompose_question_dataset["test"].add_column("user_queries", [None] * len(decompose_question_dataset["test"]))

            # 定义一个函数用于修改特定 id 的数据
            def add_user_queries(example):
                if example["id"] == question_id:
                    example["user_queries"] = question_list
                return example
            decompose_question_dataset["test"] = decompose_question_dataset["test"].map(add_user_queries)

    print("问题分解后的数据集样例:",decompose_question_dataset['test'][0])