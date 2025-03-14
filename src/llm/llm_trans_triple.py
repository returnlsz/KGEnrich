import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm

def llm_trans_triple(dataset_name=None,llm=None):
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
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/question_decompose_datasets'

    decompose_question_dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_question_decompose.parquet.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 打印数据集的信息
    print(decompose_question_dataset)

    ###############################################################################################################

    ###############################################################################################################

    # step 3.1
    # 需要先对graph进行预剪枝,否则后续进行三元组翻译的时候上下文会非常长,导致llm回答时间特别长,这里直接用embedding的方法

    # 最终输出文件的名称命名为{dataset_name}_{llm}_{initial_pruning}

    ###############################################################################################################
    # TODO:


    ###############################################################################################################

    # step 3.2
    # 把graph中的每个三元组翻译成query,并储存起来,字段如下：
    # id
    # question
    # user_queries
    # answer
    # q_entity
    # a_entity
    # graph
    # triple_unit_queries
    # choices

    # 最终输出文件的名称命名为{dataset_name}_{llm}_{triple_unit_queries}

    ###############################################################################################################

    # 建立三元组翻译的prompt
    triples_trans_prompt = {}
    with tqdm(decompose_question_dataset['test'], desc="Building triple transaction prompts") as pbar:
        for each_sample in pbar:
            pbar.set_postfix(current_question=each_sample["question"])
            sub_graph = each_sample["graph"]
            
            triple_text_list = []
            for triple in sub_graph:
                if isinstance(triple, list) and len(triple) == 3:  # 确保每个三元组是长度为3的列表
                    triple_text_list.append(f"({triple[0]}, {triple[1]}, {triple[2]})")
            
            # 用换行符连接每个三元组文本
            sub_graph_text = ",".join(triple_text_list)

            mode = "triples_trans"
            # 调用 PromptBuilder，传入 question
            prompt_builder = PromptBuilder(sub_graph_text,mode)
            triples_trans_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # 调用llm翻译三元组
    llm_chat = llm_client() 
    mode = "triples_trans"
    triple_queries = {}
    with tqdm(triples_trans_prompt.items(),desc="Call LLM for triple transaction") as pbar:
        for question_id,each_triples_trans_prompt in pbar:
            pbar.set_postfix(current_question=question_id)
            response = llm_chat.response(each_triples_trans_prompt,mode)

            # 示例response文本
            # response = """Natural Language Question: 
            # (Beijing,located in,?):Which country does Beijing locate? (?,located in,China):What cities or places are located in China?
            # (Eiffel Tower, located in, ?):In which city is the Eiffel Tower located? (?, located in, Paris):What landmarks or places are located in Paris?
            # (Apple, founded by, ?):Who founded Apple? (?, founded by, Steve Jobs):Which companies or organizations were founded by Steve Jobs?
            # (Python, created by, ?):Who created Python? (?, created by, Guido van Rossum):What programming languages or projects were created by Guido van Rossum?
            # (Tesla, CEO of, ?):Who is the CEO of Tesla? (?, CEO of, Elon Musk):Which companies or organizations have Elon Musk as their CEO?
            # """
            def extract_triple_queries(response):
                # 初始化最终结果列表
                triple_queries = []
                
                # 首先查找是否有"Natural Language Question: "
                start_index = response.find("Natural Language Question: ")
                if start_index != -1:
                    # 如果存在，提取从"Natural Language Question: "开始的内容
                    response = response[start_index + len("Natural Language Question: "):]
                
                # 按行分割内容
                lines = response.splitlines()
                
                for line in lines:
                    # 跳过空行
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 找到三元组部分和对应问题的分隔符 ":"
                    if ":" in line:
                        part = line.split(":", 2)
                        if len(part) == 2:
                            triple_part = part[0]
                            question_part = part[1]
                            question_part = question_part.strip()
                            triple_queries.append(question_part)
                        else:
                            triple_part = part[1]
                            question_part2 = part[2]
                            question_part = triple_part.split("(", 1)[0]
                            question_part = question_part.strip()
                            question_part2 = question_part2.strip()
                            triple_queries.append([question_part,question_part2])
                return triple_queries
            
            triple_queries[question_id] = extract_triple_queries(response)

    print("三元组翻译后的样例:",triple_queries)

    # 匹配到原数据集并新增字段triple_unit_queries
    triple_trans_dataset = decompose_question_dataset
    # 确保每条记录都初始化了 'triple_unit_queries' 列
    if "triple_unit_queries" not in triple_trans_dataset["test"].column_names:
        triple_trans_dataset["test"] = triple_trans_dataset["test"].add_column("triple_unit_queries", [None] * len(triple_trans_dataset["test"]))

    with tqdm(triple_queries.items(), desc="Mapping triple transaction to datastes") as pbar:
        for question_id,each_question_triple_queries in pbar:
            # 定义一个函数用于修改特定 id 的数据
            def add_triple_queries(example):
                if example["id"] == question_id:
                    example["triple_unit_queries"] = each_question_triple_queries
                return example
            triple_trans_dataset["test"] = triple_trans_dataset["test"].map(add_triple_queries)

    print("三元组翻译后的数据集结构:",triple_trans_dataset)