import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

def llm_trans_triple_main(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,llm_pruning_top_k=100,resume_path=None):
    return asyncio.run(llm_trans_triple(dataset_name,llm,initial_pruning_llm,initial_pruning_topk,llm_pruning_top_k,resume_path))

async def llm_trans_triple(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,llm_pruning_top_k=100,resume_path=None):
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
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_dataset'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_{llm_pruning_top_k}_llm_pruning.parquet'})

    # dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_question_decompose.parquet'})
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'
    # 打印数据集的信息
    print(dataset)

    # 获取当前时间
    current_time = datetime.now()
    
    # 将时间格式化为字符串（如 "2023-10-10_14-30-00"）,将这里替换成上次未完成的parquet名称可以继续
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/triple_trans_datasets/{dataset_name}_{llm}_triple_trans_{time_str}.parquet"
    else:
        write_data_dir = resume_path

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    # write_data_dir = f"preprocess_datasets/triple_trans_datasets/{dataset_name}_{llm}_triple_trans.parquet"
    
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    triple_trans_dataset = None
    finished_id = []

    # 检查文件是否存在，如果不存在，则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在，创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在，已创建新的空文件：{write_data_dir}")
        # 初始化dataset
        triple_trans_dataset = DatasetDict({
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
                "triple_unit_queries":[]
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成翻译三元组任务")
        triple_trans_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        for sample in triple_trans_dataset["test"]:
            finished_id.append(sample["id"])


    ###############################################################################################################


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
    # pruned_graph
    # triple_unit_queries
    # choices

    # 最终输出文件的名称命名为{dataset_name}_{llm}_{triple_trans}

    ###############################################################################################################

    # 建立三元组翻译的prompt
    mode = "triples_trans"
    triples_trans_prompt = {}
    with tqdm(dataset['test'], desc="Building triple transaction prompts") as pbar:
        for each_sample in pbar:
            # 筛选掉已经完成的sample
            if each_sample["id"] in finished_id:
                continue

            pbar.set_postfix(current_question=each_sample["question"])
            sub_graph = each_sample["pruned_graph"]
            
            triple_text_list = []
            for triple in sub_graph:
                if isinstance(triple, list) and len(triple) == 3:  # 确保每个三元组是长度为3的列表
                    triple_text_list.append(f"({triple[0]}, {triple[1]}, {triple[2]})")
            
            # 使用列表推导式给每个元素添加序号
            triple_text_list = [f"{i+1}.{text}" for i, text in enumerate(triple_text_list)]

            # 用换行符连接每个三元组文本
            sub_graph_text = "\n".join(triple_text_list)
            # 加上input前缀
            pre_text = "Input:\n" \
            "Triple(s):\n"

            sub_graph_text = pre_text + sub_graph_text
            # 调用 PromptBuilder，传入 question
            prompt_builder = PromptBuilder(sub_graph_text,mode)
            triples_trans_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # 调用llm翻译三元组
    llm_chat = llm_client()
    triple_queries = {}

    # await asyncio.gather(*(llm_chat.response(each_triples_trans_prompt,mode) for question_id,each_triples_trans_prompt in triples_trans_prompt.items()))
    
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
        process_single_query(llm_chat, question_id, each_triples_trans_prompt, mode)
        for question_id, each_triples_trans_prompt in triples_trans_prompt.items()
    ]
    
    # 使用 tqdm_asyncio 显示进度条
    with tqdm_asyncio(desc=f"Call {llm} for triple transaction", total=len(triples_trans_prompt)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # 等待某个任务完成
            question_id, response = result  # 从返回值解构
            pbar.update(1)  # 更新进度条
            # print(f"Processed {question_id}: {response}")  # 处理每个任务的结果

    # with tqdm(triples_trans_prompt.items(),desc=f"Call {llm} for triple transaction") as pbar:
    #     for question_id,each_triples_trans_prompt in pbar:
    #         pbar.set_postfix(current_question=question_id)
    #         response = await llm_chat.response(each_triples_trans_prompt,mode)

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
                start_index = response.find("Natural Language Question:")
                if start_index != -1:
                    # 如果存在，提取从"Natural Language Question: "开始的内容
                    response = response[start_index + len("Natural Language Question:"):]
                
                # 按行分割内容
                lines = response.splitlines()
                
                for line in lines:
                    # 跳过空行
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 找到三元组部分和对应问题的分隔符 ":"
                    part = line.split("<SEP>", 2)
                    if len(part) == 3:
                        left,right1,right2 = part
                        triple_queries.append([right1,right2])
                    elif len(part) == 2:
                        left,right = part
                        triple_queries.append([right])
                    else: 
                        #异常,在检查子图长度的时候可以检查出来
                        continue
                        
                return triple_queries
            
            processed_answer = extract_triple_queries(response)
            triple_queries[question_id] = processed_answer
            
            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            pruned_graph_len = len(example["pruned_graph"])
            print("graph长度",len(example["pruned_graph"]))
            print("生成的list的长度",len(processed_answer))
            if len(processed_answer) != pruned_graph_len:
                # 丢弃此条结果
                continue
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "triple_unit_queries": processed_answer}
                batch_results.append(example_with_prediction)

            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(triple_trans_dataset["test"]) == 0:  # 如果 triple_trans_dataset["test"] 是空的
                    triple_trans_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    triple_trans_dataset["test"] = Dataset.from_dict({
                        key: triple_trans_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in triple_trans_dataset["test"].column_names
                    })

                # 写入到文件
                triple_trans_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储

    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(triple_trans_dataset["test"]) == 0:  # 如果 triple_trans_dataset["test"] 是空的
            triple_trans_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            triple_trans_dataset["test"] = Dataset.from_dict({
                key: triple_trans_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in triple_trans_dataset["test"].column_names
            })
        triple_trans_dataset["test"].to_parquet(write_data_dir)
    
    print(f"完成{dataset_name}数据集的三元组翻译任务!")
    print("三元组翻译后的样例:",triple_trans_dataset["test"])

    # 匹配到原数据集并新增字段triple_unit_queries
    # 确保每条记录都初始化了 'triple_unit_queries' 列
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