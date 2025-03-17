import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from datasets import Dataset, DatasetDict

def llm_qa(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,task="qa",llm_pruning_top_k=-1):
    # step 0
    # 从parquet文件中加载数据集,并把数据集组织成一个list-dict,dict的字段如下:
    # id
    # question
    # answer
    # q_entity
    # a_entity
    # graph
    # choices

    ###############################################################################################################
    # 加载路径
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_dataset'
    
    if llm_pruning_top_k != -1:
        data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_dataset'
        dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_{llm_pruning_top_k}_llm_pruning.parquet'})
    else:
        data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/initial_pruning_datasets'
        dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{initial_pruning_llm}_{initial_pruning_topk}_initial_pruning.parquet'})
    # dataset = load_dataset("parquet", data_files={'test': f'/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_dataset/webqsp_gpt4o-mini_sentence-transformers_750_100_llm_pruning.parquet'})
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'
    mode = task

    # 打印数据集的信息
    print(dataset)

    # 要写入的文件的路径
    # 获取当前时间
    current_time = datetime.now()

    # 将时间格式化为字符串（如 "2023-10-10_14-30-00"）,将这里替换成上次未完成的parquet名称可以继续
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    write_data_dir = f"preprocess_datasets/qa_datasets/{dataset_name}_{llm}_{task}_{time_str}.parquet"
    # write_data_dir = "preprocess_datasets/qa_datasets/webqsp_gpt4o-mini_direct_qa_2025-03-15_21-08-12.parquet"
    # write_data_dir = "preprocess_datasets/qa_datasets/cwq_gpt4o-mini_direct_qa_2025-03-15_20-54-51.parquet"
    # 打开该文件,若不存在,则创建
    # 确保目录存在
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    qa_dataset = None
    finished_id = []

    # 检查文件是否存在，如果不存在，则创建文件
    if not os.path.exists(write_data_dir):
        # 如果文件不存在，创建一个空的 DataFrame 并保存为 parquet 文件
        df = pd.DataFrame()  # 创建空 DataFrame
        df.to_parquet(write_data_dir)
        print(f"文件不存在，已创建新的空文件：{write_data_dir}")
        # 初始化qa_dataset
        qa_dataset = DatasetDict({
            "test": Dataset.from_dict({
                "id": "",
                "question": "",
                "answer": [],
                "q_entity": [],
                "a_entity": [],
                "graph": [],
                "pruned_graph": [],
                "choices": [],
                "predictions":[]
            })
        })
    else:
        print(f"文件已存在：{write_data_dir},将从该文件继续完成{task}任务")
        qa_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # 检查已经存在的question_id
        for sample in qa_dataset["test"]:
            finished_id.append(sample["id"])



    ###############################################################################################################

    ###############################################################################################################
    # qa task
    ###############################################################################################################
    def assemble_prompt(question, subgraph):
        """
        根据给定的模板和输入的question以及subgraph,组装并返回一个prompt字符串。
        
        :param question: str,输入的问题
        :param subgraph: list,包含三元组的列表,每个三元组是一个长度为3的list
        :return: str,组装后的prompt
        """
        # 初始化模板
        template = "Input:\nquestion:\n{input_question}\ninformation:\n{triples}"
        
        # 将subgraph中的每个三元组转换为字符串
        triple_strings = ["{} {} {}".format(triple[0], triple[1], triple[2]) for triple in subgraph]
        
        # 将三元组字符串用换行符连接
        triples_section = "\n".join(triple_strings)
        
        # 使用模板组装结果
        prompt = template.format(input_question=question, triples=triples_section)
        
        return prompt
    
    # 用于提取 question 并生成 result_prompt
    def process_dataset(dataset):
        # 存储结果
        result_prompts = {}
        
        # 遍历数据集中的每一行
        for example in tqdm(dataset['test'],desc="Building question decompose prompts"):
            # 提取 question 字段
            question_id = example['id']
            # 筛选掉已经完成的sample
            if question_id in finished_id:
                continue
            question = example['question']
            if llm_pruning_top_k != -1:
                subgraph = example["pruned_graph"]
            else:
                subgraph = example["graph"]
            # 对question和information进行组装
            # 测试数据
            # # 输入的question
            # question = "谁是爱因斯坦的妻子？"

            # # 输入的subgraph
            # subgraph = [
            #     ["爱因斯坦", "妻子", "米列娃·马里克"],
            #     ["爱因斯坦", "出生地", "德国乌尔姆"],
            #     ["米列娃·马里克", "职业", "物理学家"]
            # ]
            each_prompt = assemble_prompt(question,subgraph)
            # 调用 PromptBuilder,传入 question
            prompt_builder = PromptBuilder(each_prompt,mode)
            
            # 获取生成的 result_prompt
            result_prompts[question_id] = prompt_builder.build_prompt()

            # print(result_prompts)
            # return result_prompts
        
        return result_prompts

    # 调用函数处理数据集
    qa_result_prompts = process_dataset(dataset)
    llm_chat = llm_client() 

    def process_llm_answer(llm_answer=""):
        """
        提取 llm_answer 中 "Final answer:" 后面的内容,并进行分割处理,返回一个答案列表。
        如果 llm_answer 为空或不包含 "Final answer:",则返回空列表。
        
        :param llm_answer: str,包含答案的字符串
        :return: list,处理后的答案列表
        """
        # 判断 llm_answer 是否为空字符串
        if not llm_answer:
            return []

        # 查找 "Final answer:" 的位置
        final_answer_keyword = "Final answer:"
        if final_answer_keyword in llm_answer:
            # 提取 "Final answer:" 后的内容
            content = llm_answer.split(final_answer_keyword, 1)[1].strip()
        else:
            # 如果字符串中没有 "Final answer:",返回空列表
            return []

        # 对提取的内容按逗号分隔并去除两边可能的空格
        answers = [answer.strip() for answer in content.split(",")]
        
        # 返回答案列表
        return answers

    qa_result = {}

    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 10  # 设置批次大小

    # 使用 tqdm 遍历处理
    with tqdm(qa_result_prompts.items(), desc=f"{dataset_name} using {llm} QA") as pbar:
        for question_id, qa_prompt in pbar:
            # 调用 LLM 获取回答
            llm_answer = llm_chat.response(qa_prompt, mode)

            # 对 LLM 的回答进行处理
            processed_answer = process_llm_answer(llm_answer=llm_answer)
            qa_result[question_id] = processed_answer

            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)
            if example:
                # 为 example 添加预测结果
                example_with_prediction = {**example, "predictions": processed_answer}
                batch_results.append(example_with_prediction)

            # 当批量结果达到 filter_batch_size 时，进行一次写入
            if len(batch_results) >= filter_batch_size:
                if len(qa_dataset["test"]) == 0:  # 如果 qa_dataset["test"] 是空的
                    qa_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # 批量合并到现有的 Dataset
                    qa_dataset["test"] = Dataset.from_dict({
                        key: qa_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in qa_dataset["test"].column_names
                    })

                # 写入到文件
                qa_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # 清空临时存储

    # 如果有剩余的结果，写入到文件
    if batch_results:
        if len(qa_dataset["test"]) == 0:  # 如果 qa_dataset["test"] 是空的
            qa_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            qa_dataset["test"] = Dataset.from_dict({
                key: qa_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in qa_dataset["test"].column_names
            })
        qa_dataset["test"].to_parquet(write_data_dir)
            


    # # 确保每条记录都初始化了 "predictions" 列
    # if "predictions" not in qa_dataset["test"].column_names:
    #     qa_dataset["test"] = qa_dataset["test"].add_column("predictions", [None] * len(qa_dataset["test"]))

    # # 将 filtered_graph 转化为以 question_id 为键的字典
    # qa_result_dict = {question_id: predictions for question_id, predictions in qa_result.items()}

    # def add_prediction_to_dataset(batch):
    #     # 遍历批量数据,将匹配到的 `graph` 字段更新为 `filtered_graph_dict` 中的值
    #     for i, sample_id in enumerate(batch["id"]):
    #         if sample_id in qa_result_dict:
    #             batch["predictions"][i] = qa_result_dict[sample_id]
    #     return batch
    

    # # 使用 Dataset.map 函数进行批量更新,并避免重复遍历
    # qa_dataset["test"] = qa_dataset["test"].map(
    #     add_prediction_to_dataset, 
    #     batched=True,  # 启用批量操作以加速
    #     batch_size=64,  # 设置合适的批量大小
    #     desc="更新数据集的predictions字段"
    # )
    # # 获取当前时间
    # current_time = datetime.now()

    # # 将时间格式化为字符串（如 "2023-10-10_14-30-00"）
    # time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    # qa_dataset["test"].to_parquet(f"preprocess_datasets/qa_datasets/{dataset_name}_{llm}_{task}_{time_str}.parquet")

