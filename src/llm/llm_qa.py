import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
from datetime import datetime

def llm_qa(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,task="qa"):
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
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/initial_pruning_datasets'

    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{initial_pruning_llm}_{initial_pruning_topk}_initial_pruning.parquet'})

    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'
    mode = task

    # 打印数据集的信息
    print(dataset)

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
            question = example['question']
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

    with tqdm(qa_result_prompts.items(),desc=f"{dataset_name} using {llm} QA") as pbar:
        for question_id,qa_prompt in pbar:
            llm_answer = llm_chat.response(qa_prompt, mode)
            # 对llm的回答进行处理
            processed_answer = process_llm_answer(llm_answer=llm_answer)
            qa_result[question_id] = processed_answer
    
    qa_dataset = dataset
    # 将预测结果加入到dataset中
    # 确保每条记录都初始化了 'user_queries' 列
    if "predictions" not in qa_dataset["test"].column_names:
        qa_dataset["test"] = qa_dataset["test"].add_column("predictions", [None] * len(qa_dataset["test"]))

    # 将 filtered_graph 转化为以 question_id 为键的字典
    qa_result_dict = {question_id: predictions for question_id, predictions in qa_result.items()}

    def add_prediction_to_dataset(batch):
        # 遍历批量数据,将匹配到的 `graph` 字段更新为 `filtered_graph_dict` 中的值
        for i, sample_id in enumerate(batch["id"]):
            if sample_id in qa_result_dict:
                batch["predictions"][i] = qa_result_dict[sample_id]
        return batch
    

    # 使用 Dataset.map 函数进行批量更新,并避免重复遍历
    qa_dataset["test"] = qa_dataset["test"].map(
        add_prediction_to_dataset, 
        batched=True,  # 启用批量操作以加速
        batch_size=64,  # 设置合适的批量大小
        desc="更新数据集的predictions字段"
    )
    # 获取当前时间
    current_time = datetime.now()

    # 将时间格式化为字符串（如 "2023-10-10_14-30-00"）
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    dataset["test"].to_parquet(f"preprocess_datasets/qa_datasets/{dataset_name}_{llm}_{task}_{time_str}.parquet")

