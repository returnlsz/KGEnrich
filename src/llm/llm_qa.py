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
import asyncio
from tqdm.asyncio import tqdm_asyncio

def llm_qa_main(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,task="qa",llm_pruning_top_k=-1,resume_path=None):
    return asyncio.run(llm_qa(dataset_name,llm,initial_pruning_llm,initial_pruning_topk,task,llm_pruning_top_k,resume_path))

async def llm_qa(dataset_name=None,llm=None,initial_pruning_llm="sentence-transformers",initial_pruning_topk=750,task="qa",llm_pruning_top_k=-1,resume_path=None):
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
    ###############################################################################################################
    # 使用LLM剪枝后的pruned_graph进行direct_qa
    if task == "direct_qa":
        if llm_pruning_top_k != -1:
            data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_dataset'
            dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_{llm_pruning_top_k}_llm_pruning.parquet'})
        # 直接使用initial prune的graph进行direct_qa
        else:
            data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/initial_pruning_datasets'
            dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{initial_pruning_llm}_{initial_pruning_topk}_initial_pruning.parquet'})
    else:
            # 使用feature enrich后的pruned_graph以及feature enrich进行qa
        if task == "mix_qa" or task == "feature_enrich_qa":
            data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/feature_enrich_datasets'
            dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_feature_enrich.parquet'})
        elif task == "structural_enrich_qa":
            # 使用structural enrich后的pruned_graph以及structural enrich进行qa
            data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/structural_enrich_datasets'
            dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_structural_enrich.parquet'})
        else:
            # 使用gt triples进行qa
            data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_filter_gt_triples_datasets'
            data_dir = '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/temp_datasets'
            dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/{dataset_name}_{llm}_{initial_pruning_llm}_{initial_pruning_topk}_llm_filter_gt_triples.parquet'})
        
        

    ###############################################################################################################
    # dataset = load_dataset("parquet", data_files={'test': f'/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_pruning_dataset/webqsp_gpt4o-mini_sentence-transformers_750_100_llm_pruning.parquet'})
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'
    if task == "direct_qa" or task == "gt_qa":
        mode = "direct_qa"
    else:
        mode = "qa"

    # 打印数据集的信息
    print(dataset)

    # 要写入的文件的路径
    # 获取当前时间
    current_time = datetime.now()

    # 将时间格式化为字符串（如 "2023-10-10_14-30-00"）,将这里替换成上次未完成的parquet名称可以继续
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 保存文件,最终输出文件的名称命名为{dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/qa_datasets/{dataset_name}_{llm}_{task}_{time_str}.parquet"
    else:
        write_data_dir = resume_path
    # write_data_dir = "preprocess_datasets/qa_datasets/cwq_gpt4o-mini_gt_qa_2025-03-21_18-00-37.parquet"
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
        if task == "direct_qa":
            qa_dataset = DatasetDict({
                    "test": Dataset.from_dict({
                    "id": "",
                    "question": "",
                    "answer": [],
                    "q_entity": [],
                    "a_entity": [],
                    "graph": [],
                    "pruned_graph": [],
                    "choices": []
                })
            })
        elif task == "mix_qa":
            qa_dataset = DatasetDict({
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
                    "feature_enrich_triples":[],
                    "filter_triples":[]
                })
            })
        elif task == "feature_enrich_qa":
            qa_dataset = DatasetDict({
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
        elif task == "structural_enrich_qa":
            qa_dataset = DatasetDict({
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
            # gt triples qa
            qa_dataset = DatasetDict({
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
        triple_strings = ["({}, {}, {})".format(triple[0], triple[1], triple[2]) for triple in subgraph]
        
        # 将三元组字符串用换行符连接
        triples_section = "\n".join(triple_strings)
        
        # 使用模板组装结果
        prompt = template.format(input_question=question, triples=triples_section)
        
        return prompt
    
    def assemble_gt_prompt(question, subgraph,gt_triples):
        # 初始化模板
        template = "Input:\nquestion:\n{input_question}\ninformation:\n{triples}"
        
        # triple_strings1 = ["({}, {}, {})".format(triple[0], triple[1], triple[2]) for triple in gt_triples if len(triple) == 3]
        triple_strings1 = [
            "({})".format(", ".join(triple))  # 根据 triple 的长度动态为 "(h, r, t, ...)" 格式
            for triple in gt_triples
            if len(triple) >= 3  # 确保 triple 的长度至少为 3
        ]

        # 将subgraph中的每个三元组转换为字符串,这里只取前100个
        triple_strings2 = ["({}, {}, {})".format(triple[0], triple[1], triple[2]) for triple in subgraph[:100]]
        
        triple_strings = triple_strings1 + triple_strings2
        # 将三元组字符串用换行符连接
        triples_section = "\n".join(triple_strings)
        
        # 使用模板组装结果
        prompt = template.format(input_question=question, triples=triples_section)
        
        return prompt
    
    def assemble_filter_prompt(sample):
        """
        根据给定的模板和输入的question以及subgraph,组装并返回一个prompt字符串。
        
        :param question: str,输入的问题
        :param subgraph: list,包含三元组的列表,每个三元组是一个长度为3的list
        :return: str,组装后的prompt
        """

        input_question = sample["question"]
        pruned_graph =  sample["pruned_graph"]
        filter_triples = sample["filter_triples"]

        if task == "structural_enrich_qa" or task == "mix_qa":
            structural_enrich_triples = sample["structural_enrich_triples"]
        else:
            structural_enrich_triples = []
        if task == "feature_enrich_qa" or task == "mix_qa":
            feature_enrich_triples = sample["feature_enrich_triples"]
        else:
            feature_enrich_triples = []
        triple_unit_queries = sample["triple_unit_queries"]

        # 三元组的摆放顺序:structural_enrich_triples,feature_enrich_triples,filter_triples,no_filter_triples
        # 三元组以(s,p,o)的形式输入,而不是自然语言文本形式
        filter_graph = {}
        no_filter_graph = {}
        for index,question_list in enumerate(filter_triples):
            # 如果不是filter_triples,就加入user query
            if question_list != None and len(question_list) > 0:
                s,p,o = pruned_graph[index]
                if (s,p,o) not in filter_graph:
                    filter_graph[(s,p,o)] = []
                for question in question_list:
                    if question not in filter_graph[(s,p,o)]:
                        filter_graph[(s,p,o)].append(question)
            else:
                # 如果不是filter_triples,就直接加上triple user queries
                s,p,o = pruned_graph[index]
                no_filter_triple_question_list = triple_unit_queries[index]
                if (s,p,o) not in no_filter_graph:
                    no_filter_graph[(s,p,o)] = []
                for no_filter_triple_question in no_filter_triple_question_list:
                    no_filter_graph[(s,p,o)].append(no_filter_triple_question)

        # 创建存储所有内容的列表
        triples_questions = []

        # 处理 structural_enrich_triples
        for triple in structural_enrich_triples:
            if len(triple) != 3:
                print("structural enrich生成的三元组存在长度问题!")
            else:
                triples_questions.append(f"({triple[0]}, {triple[1]}, {triple[2]})")

        # 处理 feature_enrich_triples
        for triple in feature_enrich_triples:
            if len(triple) != 3:
                print("feature enrich生成的三元组存在长度问题!")
            else:
                triples_questions.append(f"({triple[0]}, {triple[1]}, {triple[2]})")

        # 处理 filter_graph
        for triple, question_list in filter_graph.items():
            triple_str = f"({triple[0]}, {triple[1]}, {triple[2]})"
            queries = "<SEP>".join(question_list)
            triples_questions.append(f"{triple_str}<SEP>{queries}")

        # 处理 no_filter_graph
        for triple, question_list in no_filter_graph.items():
            triple_str = f"({triple[0]}, {triple[1]}, {triple[2]})"
            queries = "<SEP>".join(question_list)
            triples_questions.append(f"{triple_str}<SEP>{queries}")

        # 拼接所有内容到模板
        triples_questions_str = "\n".join(triples_questions)
        prompt = f"Input:\nquestion:\n{input_question}\ninformation:\n{triples_questions_str}"
        
        return prompt
    
    # 用于提取 question 并生成 result_prompt
    def process_dataset(dataset):
        # 存储结果
        result_prompts = {}
        
        # 遍历数据集中的每一行
        for example in tqdm(dataset['test'],desc="Building question decompose prompts"):
            if mode == "qa":
                # 筛选掉已经完成的sample
                if example["id"] in finished_id:
                    continue
                each_prompt = assemble_filter_prompt(example)
                prompt_builder = PromptBuilder(each_prompt,mode)
                result_prompts[example["id"]] = prompt_builder.build_prompt()
            else:
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

                if task == "gt_qa":
                    each_prompt = assemble_gt_prompt(question,subgraph,example["gt_triples"])
                else:
                    each_prompt = assemble_prompt(question,subgraph)
                # 调用 PromptBuilder,传入 question
                prompt_builder = PromptBuilder(each_prompt,mode)
                # print(each_prompt)
                # 获取生成的 result_prompt
                result_prompts[question_id] = prompt_builder.build_prompt()
                # print("prompt内容:",result_prompts[question_id])

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
            content = llm_answer.strip()
        content = content.replace('```', '')
        content = content.replace('{thoughts & reason}', '').replace('{/thoughts & reason}', '')
        content = content.replace('{demonstrations}', '').replace('{/demonstrations}', '')
        content = content.replace('{/instruction}','').replace('{instruction}','')
        content = content.strip()
        # 对提取的内容按<SEP>分隔并去除两边可能的空格
        answers = [answer.strip() for answer in content.split("<SEP>")]
        
        # 返回答案列表
        return answers

    qa_result = {}

    # 将 dataset["test"] 转换为一个以 id 为键的快速检索字典
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # 用于临时存储结果的列表
    batch_results = []
    filter_batch_size = 10  # 设置批次大小

    # 定义异步函数，处理单个请求
    async def process_single_query(llm_chat, question_id, each_qa_prompt, mode):
        response = await llm_chat.response(each_qa_prompt, mode)
        return question_id, response

    # 定义异步函数，处理多个请求并按完成顺序处理结果
    tasks = [
        process_single_query(llm_chat, question_id, each_qa_prompt, mode)
        for question_id, each_qa_prompt in qa_result_prompts.items()
    ]
    
    # 使用 tqdm_asyncio 显示进度条
    with tqdm_asyncio(desc=f"Call {llm} for QA", total=len(qa_result_prompts)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # 等待某个任务完成
            question_id, llm_answer = result  # 从返回值解构
            pbar.update(1)  # 更新进度条

    # # 使用 tqdm 遍历处理
    # with tqdm(qa_result_prompts.items(), desc=f"{dataset_name} using {llm} QA") as pbar:
    #     for question_id, qa_prompt in pbar:
    #         # 调用 LLM 获取回答
    #         llm_answer = llm_chat.response(qa_prompt, mode)

            # 对 LLM 的回答进行处理
            processed_answer = process_llm_answer(llm_answer=llm_answer)
            qa_result[question_id] = processed_answer

            # 通过 question_id 快速检索对应的记录
            example = id_to_example_map.get(question_id)

            # print("该sample的prompt如下:",qa_result_prompts[example["id"]])
            # print("该sample的LLM Answer如下:",llm_answer)
            
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

