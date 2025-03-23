# 该脚本用于获取当前数据集中,graph的答案覆盖率，也就是说,answer entity是否在graph的某个三元组中
# 输入:dataset
from datasets import load_dataset

def check_answer_in_graph(sample,graph_name="graph"):
    # 获取 answer 列表和 graph
    answers = sample["answer"]  # 假设 sample["answer"] 是一个列表
    graph = sample[graph_name]

    # 计数器，用于统计存在的答案个数
    match_count = 0

    # 遍历 answers 中的每个答案
    for answer in answers:
        # 遍历 graph 中的每个三元组
        for triplet in graph:
            # 检查当前答案是否存在于当前三元组中
            if answer in triplet:
                match_count += 1
                break  # 答案已匹配，不需要继续检查其他三元组
    if match_count == 0:
        # print(sample["id"])
        pass
    return match_count

def check_answer_in_graph_main(dataset,task="llm_pruning"):
    exist_number = 0
    total_number = 0
    if task == "llm_pruning":
        graph_name = "pruned_graph"
    else:
        graph_name = "graph"
    for sample in dataset["test"]:
        is_exist = check_answer_in_graph(sample,graph_name=graph_name)
        exist_number = exist_number + is_exist
        total_number = total_number + len(sample["answer"])
    
    coverage = exist_number / total_number
    print("数据集样本的答案总数:",total_number)
    print("覆盖样本的答案数:",exist_number)
    print("覆盖率为:",coverage)


if __name__ == "__main__":
    # 构建待检索查询
    # 加载路径
    # cwq数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    # dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    dataset = load_dataset(
        "parquet", 
        data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/temp_datasets/cwq_gpt4o-mini_question_decompose.parquet'}
    )
    # 访问测试集
    test_dataset = dataset['test']

    print("数据集字段概览:",test_dataset)
    exist_number = 0
    total_number = 0

    for sample in test_dataset:
        is_exist = check_answer_in_graph(sample,graph_name="graph")
        exist_number = exist_number + is_exist
        total_number = total_number + len(sample["answer"])

    coverage = exist_number / total_number
    print("数据集样本的答案总数:",total_number)
    print("覆盖样本的答案数:",exist_number)
    print("覆盖率为:",coverage)
