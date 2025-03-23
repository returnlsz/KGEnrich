import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from get_answer_entity_coverage import check_answer_in_graph_main

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

def merge_two_datasets(base_dataset, addition_dataset, field_list, key_column="id"):
    """
    合并两个数据集，在指定 key_column 的基础上更新/新增字段。

    参数：
    - base_dataset: Hugging Face Dataset（基准数据集）
    - addition_dataset: Hugging Face Dataset（用于更新或新增字段的数据集）
    - field_list: list，需要更新或新增的字段
    - key_column: str，主键字段的名称，默认是 "id"

    返回：
    - merged_dataset: 合并完成的 Hugging Face Dataset
    """

    # 将两个数据集转换为 Pandas DataFrame
    base_df = base_dataset.to_pandas()
    addition_df = addition_dataset.to_pandas()

    # 检查行数是否一致
    if len(base_df) != len(addition_df):
        raise ValueError("Base dataset and addition dataset must have the same number of rows to perform the merge.")

    # 检查主键字段是否存在
    if key_column not in base_df.columns:
        raise KeyError(f"The key column '{key_column}' is not found in the base dataset.")
    if key_column not in addition_df.columns:
        raise KeyError(f"The key column '{key_column}' is not found in the addition dataset.")

    # 按照主键 id 合并数据
    base_df.set_index(key_column, inplace=True)
    addition_df.set_index(key_column, inplace=True)

    for field in field_list:
        if field in addition_df.columns:
            if field in base_df.columns:
                # 覆盖 base 中的字段
                base_df[field].update(addition_df[field])
            else:
                # 新增字段
                base_df[field] = addition_df[field]
        else:
            print(f"Field '{field}' not found in the addition dataset. Skipping...")

    # 重置索引并转换为 Hugging Face Dataset 格式
    base_df.reset_index(inplace=True)
    merged_dataset = Dataset.from_pandas(base_df)

    return merged_dataset

if __name__ == "__main__":
    # 示例数据集加载
    # Base 数据集
    # base_data = {
    #     "id": [1, 2, 3],  # 主键
    #     "name": ["Alice", "Bob", "Charlie"],
    #     "age": [25, 30, 35]
    # }

    # # Addition 数据集
    # addition_data = {
    #     "id": [1, 2, 4],  # 主键，其中 ID=4 在 base 数据集中不存在
    #     "name": ["Anna", "Ben", "Daisy"],   # 更新 name 字段
    #     "height": [165, 175, 180]           # 新增 height 字段
    # }

    # # 转换为 Hugging Face 的 Dataset 格式
    # base_dataset = Dataset.from_pandas(pd.DataFrame(base_data))
    # addition_dataset = Dataset.from_pandas(pd.DataFrame(addition_data))

    # # 需要合并的字段列表
    # field_list = ["name", "height"]

    # cwq数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'

    base_dataset = load_dataset(
        "parquet", 
        data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/question_decompose_datasets/cwq_gpt4o-mini_question_decompose.parquet'}
    )

    # addition_dataset = load_dataset(
    #     "parquet", 
    #     data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_filter_gt_triples_datasets/webqsp_gpt4o-mini_sentence-transformers_750_llm_filter_gt_triples.parquet'}
    # )

    addition_dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    field_list = ["graph"]

    print("base_dataset:",base_dataset)
    print("addition_dataset:",addition_dataset)
    # 调用 merge 函数合并数据集
    merged_dataset = merge_two_datasets(base_dataset["test"], addition_dataset["test"], field_list)
    dataset_dict = DatasetDict({"test": merged_dataset})

    # 打印合并后的数据集
    print(dataset_dict)

    # 写入到文件
    dataset_dict["test"].to_parquet("/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/temp_datasets/cwq_gpt4o-mini_question_decompose.parquet")
    print("文件已保存!")

    # 验证下是不是真的转换成功
    # check_answer_in_graph_main(dataset_dict,task="initial_pruning")