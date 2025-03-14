# 该脚本用于训练特定数据集上的KGE模型
# dataset_name,KGE_name

from pykeen.pipeline import pipeline
from pykeen.models import Model  # 用于加载模型
from pykeen.triples import TriplesFactory
import torch
from pykeen.constants import PYKEEN_CHECKPOINTS
from pykeen.datasets import get_dataset
from pykeen.predict import predict_triples
import numpy as np
from pykeen.datasets import WN18
from datasets import load_dataset
from tqdm import tqdm
import csv
import pandas as pd

dataset_name = "cwq"
KGE_name = "RotatE"
training_samples_base_url = "/Users/jiangtong/KnowledgeEnrich/project/weights/training_samples/"
# 加载路径
# cwq数据集
data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
# webqsp数据集
# data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'

# # 使用通配符匹配所有以 "test" 开头的 parquet 文件
# dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

# # 打印数据集的信息
# print(dataset)

# subgraph = []
# # 将 subgraph 转换为符合要求的格式：Sequence[tuple[str, str, str]]
# for sample in tqdm(dataset["test"],desc="转换成Pykeen要求的格式"):
#     sample_graph = sample["graph"]
#     for each_sample_graph in sample_graph:
#         subgraph.append(each_sample_graph)
# converted_subgraph = [tuple(triple) for triple in subgraph]
# # 转换为 np.ndarray，形状为 (n, 3)
# # print("converted_subgraph的长度:",len(converted_subgraph))

# 指定输出的 CSV 文件名
csv_file_name = training_samples_base_url + f"{dataset_name}_{KGE_name}_train_samples.csv"

# # 将数据写入 CSV 文件
# with open(csv_file_name, mode="w", newline="", encoding="utf-8") as csvfile:
#     csv_writer = csv.writer(csvfile)
    
#     # 如果需要添加表头，可以取消注释以下行
#     # csv_writer.writerow(["Subject", "Predicate", "Object"])
    
#     # 写入数据行
#     csv_writer.writerows(converted_subgraph)

# print(f"Data successfully written to {csv_file_name}")

# '''将以逗号分隔的csv文件转换为\t分隔'''
# data = pd.read_csv(csv_file_name)
# data.to_csv(csv_file_name,sep = '\t', index=False)

my_subgraph = TriplesFactory.from_path(csv_file_name)

# my_subgraph = TriplesFactory.from_labeled_triples(subgraph_ndarray)

training, testing = my_subgraph.split([.8, .2])

# 使用 PyKEEN 的 pipeline 函数快速训练 RotatE 模型
result = pipeline(
    model=KGE_name,
    training = training, testing = testing,  # 替换为你自己的数据集，例如 WN18 或 FB15k
    result_tracker="tensorboard",
    optimizer='Adam',
    stopper="early",
    training_kwargs=dict(
        num_epochs=2000,
        batch_size=256,
        checkpoint_frequency=30,
        checkpoint_name=f'{dataset_name}_{KGE_name}_checkpoint.pt',
        checkpoint_on_failure=True,
    ),
    random_seed=100
)

# 训练后的模型
model = result.model

model_save_path = f'/Users/jiangtong/KnowledgeEnrich/project/weights/{dataset_name}_{KGE_name}'
# 保存训练好的模型
result.save_to_directory(model_save_path)

model_path = f"/Users/jiangtong/KnowledgeEnrich/project/weights/{dataset_name}_{KGE_name}/trained_model.pkl"
# # 从保存的文件加载模型
loaded_model = torch.load(PYKEEN_CHECKPOINTS.joinpath(model_path),weights_only=False)
# loaded_model = Model.load(path=model_save_path,torch_load_kwargs={"weights_only": False})
print("模型已从文件加载")

# 定义要打分的三元组
# dataset = get_dataset(dataset=WN18)

# # 定义自己的数据集
# # 提供triple，给出预测分数
# triple = TriplesFactory.from_labeled_triples(np.array([('a',3,'b')]))
# score = predict_triples(model=loaded_model,triples = triple)
# print(score.scores)

# 使用加载后的模型打分
pack = predict_triples(model=loaded_model, triples=testing)
df = pack.process(factory=training).df

# print(df)

# 输出结果
print(df.nlargest(n=5, columns="score"))