# 该脚本用于计算给定子图的语义丰富度,计算方法:score2=sum{Embedding score(h,r,t) in eG}
# input:
# subgraph
# embedding_model:这里使用在WN18上预训练的RotatE
# output:
# semantic socre

from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm
from pykeen.constants import PYKEEN_CHECKPOINTS
from pykeen.datasets import get_dataset
from pykeen.predict import predict_triples
from pykeen.triples import TriplesFactory

def caculate_semantic_score(subgraph=[],embedding_model=None):
    semantic_score = 0
    # 将 subgraph 转换为符合要求的格式：Sequence[tuple[str, str, str]]
    converted_subgraph = [tuple(triple) for triple in subgraph]
    # 转换为 np.ndarray，形状为 (n, 3)
    subgraph_ndarray = np.array(converted_subgraph)

    my_subgraph = TriplesFactory.from_labeled_triples(subgraph_ndarray)
    pack = predict_triples(model=embedding_model, triples=my_subgraph)
    # 映射id
    df = pack.process(factory=my_subgraph).df
    print(type(df))
    semantic_score = semantic_score + df.scores

    # # 使用加载后的模型打分
    # for triple_tuple in my_subgraph:

    #     pack = predict_triples(model=embedding_model, triples=triple_tuple)
    #     # 映射id
    #     df = pack.process(factory=dataset.training).df
    #     semantic_score = semantic_score + df.scores
    return semantic_score

if __name__ == "__main__":
    # 加载路径
    # cwq数据集
    data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    # 打印数据集的信息
    print(dataset)

    # 加载KGE模型
    model_path = "/Users/jiangtong/KnowledgeEnrich/project/weights/trained_model.pkl"
    embedding_model = torch.load(PYKEEN_CHECKPOINTS.joinpath(model_path),weights_only=False)

    semantic_score = []
    # 计算得分
    for sample in tqdm(dataset["test"],desc="计算语义丰富性中..."):
        subgraph = sample["graph"]
        semantic_score.append(caculate_semantic_score(subgraph=subgraph,embedding_model=embedding_model))
    print(semantic_score)
        
    
    

    