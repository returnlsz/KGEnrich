
from datasets import load_dataset
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
# cwq数据集
# data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
# webqsp数据集
data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'
# data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

# 使用通配符匹配所有以 "test" 开头的 parquet 文件
dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

# 打印数据集的信息
print(dataset)

###############################################################################################################

###############################################################################################################

# step 4
# 剪枝，根据每个user queries找到与其相似的top k个triple unit queries，需要融合子图（去重复），并将对应的三元组加入到pruned_graph中，字段如下:
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

# 最终输出文件的名称命名为{dataset_name}_{llm}_{embedding_model_name}_{faiss}_{topk}

###############################################################################################################
# TODO:是否还需要根据user unit query进行剪枝,因为之前已经使用了origin query进行剪枝了
 
# pruning_llm = "sentence-transformers"

# with tqdm(triple_trans_dataset["test"], desc="Using user queries search topk triple unit queries and corresponding triple") as pbar:
#     for sample in pbar:
#         user_queries = sample["user_queries"]
#         triple_unit_queries = sample["triple_unit_queries"]
#         # 需要拆解triple_unit_queries构建一个index的索引,key-value:triple index-question



###############################################################################################################