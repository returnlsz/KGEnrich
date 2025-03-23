from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datasets import load_dataset

# 初始化 Sentence-BERT 模型
model = SentenceTransformer("/Users/jiangtong/KnowledgeEnrich/project/sentence-transformers")

# 构建待检索查询
dataset = load_dataset("parquet", data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/test_process.parquet'})
# 访问测试集
test_dataset = dataset['test']

print("数据集的字段:",test_dataset)
# 访问样本
index = 1
sample = test_dataset[index]

print("该样本的三元组数量为:",len(sample["graph"]))
corpus = []
# 将 first_sample 中每个元素转换为字符串后存放到 corpus
for sublist in sample["graph"]:
    corpus.append(" ".join(map(str, sublist)))

# # 假设有一批待检索的查询
# corpus = [
#     "['Mascot', 'type.type.expected_by', 'sports_team_mascot']",
#     "['San Francisco Giants', 'baseball.baseball_team.team_stats', 'm.05n69q3']",
#     "['San Francisco Giants', 'sports.sports_team.arena_stadium', 'Seals Stadium']",
#     "['Lou-Seal.jpg', 'common.image.size', 'm.0kksz7']",
#     "['Team', 'type.property.schema', 'Mascot']",
#     "['St. George Cricket Grounds', 'sports.sports_facility.teams', 'San Francisco Giants']",
#     "['San Francisco Giants', 'sports.sports_team.team_mascot', 'Crazy Crab']"
# ]

# 对候选集进行嵌入
corpus_embeddings = model.encode(corpus)

# 对嵌入向量进行归一化（余弦相似度需要将向量归一化到单位范数）
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# 构建 FAISS 索引（使用内积）
embedding_dim = corpus_embeddings.shape[1]  # 嵌入向量的维度
index = faiss.IndexFlatIP(embedding_dim)    # 使用内积
index.add(corpus_embeddings)               # 添加嵌入向量到索引中

# 输入一个查询
query = sample["question"]
query_embedding = model.encode([query])    # 对查询嵌入

# 对查询向量进行归一化
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

# 检索 top k
k = 100
distances, indices = index.search(query_embedding, k)

# 输出最相似的 k 个候选查询及相似度
print("Input Query:", query)
print("Answer:",sample["answer"])
print("\nTop-k Results:")
for i, idx in enumerate(indices[0]):
    print(f"{i + 1}. {corpus[idx]} (Similarity: {distances[0][i]:.4f})")

