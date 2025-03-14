from datasets import load_dataset

# 加载数据集
dataset = load_dataset("parquet", data_files={
    'test': '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/test-00000-of-00003-e62a559c5d2b56ca.parquet'
})

# 访问测试集
test_dataset = dataset['test']

sample_num = 30

# 获取测试集的前 5 个数据
test_subset = test_dataset.select(range(sample_num))  # 选取前 5 个样本

# 保存为 Parquet 文件
test_subset.to_parquet("test_process.parquet")

print(f"前 {sample_num} 个测试样本已保存为 'test_process.parquet' 文件")