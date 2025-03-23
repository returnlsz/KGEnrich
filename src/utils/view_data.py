from datasets import load_dataset

def write_to_file(top_k_samples, file_name="view_data_result.txt"):
    """
    将前 k 个样本写入文件。
    :param top_k_samples: list,包含前 k 个样本的字典
    :param file_name: str,写入的文件路径
    """
    # 将字典内容转换为字符串形式（比如 JSON 格式）
    import json
    top_k_samples_str = json.dumps(top_k_samples, indent=4, ensure_ascii=False)
    
    # 打开文件并覆盖写入
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(top_k_samples_str)

# 加载数据集
dataset = load_dataset(
    "parquet", 
    data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/temp_datasets/webqsp_gpt4o-mini_question_decompose.parquet'}
)

# 访问测试集
test_dataset = dataset['test']

print("该数据集信息:", test_dataset)

# 访问前 top_k 个样本
top_k = 100
top_k_samples = [test_dataset[i] for i in range(top_k)]

write_path = "/Users/jiangtong/KnowledgeEnrich/project/view_data_result.txt"

# 打印前 k 个样本的 key 名称（只打印第一个样本的 key 名称为参考）
for key in top_k_samples[0].keys():
    print(key)

# 调用函数将 top_k_samples 写入到文件
write_to_file(top_k_samples, write_path)