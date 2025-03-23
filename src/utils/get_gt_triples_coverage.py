# 此脚本用于检验给定triples在subgraph中的覆盖率
# 输入的dataset需要有gt_triples和graph字段

from datasets import load_dataset

def get_gt_triples_coverage(gt_triples,subgraph):
    match_num = 0
    over_3_gt_triples_num = 0
    if len(gt_triples) == 0:
        return 0,0
    for gt_triple in gt_triples:
        if len(gt_triple) != 3:
            over_3_gt_triples_num += 1
            continue
        else:
            for triple in subgraph:
                if gt_triple[0] == triple[0] and gt_triple[1] == triple[1] and gt_triple[2] == triple[2]:
                    match_num += 1
                    break
                # print("存在长度不为3的gt_triple,跳过!")
    return (match_num/len(gt_triples),over_3_gt_triples_num/len(gt_triples))

def get_gt_triples_coverage_main(dataset):
    coverage_list = []
    over_3_gt_triples_list = []
    gt_zero_num = 0

    for sample in dataset["test"]:
        gt_triples = sample["gt_triples"]
        subgraph = sample["graph"]
        cover_ratio,over_3_gt_triples_ratio = get_gt_triples_coverage(gt_triples,subgraph)
        if cover_ratio == 0 and over_3_gt_triples_ratio == 0:
            gt_zero_num += 1
        over_3_gt_triples_list.append(over_3_gt_triples_ratio)
        coverage_list.append(cover_ratio)
    print("gt_triples在graph中的覆盖率为:",sum(coverage_list)/len(coverage_list))
    print("长度不为3的gt_triple的比例为:",sum(over_3_gt_triples_list) / len(over_3_gt_triples_list))
    print("gt_triples字段长度为0的比例:",gt_zero_num/len(coverage_list))

if __name__ == "__main__":
    # 构建待检索查询
    # 加载路径
    # cwq数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-cwq/RoG-cwq/data/'
    # webqsp数据集
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/datasets/RoG-webqsp/data/'
    # data_dir = '/Users/jiangtong/KnowledgeEnrich/project/'

    # 使用通配符匹配所有以 "test" 开头的 parquet 文件
    # dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    dataset = load_dataset(
        "parquet", 
        data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/llm_filter_gt_triples_datasets/cwq_gpt4o-mini_sentence-transformers_750_llm_filter_gt_triples.parquet'}
    )
    # 访问测试集
    # test_dataset = dataset['test']

    get_gt_triples_coverage_main(dataset)
