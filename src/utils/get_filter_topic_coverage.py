# 此脚本用于获取filter triples里面的topic entity覆盖率
from datasets import load_dataset

def get_filter_topic_coverage(filter_graph,topics):
    cover_num = 0
    for topic in topics:
        for filter_triple in filter_graph:
            if topic in filter_triple:
                cover_num += 1
                break

    return cover_num/len(topics)


def get_filter_topic_coverage_main(dataset=None):
    cover_ratio = 0
    total_length = 0
    for sample in dataset["test"]:
        topics = sample["q_entity"]
        pruned_graph = sample["pruned_graph"]
        filter_triples = sample["filter_triples"]
        total_length += 1

        filter_graph = []

        for index,question_list in enumerate(filter_triples):
            if question_list == None or len(question_list) == 0:
                pass
            else:
                filter_graph.append(pruned_graph[index])
        
        each_cover_ratio = get_filter_topic_coverage(filter_graph,topics)
        if each_cover_ratio == 0:
            print(sample["id"])
            pass
        cover_ratio += each_cover_ratio
    return cover_ratio/total_length

if __name__ == "__main__":
    dataset = load_dataset(
        "parquet", 
        data_files={'test': '/Users/jiangtong/KnowledgeEnrich/project/preprocess_datasets/filter_triple_datasets/webqsp_gpt4o-mini_filter_triple.parquet'}
    )
    print("当前数据集在filter_graph里面的topic entity覆盖率为:",get_filter_topic_coverage_main(dataset))