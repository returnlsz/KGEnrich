# 主函数
# pipeline:
# 预剪枝->llm call:剪枝->llm call:分解query->llm call:triple trans->llm call:筛选query相关的三元组
# ->llm call:structral enrich->llm call:feature enrich->子图质量评估->llm call:qa->eval评价指标
import sys
import os
from src.utils.initial_prune import initial_prune
from src.llm.llm_decompose_query import llm_decompose_query
from src.llm.llm_trans_triple import llm_trans_triple
from src.llm.llm_qa import llm_qa

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse

def run(args):
    task = args.task
    if task == "initial_pruning":
        initial_prune(args.d,args.embedding_model,args.pruning_top_k)
    elif task == "llm_pruning":
        pass
    elif task == "query_decompose":
        llm_decompose_query(args.d,args.embedding_model,args.pruning_top_k)
    elif task == "triple_trans":
        llm_trans_triple(args.d,args.llm)
    elif task == "qa" or task == "direct_qa":
        llm_qa(args.d,args.llm,args.embedding_model,args.pruning_top_k,task)
    else:
        print("功能未实现!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str,help="测试集名称", choices=["cwq","webqsp"],default="cwq")
    argparser.add_argument("--embedding_model",type=str,help="测试集路径",choices=["sentence-transformers"])
    argparser.add_argument("--llm",type=str,help="大模型",choices=["gpt3.5","gpt4o-mini"])
    argparser.add_argument("--task",type=str,help="执行任务",choices=["initial_pruning","llm_pruning","query_decompose","triple_trans","structral_enrich","feature_enrich","graph_eval","qa","direct_qa","qa_evaluate"])
    argparser.add_argument('--eval_top_k',help="取topk个结果进行qa eval",type=int, default=-1)
    argparser.add_argument('--pruning_top_k',help="在剪枝的时候取相似度最高的top k个结果",type=int, default=100)

    args = argparser.parse_args()
    run(args=args)
