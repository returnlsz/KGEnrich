# 主函数
# pipeline:
# 预剪枝->llm call:剪枝->llm call:分解query->llm call:llm prune->llm call:triple trans->llm call:筛选query相关的三元组(filter triples)
# ->llm call:structral enrich
# ->llm call:feature enrich
## ->llm call:qa->eval评价指标
## ->子图质量评估
import sys
import os
from src.utils.initial_prune import initial_prune
from src.llm.llm_decompose_query import llm_decompose_query
from src.llm.llm_trans_triple import llm_trans_triple_main
from src.llm.llm_qa import llm_qa
from src.utils.evaluate_result import eval_result_main
from src.llm.llm_prune import llm_prune

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import asyncio

def run(args):
    task = args.task
    if task == "initial_pruning":
        initial_prune(args.d,args.embedding_model,args.pruning_top_k)
    elif task == "llm_pruning":
        llm_prune(args.d,args.llm,args.embedding_model,args.pruning_top_k,args.llm_pruning_top_k)
    elif task == "query_decompose":
        llm_decompose_query(args.d,args.llm,args.embedding_model,args.pruning_top_k)
    elif task == "triple_trans":
        llm_trans_triple_main(args.d,args.llm,args.embedding_model,args.pruning_top_k,args.llm_pruning_top_k)
    elif task == "qa" or task == "direct_qa":
        llm_qa(args.d,args.llm,args.embedding_model,args.pruning_top_k,task,args.llm_pruning_top_k)
    elif task == "qa_evaluate":
        eval_result_main(args.d,args.cal_f1,args.qa_eval_top_k,args.llm)
    else:
        print("功能未实现!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str,help="测试集名称", choices=["cwq","webqsp"],default="cwq")
    argparser.add_argument("--embedding_model",type=str,help="测试集路径",choices=["sentence-transformers"])
    argparser.add_argument("--llm",type=str,help="大模型",choices=["gpt3.5","gpt4o-mini"])
    argparser.add_argument("--task",type=str,help="执行任务",choices=["initial_pruning","llm_pruning","query_decompose","triple_trans","filter_triples","structral_enrich","feature_enrich","graph_eval","qa","direct_qa","qa_evaluate"])
    argparser.add_argument('--eval_top_k',help="取topk个结果进行qa eval",type=int, default=-1)
    argparser.add_argument('--pruning_top_k',help="在剪枝的时候取相似度最高的top k个结果",type=int, default=750)
    argparser.add_argument('--llm_pruning_top_k',help="在llm剪枝的时候取相似度最高的top k个结果",type=int, default=300)
    argparser.add_argument('--qa_eval_top_k',help="在llm剪枝的时候取相似度最高的top k个结果",type=int, default=-1)
    argparser.add_argument('--cal_f1', action='store_true', help="在qa eval的时候使用f1分数",default=True)

    args = argparser.parse_args()
    run(args=args)
