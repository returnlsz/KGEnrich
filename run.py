# 主函数
# pipeline:
# 预剪枝->llm call:分解query->llm query prune->llm call:triple trans->llm call:筛选query相关的三元组(filter triples)
# ->llm call:structral enrich
# ->llm call:feature enrich
## ->llm call:qa->eval评价指标
## ->子图质量评估
import sys
import os
from src.utils.initial_prune import initial_prune
from src.llm.llm_decompose_query import llm_decompose_query
from src.llm.llm_trans_triple import llm_trans_triple_main
from src.llm.llm_qa import llm_qa_main
from src.utils.evaluate_result import eval_result_main
from src.llm.llm_prune import llm_prune
from src.llm.llm_filter_triples import llm_filter_triples_main
from src.llm.llm_structural_enrich import llm_structural_enrich_main
from src.llm.llm_feature_enrich import llm_feature_enrich_main
from src.llm.llm_prune_recall_v1 import llm_prune_recall_v1

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse

def run(args):
    task = args.task
    if task == "initial_pruning":
        initial_prune(args.d,args.embedding_model,args.pruning_top_k)
    elif task == "llm_pruning":
        llm_prune(args.d,args.llm,args.embedding_model,args.pruning_top_k,args.llm_pruning_top_k,args.resume_path)
    elif task == "llm_pruning_three_channels":
        llm_prune_recall_v1(args.d,args.llm,args.embedding_model,args.pruning_top_k,args.llm_pruning_top_k,args.task,args.resume_path)
    elif task == "query_decompose":
        llm_decompose_query(args.d,args.llm,args.embedding_model,args.pruning_top_k,args.resume_path)
    elif task == "triple_trans":
        llm_trans_triple_main(args.d,args.llm,args.embedding_model,args.pruning_top_k,args.llm_pruning_top_k,args.resume_path)
    elif task == "filter_triples":
        llm_filter_triples_main(args.d,args.llm,args.resume_path)
    elif task == "structral_enrich":
        llm_structural_enrich_main(args.d,args.llm,args.resume_path)
    elif task == "feature_enrich":
        llm_feature_enrich_main(args.d,args.llm,args.resume_path)
    elif task == "structural_enrich_qa" or task == "feature_enrich_qa" or task == "mix_qa" or task == "direct_qa":
        llm_qa_main(args.d,args.llm,args.embedding_model,args.pruning_top_k,task,args.llm_pruning_top_k,args.resume_path)
    elif task == "qa_evaluate":
        eval_result_main(args.d,args.cal_f1,args.qa_eval_top_k,args.llm)
    else:
        print("功能未实现!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str,help="测试集名称", choices=["cwq","webqsp"],default="cwq")
    argparser.add_argument("--embedding_model",type=str,help="测试集路径",choices=["sentence-transformers"])
    argparser.add_argument("--llm",type=str,help="大模型",choices=["gpt3.5","gpt4o-mini"])
    argparser.add_argument("--task",type=str,help="执行任务",choices=["initial_pruning","llm_pruning","llm_pruning_three_channels","query_decompose","triple_trans","filter_triples","structral_enrich","feature_enrich","graph_eval","direct_qa","feature_enrich_qa","structural_enrich_qa","mix_qa","qa_evaluate"])
    argparser.add_argument('--eval_top_k',help="取topk个结果进行qa eval",type=int, default=-1)
    argparser.add_argument('--pruning_top_k',help="在initial剪枝的时候取相似度最高的top k个结果",type=int, default=750)
    argparser.add_argument('--llm_pruning_top_k',help="在llm剪枝的时候取相似度最高的top k个结果",type=int, default=100)
    argparser.add_argument('--qa_eval_top_k',help="在llm剪枝的时候取相似度最高的top k个结果",type=int, default=-1)
    argparser.add_argument('--cal_f1', action='store_true', help="在qa eval的时候使用f1分数",default=True)
    argparser.add_argument('--resume_path',type=str,help="从文件中重新执行任务,需要提供相对路径",default=None)

    args = argparser.parse_args()
    run(args=args)
