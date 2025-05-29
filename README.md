# Enrich-on-Graph
本仓库是Enrich-on-Graph: Query-Graph Alignment for Complex Reasoning with LLM Enriching的代码实现
## Update
- [x] 后续将继续更新，仓库地址: https://github.com/returnlsz/KGEnrich
## 使用说明
常用指令见当前目录下的cmd.ipynb
### Question Answering
运行主脚本,执行webqsp的initial direct_qa 
> python run.py -d "webqsp" --embedding_model "sentence-transformers" --llm "gpt4o-mini" --task "direct_qa" --eval_top_k -1 --pruning_top_k 750

运行主脚本,执行webqsp的llm direct_qa
> python run.py -d "webqsp" --embedding_model "sentence-transformers" --llm "gpt4o-mini" --task "direct_qa" --eval_top_k -1 --pruning_top_k 750 --llm_pruning_top_k 100
### Query Decomposing
运行主脚本,执行webqsp的query decompose
> python run.py -d "webqsp" --embedding_model "sentence-transformers" --llm "gpt4o-mini" --task "query_decompose" --eval_top_k -1 --pruning_top_k 750

运行主脚本,执行cwq的query decompose
> python run.py -d "cwq" --embedding_model "sentence-transformers" --llm "gpt4o-mini" --task "query_decompose" --eval_top_k -1 --pruning_top_k 750
### Triple Transaction
> python run.py -d "cwq" --embedding_model "sentence-transformers" --llm "gpt4o-mini" --task "triple_trans" --eval_top_k -1 --pruning_top_k 750 --llm_pruning_top_k 100
