from llm.base import llm_rerank

def simple_rerank(query: str, chunks: list[str], topk: int = 5)->list[str]:
	resp = llm_rerank(query, chunks)
	rank = [i["index"] for i in resp["results"]]
	return [chunks[rank[i]] for i in range(topk)]
