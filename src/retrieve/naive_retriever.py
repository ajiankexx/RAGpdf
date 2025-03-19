from pymilvus import MilvusClient
import torch
import torch.nn.functional as F
import numpy as np

from retrieve.embed import data_schema
from llm.base import llm_embedding

def naive_retriever(client: MilvusClient, chunks_name:str, query: str, topk: int = 4, output_fields=['id']):
    query_embed = torch.tensor([llm_embedding(i) for i in query])
    query_embed = F.normalize(query_embed, p=2, dim=-1)
    query_embed = list(map(np.float32, query_embed))

    return client.search(
        collection_name=chunks_name,
        data = query_embed,
        limit = topk,
        output_fields=output_fields
    )