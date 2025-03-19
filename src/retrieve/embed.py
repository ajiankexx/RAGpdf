from pymilvus import MilvusClient
import logging

from llm.base import llm_embedding

data_schema = {"id":None, "vector":None, "text":None, "subject":None}
def vector_db_embed(client: MilvusClient,  chunks: list[str], chunks_name: str, embed_dim=1024, data_schema=data_schema):
    has_chunks_name = client.has_collection(collection_name=chunks_name)
    if has_chunks_name:
        client.drop_collection(collection_name=chunks_name)

    client.create_collection(
        collection_name=chunks_name,
        dimension=embed_dim
    )

    chunks_vec = []
    for idx, chunk in enumerate(chunks):
        try:
            chunk_vec = llm_embedding(chunk)
            chunks_vec.append(chunk_vec)
        except Exception as e:
            chunks_vec.append([0]*embed_dim)
            logging.error(f"{chunk} failed.")
    data = [
        {"id":i, "vector":chunks_vec[i], "text":chunks[i], "subject":"history"} for i in range(len(chunks_vec))
    ]
    client.insert(collection_name=chunks_name, data=data)
