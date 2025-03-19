from pdf_parse.simple_parser import simple_parser
from achunk.simple_chunker import simple_chunker
# from RAGpdf.src.achunk.simple_chunker import simple_chunker
from retrieve.embed import vector_db_embed
from retrieve.naive_retriever import naive_retriever

from pymilvus import MilvusClient

file_path = "/root/llm/RAGpdf/pdf/Chain_of_Retrieval_Augumented_Generation.pdf"
chunks_name = "Chain_of_Retrieval_Augumented_Generation"
query = ['论文的introduction主要讲了什么内容？']
client = MilvusClient("/root/llm/RAGpdf/vectordb/paper.db")

pdf_content = simple_parser(file_path)
chunks = simple_chunker(pdf_content)

vector_db_embed(client, chunks, chunks_name)
search_result = naive_retriever(client, chunks_name, query)
source = [[i['entity']['text'] for i in single] for single in search_result]
for i in source:
    for j in i:
        print(j)
        print("*****************")
