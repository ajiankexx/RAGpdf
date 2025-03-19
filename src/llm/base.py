import os
import json
import requests
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SILICONFOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

BASE_URL = {
    "siliconflow":"https://api.siliconflow.cn/v1",
    "deepseek":"https://api.deepseek.com",
    "rerank-siliconflow":"https://api.siliconflow.cn/v1/rerank"
}
API_KEY = {
    "siliconflow":SILICONFOW_API_KEY,
    "deepseek":DEEPSEEK_API_KEY,
    "rerank-siliconflow":SILICONFOW_API_KEY
}

retry_config = retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception),
       reraise=True
)

# 修改stream为true会导致返回类型改变，从而导致提示错误
@retry_config
def llm_chat(messages: list, base_url=BASE_URL['siliconflow'],
             api_key=API_KEY['siliconflow'],
             model_name="Qwen/Qwen2.5-7B-Instruct")->str:
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
    )
    if resp.choices[0].message.content:
        return resp.choices[0].message.content
    else:
        return "Empty Response"

@retry_config
def llm_embedding(prompt: str, base_url=BASE_URL['siliconflow'],
                  api_key=API_KEY['siliconflow'],
                  model_name="BAAI/bge-m3")->list:
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.embeddings.create(
        input=prompt,
        model=model_name,
        encoding_format="float"
    )
    return resp.data[0].embedding

@retry_config
def llm_rerank(query: str, documents: list[str], 
               base_url=BASE_URL['rerank-siliconflow'],
               api_key=API_KEY['rerank-siliconflow']) -> dict:
    """
    resp.text has below format(for example):
    {"id":"0195ae84b2ad76a4b9190a383958da67",
    "results":[{"index":0,"relevance_score":0.99983656},
               {"index":1,"relevance_score":0.004962942},
               {"index":3,"relevance_score":0.0012693645},
               {"index":2,"relevance_score":0.000017778551}],
    "meta":{"billed_units":{"input_tokens":59,"output_tokens":0,"search_units":0,"classifications":0},
            "tokens":{"input_tokens":59,"output_tokens":0}}}

    To get rank:
    rank = [i["index"] for i in resp['results']]
    """
    url = BASE_URL['rerank-siliconflow']
    payload = {
        "query":query,
        "documents":documents,
        "model":"BAAI/bge-reranker-v2-m3",
        "return_documents":False,
        "max_chunks_per_doc":1024,
        "overlap_tokens":80
    }
    headers = {
        "Authorization":f"Bearer {api_key}",
        "Content-Type":"application/json"
    }
    resp = requests.request("POST", url, json=payload, headers=headers)
    return json.loads(resp.text)

def main():
    messages=[{"role":"user","content":"hello"}]
    prompts = ["hello owrld", "how are you?"]

    chat_task = llm_chat(messages)
    print("llm_chat: ", chat_task)
    embedding_task = [llm_embedding(promt)[:20] for promt in  prompts]
    print("embedding: ", embedding_task)
    
    query = "who are you?"
    documents = ["who are you", "my name is peter, i am a student.", "China is a beautiful contry.", "I want to work for people."]
    resp = llm_rerank(query, documents)
    rank = [i["index"] for i in resp['results']]
    print(resp)
    print(rank)
    
if __name__=="__main__":
    main()