from openai import OpenAI
import os
import asyncio
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SILICONFOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

BASE_URL = {
    "siliconflow":"https://api.siliconflow.cn/v1",
    "deepseek":"https://api.deepseek.com"
}
API_KEY = {
    "siliconflow":SILICONFOW_API_KEY,
    "deepseek":DEEPSEEK_API_KEY
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

def main():
    messages=[{"role":"user","content":"hello"}]
    prompts = ["hello owrld", "how are you?"]

    chat_task = llm_chat(messages)
    print("llm_chat: ", chat_task)
    embedding_task = [llm_embedding(promt)[:20] for promt in  prompts]
    print("embedding: ", embedding_task)
    
if __name__=="__main__":
    main()