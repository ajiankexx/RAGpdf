from openai import AsyncOpenAI
import os
import asyncio
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SILICONFOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

retry_config = retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception),
       reraise=True
)

@retry_config
async def llm_chat(messages: list)->str:
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY)
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )
    if resp.choices[0].message.content:
        return resp.choices[0].message.content
    else:
        return "Empty Response"

@retry_config
async def llm_embedding(prompt: str)->list:
    client = AsyncOpenAI(base_url="https://api.siliconflow.cn/v1",
                    api_key=SILICONFOW_API_KEY)
    resp = await client.embeddings.create(
        input=prompt,
        model="BAAI/bge-m3",
        encoding_format="float"
    )
    return resp.data[0].embedding

async def main():
    messages=[{"role":"user","content":"hello"}]
    prompts = ["hello owrld", "how are you?"]
    

    chat_task = llm_chat(messages)
    embedding_task = [llm_embedding(promt) for promt in  prompts]
    results = await asyncio.gather(chat_task, *embedding_task)
    
    print(results[0])
    for i, embedding in enumerate(results[1:], 1):
        print(f"embedding {i} (first 5): ", embedding[:5])

    
if __name__=="__main__":
    asyncio.run(main())
