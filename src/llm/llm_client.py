import sys
import os
import asyncio
import httpx

# 添加路径
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from llm.prompt_builder import *

class llm_client:
    def __init__(self, base_url="https://api.key77qiqi.cn/v1", api_key="sk-rh8wi5OXclyhFu7spfK69E7UHU5BkOdIqRsl0xslPiFRgQg3", model='gpt-4o-mini'):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model  # 模型名称

    async def response(self, question, mode):
        prompt_builder = PromptBuilder(question=question, mode=mode)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_builder.build_prompt()
                }
            ],
            "temperature": 0.2,
            "max_tokens": 16384,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1
        }

        # async with httpx.AsyncClient(timeout=httpx.Timeout(180.0),proxies=None) as client:
        #     retries = 20  # 最大重试次数
        #     for attempt in range(retries):
        #         try:
        #             response = await client.post(url, headers=headers, json=payload)
        #             response.raise_for_status()  # 检查是否有非 2xx 响应
        #             result = response.json()
        #             return result["choices"][0]["message"]["content"]  # 返回结果
        #         except httpx.ReadTimeout:
        #             print(f"Attempt {attempt + 1} failed due to timeout. Retrying...")
        #             if attempt == retries - 1:
        #                 raise  # 如果超出重试次数，抛出异常
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), proxies=None) as client:
            retries = 100  # 最大重试次数
            for attempt in range(retries):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()  # 检查是否有非 2xx 响应
                    result = response.json()
                    return result["choices"][0]["message"]["content"]  # 返回结果
                except httpx.ReadTimeout:
                    print(f"Attempt {attempt + 1} failed due to timeout. Retrying...")
                except httpx.ConnectError as exc:
                    print(f"Attempt {attempt + 1} failed due to connection error: {exc}. Retrying...")
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 524:
                        print(f"Attempt {attempt + 1} failed due to 524 status code (server timeout). Retrying...")
                    else:
                        print(f"HTTP error occurred: {exc}. Retrying...")
                # 如果达到最大重试次数，抛出异常
                if attempt == retries - 1:
                    print(f"Maximum retries reached after {retries} attempts.")
                    raise
                # 添加延迟以避免过快重复请求
                await asyncio.sleep(2)  # 等待 2 秒后再次尝试
        # async with httpx.AsyncClient(timeout=httpx.Timeout(180.0),proxies=None) as client:
        #     response = await client.post(url, headers=headers, json=payload)
        #     response.raise_for_status()
        #     result = response.json()
        # return result["choices"][0]["message"]["content"]