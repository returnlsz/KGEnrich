import sys
import os
import asyncio
import httpx

# 添加路径
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from llm.prompt_builder import *

class llm_client:
    def __init__(self, base_url="https://api.key77qiqi.cn/v1", api_keys=None, model='gpt-4o-mini-2024-07-18'):
        """
        初始化 LLM 客户端
        :param base_url: API 的基础地址
        :param api_keys: 包含多个 API Key 的列表
        :param model: 使用的模型名称,gpt-4o,gpt-4o-mini,gpt-4o-mini-2024-07-18
        """
        self.base_url = base_url
        # 如果未提供 api_keys,则使用一个默认的 API Key
        self.api_keys = api_keys or ["sk-SgiEuM72oCrNUpDZ9b87F351103e4d218d69B42e36C859Df","sk-rh8wi5OXclyhFu7spfK69E7UHU5BkOdIqRsl0xslPiFRgQg3"]
        self.model = model  # 模型名称
        self.api_key_index = 0  # 用于跟踪当前正在使用的 API Key

    def switch_api_key(self):
        """
        切换到下一个 API Key。如果达到列表末尾,则从头开始循环。
        """
        self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
        print(f"切换到新的 API Key: {self.api_keys[self.api_key_index]}")

    async def response(self, question, mode):
        """
        向 API 发送请求并返回响应。
        :param question: 用户提出的问题
        :param mode: 模式（例如对 prompt 的不同处理方式）
        :return: 模型生成的响应内容
        """
        # prompt_builder = PromptBuilder(question=question, mode=mode)
        # print("当前prompt如下:",prompt_builder.build_prompt())
        url = f"{self.base_url}/chat/completions"
        # print("当前prompt如下:",question)
        # 设置最大重试次数
        retries = 100

        # 开始重试逻辑
        for attempt in range(retries):
            # 设置当前的 API Key 和请求头
            current_api_key = self.api_keys[self.api_key_index]
            headers = {
                "Authorization": f"Bearer {current_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 16000,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": 1
            }

            try:
                # 使用 httpx.AsyncClient 发送请求
                async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), proxies=None) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()  # 如果状态码不是 2xx,则抛出异常
                    result = response.json()
                    return result["choices"][0]["message"]["content"]  # 返回结果

            except httpx.ConnectTimeout:
                print(f"Attempt {attempt + 1}: 连接超时,正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key
            except httpx.ReadTimeout:
                print(f"Attempt {attempt + 1}: 读取超时,正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key
            except httpx.ConnectError as exc:
                print(f"Attempt {attempt + 1}: 连接错误 {exc},正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key
            except httpx.HTTPStatusError as exc:
                # 特殊处理 429 错误
                if exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after is not None:
                        wait_time = int(retry_after)
                    else:
                        wait_time = 5  # 默认等待时间（5秒）
                    print(f"Attempt {attempt + 1}: 请求频率过高 (429 Too Many Requests)，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)  # 等待指定时间后重试
                # 检查是否遇到 5xx 系列的错误
                elif 500 <= exc.response.status_code < 600:
                    print(f"Attempt {attempt + 1}: 服务器错误 {exc.response.status_code},正在重试...")
                    self.switch_api_key()  # 切换到下一个 API Key
                else:
                    print(f"HTTP 错误: {exc.response.status_code}。")
                    raise
            except Exception as exc:
                print(f"未知错误: {exc},正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key

            # 如果达到最大重试次数,则抛出异常
            if attempt == retries - 1:
                print(f"最大重试次数已到 ({retries})，请求失败。")
                raise

            # 添加延迟以避免频繁请求，但优先等待 429 的 Retry-After
            await asyncio.sleep(2)  # 等待 2 秒后再次尝试
