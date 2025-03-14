import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from openai import OpenAI
from llm.prompt_builder import *

class llm_client:
    def __init__(self, base_url="https://api.key77qiqi.cn/v1", api_key="sk-rh8wi5OXclyhFu7spfK69E7UHU5BkOdIqRsl0xslPiFRgQg3", model='gpt-4o-mini'):

        # config
        self.client = OpenAI(base_url=base_url,\
        api_key=api_key)

        self.model = model # 模型名称

    def response(self,question,mode):
        prompt_builder = PromptBuilder(question=question,mode=mode)

        # 响应
        response = self.client.chat.completions.create(
            model = self.model,
            messages=[
            {
                "role": "user",
                "content": prompt_builder.build_prompt()
            }
            ],
            temperature=0.2,
            max_tokens=16384,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n = 1
        )
        return response.choices[0].message.content