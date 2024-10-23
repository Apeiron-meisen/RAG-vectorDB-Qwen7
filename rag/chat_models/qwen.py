import requests
from typing import Text
import json
import time
import logger
import dotenv,os
dotenv.load_dotenv()

LLM_GENERATE_URL = os.getenv("LLM_GENERATE_URI") + "/v1/chat/completions"
LLM_JUDGE_URL = os.getenv("LLM_JUDGE_URI") + "/v1/chat/completions"
LLM_QUERY_REWRITE_URL = os.getenv('LLM_QUERY_REWRITE_URL') +  "/v1/chat/completions"

headers = {
    'Content-Type': 'application/json'
}


class QwenGenerate:
    def __call__(self, system_content: Text, prompt: Text):
        data = {
            'model': 'qwen',
            'messages': [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0,
            "stream": True
        }
        return requests.post(LLM_GENERATE_URL,
                             headers=headers,
                             data=json.dumps(data),
                             stream=True,
                             timeout=2.5)


class QwenContextJudge:
    def __call__(self, prompt: Text, timeout=1.9):
        start_time = time.time()
        data = {
            'model': 'qwen',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0
        }
        response = requests.post(LLM_JUDGE_URL,
                                 headers=headers,
                                 data=json.dumps(data),
                                 timeout=timeout)
        response.raise_for_status()
        response = response.json()

        logger.info(f"Context judge cost time: {time.time() - start_time}")

        return response["choices"][0]["message"]["content"]


class QwenNoContextJudge:
    def __call__(self, prompt: Text, timeout=1.9):
        start_time = time.time()
        data = {
            'model': 'qwen',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0
        }
        response = requests.post(LLM_JUDGE_URL,
                                 headers=headers,
                                 data=json.dumps(data),
                                 timeout=timeout)
        response.raise_for_status()
        response = response.json()

        logger.info(f"No context judge cost time: {time.time() - start_time}")

        return response["choices"][0]["message"]["content"]


class QwenQueryRewriter:
    def __call__(self, prompt: Text, timeout=0.4):
        data = {
            'model': 'qwen',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0
        }

        response = requests.post(LLM_QUERY_REWRITE_URL,
                                 headers=headers,
                                 data=json.dumps(data),
                                 timeout=timeout)
        response.raise_for_status()
        response = response.json()

        return response["choices"][0]["message"]["content"]
