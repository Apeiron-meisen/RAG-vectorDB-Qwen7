import json
from typing import Dict
import logger
import re
import time
from redis_tool import RedisClient
from concurrent.futures import ThreadPoolExecutor

MODEL_TO_LINE = {
    "R222EL": "A",
    "R223EL": "BL",
    "R223ML": "BH",
    "R224ML": "C",
    "R225ML": "D"
}

MODEL_TO_NAME = {
    "R222EL": "B9 1.5T 118KW 尊享版",
    "R223EL": "B9 1.5T 118KW 尊享版(显眼包)",
    "R223ML": "B9 2.0T 162KW 尊贵版",
    "R224ML": "B9 2.0T 162KW 尊贵版(大迈包)",
    "R225ML": "B9 2.0T 162KW 至尊版"
}


REDIS_EXPIRE_TIME = "60"
PROMPT_TEMPLATE_JUDGE = """
请选择正确的答案输出（单选）

{content}

【A】query的意图不具体，passage的内容与query相关
【B】query的意图不具体，passage的内容与query不相关
【C】query的意图明确，passage的内容能回答query
【D】query的意图明确，passage的内容不能回答query

请输出正确答案选项字母："""

PROMPT_TEMPLATE_NO_CONTEXT_JUDGE = """
请选择正确的答案输出（单选）

【A】"{query}"是车控指令，是疑问句
【B】"{query}"是车控指令，不是疑问句
【C】"{query}"是车知识问题，是疑问句
【D】"{query}"是车知识问题，不是是疑问句
【E】"{query}"是非汽车领域的疑问句
【F】以上都不是

请输出正确答案选项字母："""

Q_PATTERNS = [
    "灯泡",
    "机油",
    "充电",
    "小红人",
    "大宝剑",
    "几号油",
    "黑化饰条",
    "(智能驾驶|自动驾驶|智驾).*开关"
]


REDIS_CLIENT = RedisClient()
THREAD_POOL = ThreadPoolExecutor(max_workers=20)


def run_llm_context_judge(llm, prompt, query):
    if len(query) <= 20:
        for pattern in Q_PATTERNS:
            if re.search(pattern, query):
                return "C"
    return llm(prompt)


def run_llm_no_context_judge(llm, prompt):
    return llm(prompt)


def request_llm_generate(llm, system_content, prompt):
    strat_time = time.time()
    resp = llm(system_content, prompt)
    logger.info(f"request_llm_generate time: {time.time() - strat_time}")
    return resp


def get_image_url(text, prompter):
    numbers_string_list = re.findall(r"【(.+?)】", text)
    if not numbers_string_list:
        return "INVALID"
    numbers = [n.strip() for n in numbers_string_list[0].split(",")]
    selected_docs = []
    for n in numbers:
        sd = prompter.get_document_by_choice(n)
        if sd:
            selected_docs.append(sd)
    if not selected_docs:
        return "NULL"

    images = []
    for sd in selected_docs:
        for img in sd.get("images", []):
            if img not in images:
                images.append(img)
    if not images:
        return "NULL"

    return images[0]


class QueryEngine:

    def __init__(self,
                 retriever,
                 judge_prompter_class,
                 qa_prompter_class,
                 llm_context_judge,
                 llm_generate,
                 llm_no_context_judge,
                 max_completion_time=1.5):
        self.retriever = retriever
        self.judge_prompter_class = judge_prompter_class
        self.qa_prompter_class = qa_prompter_class
        self.llm_context_judge = llm_context_judge
        self.llm_generate = llm_generate
        self.llm_no_context_judge = llm_no_context_judge
        self.max_completion_time = max_completion_time

    def run(self, message: Dict):
        if not message["text"]:
            yield {"status": -1}
            return

        model_code = message.get("vehicleModelCode", "")
        if model_code not in MODEL_TO_LINE:
            yield {"status": -1}
            return

        t1 = time.time()
        message["vehicle_model"] = MODEL_TO_LINE[model_code]
        retrieved_documents = self.retriever.retrieve(message)
        t2 = time.time()
        logger.info(f"Retrieve time: {t2 - t1}")
        if not retrieved_documents:
            yield {"status": -1}
            return

        start_time = time.time()
        prompter_context_judge = self.judge_prompter_class(message["text"],
                                                           retrieved_documents,
                                                           "",
                                                           PROMPT_TEMPLATE_JUDGE)
        prompt_no_context_judge = PROMPT_TEMPLATE_NO_CONTEXT_JUDGE.format(query=message["text"])
        logger.info(prompt_no_context_judge)
        logger.info(prompter_context_judge.prompt)

        handler_context_judge = THREAD_POOL.submit(run_llm_context_judge,
                                                   self.llm_context_judge,
                                                   prompter_context_judge.prompt,
                                                   message["text"])
        handler_no_context_judge = THREAD_POOL.submit(run_llm_no_context_judge,
                                                      self.llm_no_context_judge,
                                                      PROMPT_TEMPLATE_NO_CONTEXT_JUDGE.format(query=message["text"]))

        prompter_qa = self.qa_prompter_class(message["text"],
                                             retrieved_documents,
                                             MODEL_TO_NAME[model_code])
        logger.info(prompter_qa.prompt)
        system_content = f"你是{MODEL_TO_NAME[model_code]}的问答系统"
        handler_generate = THREAD_POOL.submit(request_llm_generate,
                                              self.llm_generate,
                                              system_content,
                                              prompter_qa.prompt)

        context_judge_result = handler_context_judge.result()
        logger.info(f"context_judge_result: {context_judge_result}")
        no_context_judge_result = handler_no_context_judge.result()
        logger.info(f"no_context_judge_result: {no_context_judge_result}")

        logger.info(f"Judge cost time: {time.time() - start_time}")

        if context_judge_result == "C" and \
                (retrieved_documents[0]["is_standard_query"] or no_context_judge_result not in ["B", "F"]):
            resp_generate = handler_generate.result()
            incompletion = ""
            index = 0
            is_image_processed = False
            for chunk in resp_generate.iter_lines():
                chunk = chunk.decode('utf-8')
                if chunk.startswith('data:'):
                    content = chunk[len('data:'):].strip()
                    if content == '[DONE]':
                        incompletion = re.sub(r"\{|\}", "", incompletion)
                        yield {"status": 1, "text": incompletion, "index": index}
                        break
                    content = json.loads(content)
                    if "content" in content["choices"][0]["delta"]:
                        incompletion += content["choices"][0]["delta"]["content"]
                        if not is_image_processed:
                            image_url = get_image_url(incompletion, prompter_qa)
                            if image_url == "INVALID":
                                continue
                            else:
                                is_image_processed = True
                                if image_url != "NULL":
                                    yield {"status": 0, "image": image_url, "index": index}
                                    index += 1
                                incompletion = re.sub(r"【.+?】", "", incompletion)

                        if re.search(r"！|。|，|；|？|,|\?", incompletion):
                            text = incompletion
                            incompletion = ""
                            text = re.sub(r"B9 1.5T 118KW 尊享版", "迈腾B9 1.5T 尊享版", text)
                            text = re.sub(r"B9 1.5T 118KW 尊享版(显眼包)", "迈腾B9 1.5T 尊享版(显眼包)", text)
                            text = re.sub(r"B9 2.0T 162KW 尊贵版", "迈腾B9 2.0T 尊贵版", text)
                            text = re.sub(r"B9 2.0T 162KW 尊贵版(大迈包)", "迈腾B9 2.0T 尊贵版(大迈包)", text)
                            text = re.sub(r"B9 2.0T 162KW 至尊版", "迈腾B9 2.0T 至尊版", text)
                            text = re.sub(r"\{|\}", "", text)
                            yield {"status": 0, "text": text, "index": index}
                            index += 1
        else:
            yield {"status": -1}
