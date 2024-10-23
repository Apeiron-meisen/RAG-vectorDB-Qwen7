import json
import re
import time
from flask import Flask, Response, request

from rag.chat_models import QwenGenerate, QwenContextJudge, QwenNoContextJudge
from rag.query_engine import QueryEngine
from rag.retrievers import BaseRetriever
from rag.prompters import QAPrompter, JudgePrompter
from rag.stores.api_db import APIStore

app = Flask(__name__)


def load_query_engine():
    llm_generate = QwenGenerate()
    llm_context_judge = QwenContextJudge()
    llm_no_context_judge = QwenNoContextJudge()
    vdb = APIStore()
    retriever = BaseRetriever(50, 20, vdb)
    qe = QueryEngine(retriever=retriever,
                     judge_prompter_class=JudgePrompter,
                     qa_prompter_class=QAPrompter,
                     llm_generate=llm_generate,
                     llm_context_judge=llm_context_judge,
                     llm_no_context_judge=llm_no_context_judge)
    return qe


QUERY_ENGINE = load_query_engine()


def stream(query, trace_id, vin):
    message = {
        "trace": trace_id,
        "brand": "xx",
        "vehicleProject": "xxx",
        "vehicleModelCode": "R225ML",
        "vin": vin
    }
    message["text"] = query
    for chunk in QUERY_ENGINE.run(message):
        print(chunk)
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        time.sleep(0.2)


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    print(data)
    trace_id = data.get("trace_id", "")
    vin = data.get("vin", "")
    query = data.get("query", "")
    return Response(stream(query, trace_id, vin), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7116)
