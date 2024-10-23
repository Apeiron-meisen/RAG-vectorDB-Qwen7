from typing import Text
import tcvectordb
from tcvectordb.model.document import Document, SearchParams
from tcvectordb.model.enum import FieldType, IndexType, MetricType, ReadConsistency
import os
from rag.embedders import BGEEmbedder
from tcvectordb.model.document import Document, Filter

client = tcvectordb.VectorDBClient(url=os.environ.get("TENCENT_VDB_URL"),
                                   username=os.environ.get("TENCENT_VDB_USERNAME"),
                                   key=os.environ.get("TENCENT_VDB_KEY").replace("**", ""))
db = client.database('faq')
collection = db.collection("faq-answers")
ef = BGEEmbedder()


def get_best_answer(faq_id: Text, question: Text):
    results = collection.search(
        vectors=ef([question]),
        filter=Filter(f'faqId="{faq_id}"'),
        params=SearchParams(ef=200),
        limit=1,
        output_fields=['answer']
    )

    answer = [{"answerStep": 1, "contents": [{"answerType": "TEXT", "answerResult": []}]}]
    answer[0]["contents"][0]["answerResult"].append(results[0][0]["answer"])

    return answer
