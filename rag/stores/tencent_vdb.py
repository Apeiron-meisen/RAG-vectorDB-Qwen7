from typing import Text, Dict, List
import os
import tcvectordb
from tcvectordb.model.document import SearchParams, Filter


class TencentVDBStore:

    def __init__(self, embedd_func):
        client = tcvectordb.VectorDBClient(url=os.environ.get("TENCENT_VDB_URL"),
                                           username=os.environ.get("TENCENT_VDB_USERNAME"),
                                           key=os.environ.get("TENCENT_VDB_KEY").replace("**", ""))
        self.db = client.database('faq')
        self.collection = self.db.collection("faq-all")
        self.embedd_func = embedd_func

    def get(self,
            document_ids: List,
            output_fields: List) -> List:
        return self.collection.query(document_ids=document_ids,
                                     output_fields=output_fields,
                                     limit=len(document_ids))

    def query(self,
              query_str: Text,
              top_k: int,
              filters: Text,
              output_fields: List) -> List[Dict]:

        results = self.collection.search(
            vectors=self.embedd_func([query_str]),
            filter=Filter(filters),
            params=SearchParams(ef=200),
            limit=top_k,
            output_fields=output_fields
        )
        doc_list = []
        for i, docs in enumerate(results):
            for doc in docs:
                doc_list.append(doc)

        return doc_list
