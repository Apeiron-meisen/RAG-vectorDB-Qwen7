from typing import Dict, List
import requests

from rag.stores.api_db import APIStore


class BaseRetriever:
    def __init__(self, top_k:int, limit:int, vdb_store:APIStore):
        self.limit = limit
        self.top_k = top_k
        self.vdb_store = vdb_store

    def retrieve(
            self, message: Dict
    ) -> List[Dict]:

        documents = self.vdb_store.query(query_str=message["text"], vehicle_model=message["vehicle_model"])

        return documents[:self.limit]
