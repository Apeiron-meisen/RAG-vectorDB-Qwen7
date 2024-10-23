import json
import chromadb
from typing import List, Text, Dict
import logger
import re
import requests
from redis_tool import RedisClient

ENCODE_URL = "http://10.88.0.16:30339/bge-m3-server/encode"
RERANK_URL = "http://10.88.0.16:30339/bge-m3-server/compute-score"
# ENCODE_URL = "http://10.89.72.17:5099/bge-m3-server/encode"
# RERANK_URL = "http://10.89.72.17:5099/bge-m3-server/compute-score"

REDIS_CLIENT = RedisClient()


def query_chunk(url, document_ids):
    params = {
        "documentIds": document_ids,
        "filter": "",
        "limit": None,
        "offset": 0,
        "outputFields": [
        ],
        "retrieveVector": False
    }
    results = requests.post(url, json=params)
    documents = []
    for doc in results.json():
        doc.pop("vector")
        doc_fields = doc.pop("docFields")
        for df in doc_fields:
            doc[df["name"]] = df["value"]
        documents.append(doc)

    return documents


def search_chunk(url,
                 vectors,
                 params,
                 output_fields,
                 filter,
                 limit):
    params = {
        "filter": filter,
        "limit": limit,
        "outputFields": output_fields,
        "params": params,
        "retrieveVector": False,
        "vectors": vectors
    }
    results = requests.post(url, json=params)
    documents = []
    for docs in results.json():
        for doc in docs:
            for df in doc.pop("docFields"):
                doc.pop("vector")
                doc[df["name"]] = df["value"]
            documents.append(doc)
    return documents


def encode(query_list):
    resp = requests.post(ENCODE_URL, json={"sentences": query_list})
    resp = resp.json()

    return resp["query_dense_vecs"]


def rerank(sentence_pairs, bath_size):
    params = {
        "sentence_pairs": sentence_pairs,
        "normalize": True,
        "batch_size": bath_size
    }
    resp = requests.post(RERANK_URL, json=params)
    resp = resp.json()

    return resp


def rrf_fusion(rankings, k=60):
    """
    Applies RRF fusion on the given rankings.

    Parameters:
    rankings (list of lists): A list of ranked lists, where each sublist is a list of document IDs ranked by one retrieval system.
    k (int): The RRF parameter. Default is 60.

    Returns:
    list: A list of document IDs sorted by their RRF scores.
    """
    from collections import defaultdict

    # Initialize a dictionary to hold RRF scores
    rrf_scores = defaultdict(float)

    # Calculate RRF scores for each document in each ranking
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            rrf_scores[doc_id] += 1 / (k + rank + 1)

    # Sort document IDs by their RRF scores in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    # Return only the document IDs
    return [doc_id for doc_id, score in sorted_docs]


def retrieve(query, parent_collection, child_collection):

    limit = 100
    threshold_rank_big = 0.5
    top_n = 32

    query_dense_vecs = encode([query])

    docs = child_collection.query(query_embeddings=query_dense_vecs, n_results=top_n)

    faq_ids = set()
    rank_small = []

    is_standard_query = False
    if docs["distances"][0][0] <= 0.01:
        is_standard_query = True

    for idx in docs["ids"][0]:
        idx = idx.split('-')[0]
        if idx not in faq_ids:
            rank_small.append(idx)
            faq_ids.add(idx)

    logger.info(f"Size of rank_small is {len(rank_small)}")

    docs = parent_collection.query(query_embeddings=query_dense_vecs, n_results=top_n)

    faq_ids = set()
    rank_big = []
    for idx, dist in zip(docs["ids"][0], docs["distances"][0]):
        if dist > threshold_rank_big:
            break
        if idx not in faq_ids:
            rank_big.append(idx)
            faq_ids.add(idx)

    rank = [rank_small]
    if rank_big:
        logger.info(f"Size of rank_big is {len(rank_big)}")
        rank.append(rank_big)
    rank = rrf_fusion(rank)

    logger.info(f"Size of RRF rank is {len(rank)}")

    doc_list = []
    sentence_pairs = []
    docs = parent_collection.get(ids=rank[:top_n])
    for idx, doc in zip(docs["ids"][0], docs["metadatas"]):
        doc["id"] = idx
        doc["standard_answer"] = doc.pop("stand_answer")
        doc_list.append(doc)
        sentence_pairs.append([query, f'{doc["standard_question"]}，{doc["standard_answer"]}'])

    scores = rerank(sentence_pairs, bath_size=top_n)
    doc_scores = [(doc, s) for doc, s in zip(doc_list, scores)]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Rerank is done")

    doc_list = [doc for doc, _ in doc_scores]
    for doc in doc_list:
        doc["is_standard_query"] = is_standard_query

    return doc_list


class APIStore:
    def __init__(self):
        client = chromadb.HttpClient(host="10.89.72.17", port=8000)
        self.parent_collection = client.get_collection(name="qa_parent")
        self.child_collection = client.get_collection(name="qa_child")

    def query(self,
              query_str: Text,
              vehicle_model: Text) -> List[Dict]:
        doc_list = retrieve(query_str, self.parent_collection, self.child_collection)
        return doc_list
