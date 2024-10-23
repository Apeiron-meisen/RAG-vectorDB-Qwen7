from typing import List, Text
import requests
from tqdm import tqdm
import os


EMBEDDING_SERVER_URL = os.environ.get("BGE_EMBEDDING_URL")


class BGEEmbedder:

    def __call__(self, texts: List[Text]) -> List:
        embeddings = []
        for text in texts:
            emb = requests.post(EMBEDDING_SERVER_URL, json={"text": text})
            embeddings.append(emb.json())

        return embeddings
