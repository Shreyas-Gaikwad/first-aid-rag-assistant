import json
import faiss
import numpy as np
from pathlib import Path
from src.embeddings import embed_texts

CHUNKS_PATH = Path("data/extracted/chunks.json")


class VectorStore:
    def __init__(self):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        texts = [c["text"] for c in self.chunks]
        embeddings = embed_texts(texts).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query, k=5):
        q_emb = embed_texts([query]).astype("float32")
        _, idx = self.index.search(q_emb, k)
        return [self.chunks[i] for i in idx[0]]