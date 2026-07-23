import json
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from src import config


def tokenize(text):
    return text.lower().replace(".", " ").replace(",", " ").split()


class Retriever:
    def __init__(self):
        with open(config.ROOT / "data" / "chunks.json") as f:
            self.chunks = json.load(f)

        self.model = TextEmbedding(model_name=config.EMBED_MODEL)
        self.embeddings = np.array(list(self.model.embed([c["text"] for c in self.chunks])))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def encode(self, text):
        return np.array(list(self.model.embed([text])))[0]

    def best_score(self, question):
        return float((self.embeddings @ self.encode(question)).max())

    def search(self, question, os_version=None, k=config.TOP_K, rrf_k=config.RRF_K):
        q_vec = self.encode(question)
        dense_rank = np.argsort(self.embeddings @ q_vec)[::-1]

        sparse_scores = self.bm25.get_scores(tokenize(question))
        sparse_rank = np.argsort(sparse_scores)[::-1]

        fused = {}
        for rank, idx in enumerate(dense_rank[:50]):
            fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_rank[:50]):
            if sparse_scores[idx] > 0:
                fused[idx] = fused.get(idx, 0) + config.BM25_WEIGHT / (rrf_k + rank + 1)

        results = []
        for idx, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            chunk = self.chunks[idx]
            if os_version and chunk["os_version"] != os_version:
                continue
            results.append((chunk, score))
            if len(results) == k:
                break
        return results


if __name__ == "__main__":
    r = Retriever()
    code = r.chunks[0]["error_code"]
    print("query:", code)
    for chunk, score in r.search(f"how do I fix error {code}"):
        print(f"  {score:.4f}  {chunk['error_code']}  {chunk['product']}")