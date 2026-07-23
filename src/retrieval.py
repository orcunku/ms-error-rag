import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src import config


def tokenize(text):
    return text.lower().replace(".", " ").replace(",", " ").split()


class Retriever:
    """Hybrid dense + sparse retrieval over the error knowledge base.

    Loads the embedding model and both indexes once at construction so that
    per-query cost stays low. Reloading the model per request would make a
    web app unusably slow.
    """

    def __init__(self):
        with open(config.ROOT / "data" / "chunks.json") as f:
            self.chunks = json.load(f)

        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.embeddings = self.model.encode(
            [c["text"] for c in self.chunks],
            normalize_embeddings=True,
        )
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def best_score(self, question):
        """Highest cosine similarity against any chunk. Used by the guardrail."""
        q_vec = self.model.encode([question], normalize_embeddings=True)[0]
        return float((self.embeddings @ q_vec).max())

    def search(self, question, os_version=None, k=config.TOP_K, rrf_k=config.RRF_K):
        """Reciprocal Rank Fusion over dense and BM25 rankings.

        Fusion is by rank position rather than raw score, since cosine
        similarity and BM25 scores are not on a comparable scale. BM25 carries
        a slight weight advantage because an exact error-code match is a
        stronger signal than semantic similarity, which cannot meaningfully
        distinguish one hex code from another.
        """
        q_vec = self.model.encode([question], normalize_embeddings=True)[0]
        dense_rank = np.argsort(self.embeddings @ q_vec)[::-1]

        sparse_scores = self.bm25.get_scores(tokenize(question))
        sparse_rank = np.argsort(sparse_scores)[::-1]

        fused = {}
        for rank, idx in enumerate(dense_rank[:50]):
            fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank + 1)

        for rank, idx in enumerate(sparse_rank[:50]):
            # BM25 ranks all documents even when most score zero; without this
            # check, non-matching documents accrue points from position alone.
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

    print(f"exact code query: {code}")
    for chunk, score in r.search(f"how do I fix error {code}"):
        print(f"  {score:.4f}  {chunk['error_code']}  {chunk['product']}")

    print("\nsemantic query: folder access denied as admin")
    for chunk, score in r.search("folder access denied as admin"):
        print(f"  {score:.4f}  {chunk['error_code']}  {chunk['product']}")

    print("\nfiltered to Windows 11:")
    for chunk, score in r.search("folder access denied as admin", os_version="Windows 11"):
        print(f"  {score:.4f}  {chunk['os_version']}  {chunk['product']}")