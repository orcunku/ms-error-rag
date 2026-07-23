import random
import numpy as np
from src.retrieval import Retriever, tokenize
from src import config

ON_TOPIC = [
    "folder access denied as admin", "printer stuck in queue",
    "windows update keeps failing", "blue screen on startup",
    "app crashes when opening a file", "network drops randomly",
    "system not activated", "disk read errors",
]
OFF_TOPIC = [
    "how do I bake sourdough bread", "what is the capital of France",
    "best exercises for lower back pain", "explain quantum entanglement",
    "who won the world cup in 2018", "recipe for chicken curry",
    "how to change a car tyre", "what is the meaning of life",
]


def exact_code_recall(r, n=50, seed=0):
    rng = random.Random(seed)
    sample = rng.sample(range(len(r.chunks)), n)
    dense = hybrid = 0
    for i in sample:
        code = r.chunks[i]["error_code"]
        q = f"how do I fix error {code}"
        qv = r.encode(q)
        if i in np.argsort(r.embeddings @ qv)[::-1][:3]:
            dense += 1
        if any(c["error_code"] == code for c, _ in r.search(q)):
            hybrid += 1
    return dense / n, hybrid / n


def refusal_separation(r):
    on = [r.best_score(q) for q in ON_TOPIC]
    off = [r.best_score(q) for q in OFF_TOPIC]
    return min(on), max(off), min(on) - max(off)


if __name__ == "__main__":
    r = Retriever()

    dense, hybrid = exact_code_recall(r)
    print(f"exact-code recall@3   dense {dense:.0%}   hybrid {hybrid:.0%}")

    lo, hi, gap = refusal_separation(r)
    print(f"refusal separation    on-topic min {lo:.3f}  off-topic max {hi:.3f}  gap {gap:+.3f}")
    print(f"threshold {config.CONFIDENCE_THRESHOLD} "
          f"{'sits in the gap' if hi < config.CONFIDENCE_THRESHOLD < lo else 'FAILS — outside the gap'}")