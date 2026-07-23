![Retrieval Evaluation](https://github.com/orcunku/ms-error-rag/actions/workflows/eval.yml/badge.svg)

**[Live demo](https://ms-error-rag.onrender.com)** — free tier, first request after
inactivity takes ~50 seconds to wake.
# Windows Error Assistant

A retrieval-augmented question answering system over a synthetic Windows error
knowledge base. Built to explore retrieval quality and refusal behaviour rather
than to be a production support tool.

## Results
![Windows Error Assistant](screenshot.png)

Answer relevancy (RAGAS, 8 questions): **0.94**

| Metric | Dense only | Hybrid |
|---|---|---|
| Exact error-code recall@3 (50 queries) | 56% | **100%** |

Refusal threshold validated against a 16-query set: on-topic scores bottom out
at 0.692, off-topic top out at 0.524. The threshold of 0.60 sits in that gap.

Reproduce with `python -m eval.benchmark`.

## What the numbers mean

**Dense retrieval is unreliable for error codes.** Embeddings encode meaning,
and one hex string means roughly the same as another. The 56% baseline is
misleading rather than encouraging — it comes from partial character overlap on
the shared `0x8` prefix, so it succeeds often enough to look functional while
failing unpredictably. BM25 keyword matching solves this directly.

**The fusion constant mattered more than the fusion.** Reciprocal Rank Fusion
at the common default of `rrf_k=60` scored 82%. Sweeping the parameter:

| rrf_k | 60 | 20 | 10 | 5 |
|---|---|---|---|---|
| recall@3 | 82% | 86% | 92% | 100% |

Large values flatten rank differences so aggressively that a strong exact match
gets averaged away by a weak semantic one. An 18-point gain came from tuning a
constant, not from adding a component.

**Similarity scores are not calibrated.** `bge-small` compresses scores into
roughly 0.4–0.9, so an unrelated question scores 0.46 rather than near zero. An
initial threshold of 0.35 never fired. Only the relative separation between
on- and off-topic queries carries usable signal.

## Architecture

Query → confidence check → metadata filter → hybrid retrieval (dense + BM25,
RRF) → grounded generation with citations.

The confidence check runs before the LLM call, so refusals cost nothing.

- **Embeddings** `BAAI/bge-small-en-v1.5`, local, no API
- **Vector store** ChromaDB, persistent
- **Keyword** BM25Okapi
- **Generation** Llama 3.3 70B via Groq
- **UI** Gradio

## Data

All 300 error documents are synthetic — generated from templates with invented
hex codes. No Microsoft content is reproduced, which avoids licensing
constraints entirely. Generation is seeded and reproducible.

This is a deliberate trade-off, and it has a visible cost: entries within a
product family share phrasing, so retrieved chunks for a semantic query are
often near-identical (similarity spread of ~0.004 across the top 3). The model
responds by hedging — listing all candidates instead of committing to one. A
production version would need genuinely varied source documents.

## Known limitations

- Exact-code recall is measured on the query template `"how do I fix error {code}"`,
  which closely matches document phrasing. Real users paste raw error text; that
  case is untested.
- The refusal threshold is tuned to this embedding model and this corpus. It
  would need re-validation after changing either.
- No reranking stage. With near-duplicate candidates it would have little to work
  with; worth revisiting on more varied data.

## Running it

```bash
pip install -r requirements.txt
python -m src.data_gen
python -m src.ingestion
export GROQ_API_KEY="your_key"
python app.py
```

## Deployment constraints

Deployed to a 512MB container, which drove three changes:

- **Dropped ChromaDB.** Retrieval ran entirely on a NumPy array and BM25; the
  vector database was installed and loaded but never queried.
- **Replaced sentence-transformers with fastembed.** Same model, ONNX runtime
  instead of PyTorch, substantially lower memory.
- **Moved embedding computation to build time.** Recomputing 300 embeddings at
  every startup was the remaining memory spike.

The first of these is worth stating plainly: the vector database was unnecessary
for a corpus this size, and removing it simplified the system.
