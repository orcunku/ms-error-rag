from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "errors.json"
CHROMA_PATH = str(ROOT / "chroma_db")
COLLECTION_NAME = "ms_errors"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"
BM25_WEIGHT = 1.2
N_ERRORS = 300
SEED = 42

# Tuned: recall@3 on 50 exact-code queries was 82% at rrf_k=60,
# 92% at 10, 100% at 5. Lower k lets exact BM25 matches dominate.
RRF_K = 5

# Tuned: 16-query validation showed on-topic min 0.692,
# off-topic max 0.524. 0.60 is the midpoint of that 0.167 gap.
CONFIDENCE_THRESHOLD = 0.60

TOP_K = 3