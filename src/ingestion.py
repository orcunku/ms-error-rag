import json
from sentence_transformers import SentenceTransformer
import chromadb
from src import config


def build_chunks(errors):
    chunks = []
    for e in errors:
        text = (
            f"Error code {e['error_code']} on {e['os_version']} ({e['product']}). "
            f"{e['symptom']} {e['cause']} "
            f"Solution: {e['solution']}"
        )
        chunks.append({
            "text": text,
            "error_code": e["error_code"],
            "os_version": e["os_version"],
            "product": e["product"],
        })
    return chunks


def ingest():
    with open(config.DATA_FILE) as f:
        errors = json.load(f)

    chunks = build_chunks(errors)
    print(f"built {len(chunks)} chunks")

    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[f"err_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {"error_code": c["error_code"],
             "os_version": c["os_version"],
             "product": c["product"]}
            for c in chunks
        ],
    )

    # BM25 rebuilds from these at query time
    with open(config.ROOT / "data" / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"stored {collection.count()} documents in {config.CHROMA_PATH}")


if __name__ == "__main__":
    ingest()