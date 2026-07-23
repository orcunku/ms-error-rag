import json
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


if __name__ == "__main__":
    with open(config.DATA_FILE) as f:
        errors = json.load(f)
    chunks = build_chunks(errors)
    with open(config.ROOT / "data" / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"wrote {len(chunks)} chunks")