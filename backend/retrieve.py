import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent / "backend" / "data"

CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_PATH  = DATA_DIR / "faiss_index.bin"

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────

TOP_K                = 5
SIMILARITY_THRESHOLD = 0.45
MODEL_NAME           = "all-MiniLM-L6-v2"

# ────────────────────────────────────────────────────────────────
# Load assets at module import (cached for all queries)
# ────────────────────────────────────────────────────────────────

print("Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"  {len(chunks):,} chunks loaded")

print("Loading FAISS index...")
index = faiss.read_index(str(FAISS_PATH))
print(f"  {index.ntotal:,} vectors in index")

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("  Ready")

# ────────────────────────────────────────────────────────────────
# Build per-religion chunk ID lists for fast filtered search
# ────────────────────────────────────────────────────────────────
# Pre-computing these at load time means search() never has to
# iterate all chunks — it only scores the relevant religion's IDs.

_religion_ids: dict[str, list[int]] = {}
for i, chunk in enumerate(chunks):
    rel = chunk.get("religion", "unknown")
    _religion_ids.setdefault(rel, []).append(i)

print(f"  Religions indexed: {list(_religion_ids.keys())}")

# ────────────────────────────────────────────────────────────────
# Search
# ────────────────────────────────────────────────────────────────

def search(
    query: str,
    religion: str = "Buddhism",
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    language: str = "en",
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Filtering order (matches development research spec):
      1. Religion namespace  — prevents cross-religion contamination
      2. Language            — returns only chunks in the requested language
      3. Similarity threshold — confidence gate (section 6 of the research doc)

    Returns a list of dicts with keys:
      text, book, pitaka, source, religion, language, score
    """

    # 1. Embed and L2-normalise the query (must match how index was built)
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    # 2. FAISS search — retrieve more than top_k so we have room to filter
    fetch_k   = min(top_k * 10, index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)
    scores    = scores[0]
    indices   = indices[0]

    # 3. Filter by religion + language + threshold
    allowed_ids = set(_religion_ids.get(religion, []))

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:                           # FAISS padding
            continue
        if idx not in allowed_ids:              # wrong religion
            continue
        chunk = chunks[idx]
        if chunk.get("language", "en") != language:  # wrong language
            continue
        if float(score) < threshold:            # below confidence gate
            continue
        results.append({
            "text":     chunk["text"],
            "book":     chunk["book"],
            "pitaka":   chunk.get("pitaka", ""),
            "source":   chunk["source"],
            "religion": chunk["religion"],
            "language": chunk.get("language", "en"),
            "score":    float(score),
        })
        if len(results) >= top_k:
            break

    return results

# ────────────────────────────────────────────────────────────────
# CLI runner
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRetrieval test  (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            q = input("\nQuery: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if q.lower() in ("quit", "exit", "q"):
            break

        if not q:
            continue

        res = search(q)

        if not res:
            print("No results above threshold.")
            continue

        for r in res:
            print(f"\n[{r['book']} | {r['pitaka']}]  score={r['score']:.3f}")
            print(r["text"][:300] + ("…" if len(r["text"]) > 300 else ""))