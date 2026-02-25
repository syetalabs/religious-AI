import sqlite3
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────

from data_fetcher import ensure_data_files

DATA_DIR         = ensure_data_files()
CHUNKS_DB_PATH   = DATA_DIR / "chunks.db"
FAISS_PATH       = DATA_DIR / "faiss_index.bin"

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────

TOP_K                = 10
SIMILARITY_THRESHOLD = 0.45
MODEL_NAME           = "paraphrase-MiniLM-L3-v2"

# ────────────────────────────────────────────────────────────────
# Load FAISS index into RAM  
# ────────────────────────────────────────────────────────────────

print("Loading FAISS index...")
index = faiss.read_index(str(FAISS_PATH))
print(f"  {index.ntotal:,} vectors in index")

# ────────────────────────────────────────────────────────────────
# Load embedding model into RAM  
# ────────────────────────────────────────────────────────────────

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("  Ready")

# ────────────────────────────────────────────────────────────────
# Build per-religion ID filter from SQLite  
#
# Only loads chunk IDs and religion/language fields — NOT the text.
# chunk text is fetched on demand per query (see _fetch_chunks).
# ────────────────────────────────────────────────────────────────

print("Building religion index from SQLite...")
_con = sqlite3.connect(str(CHUNKS_DB_PATH), check_same_thread=False)
_con.row_factory = sqlite3.Row

_religion_ids: dict[str, list[int]] = {}
for row in _con.execute("SELECT id, religion FROM chunks"):
    _religion_ids.setdefault(row["religion"], []).append(row["id"])

print(f"  Religions indexed: {list(_religion_ids.keys())}")

# ────────────────────────────────────────────────────────────────
# Chunk fetcher — reads only the matched rows from SQLite
# ────────────────────────────────────────────────────────────────

def _fetch_chunks(ids: list[int]) -> dict[int, sqlite3.Row]:
    """
    Fetch full chunk rows for a list of FAISS result IDs.
    Uses a single SQL query with an IN clause — typically < 1 ms.
    Returns a dict keyed by chunk id for O(1) lookup.
    """
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _con.execute(
        f"SELECT id, text, book, pitaka, source, religion, language "
        f"FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {row["id"]: row for row in rows}

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
      3. Similarity threshold — confidence gate

    Returns a list of dicts with keys:
      text, book, pitaka, source, religion, language, score
    """

    # 1. Embed and L2-normalise the query
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    # 2. FAISS search — fetch more than top_k to allow for filtering
    fetch_k = min(top_k * 10, index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)
    scores  = scores[0]
    indices = indices[0]

    # 3. Pre-filter: keep only IDs belonging to the requested religion
    allowed_ids = set(_religion_ids.get(religion, []))

    candidate_ids = [
        int(idx) for score, idx in zip(scores, indices)
        if idx != -1
        and int(idx) in allowed_ids
        and float(score) >= threshold
    ]

    if not candidate_ids:
        return []

    # 4. Fetch chunk rows from SQLite in one query
    chunk_map = _fetch_chunks(candidate_ids)

    # 5. Build results preserving FAISS score order
    score_map = {
        int(idx): float(score)
        for score, idx in zip(scores, indices)
        if idx != -1
    }

    results = []
    for idx in candidate_ids:
        chunk = chunk_map.get(idx)
        if chunk is None:
            continue
        if chunk["language"] != language:
            continue
        results.append({
            "text":     chunk["text"],
            "book":     chunk["book"],
            "pitaka":   chunk["pitaka"],
            "source":   chunk["source"],
            "religion": chunk["religion"],
            "language": chunk["language"],
            "score":    score_map[idx],
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