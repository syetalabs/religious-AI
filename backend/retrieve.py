import os
import sqlite3
import faiss
from pathlib import Path

# Optional: use smaller local embedding
MODEL_NAME = "paraphrase-MiniLM-L3-v2"

# Global placeholders
index = None
model = None
_con = None
_religion_ids = {}

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
DATA_DIR = Path("/tmp/religious-ai-data")
CHUNKS_DB_PATH = DATA_DIR / "chunks.db"
FAISS_PATH = DATA_DIR / "faiss_index.bin"

# ────────────────────────────────────────────────────────────────
# Lazy loader
# ────────────────────────────────────────────────────────────────
def _lazy_load():
    global index, model, _con, _religion_ids
    if index is None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        print("Loading FAISS index...")
        index = faiss.read_index(str(FAISS_PATH))
        print(f"{index.ntotal:,} vectors loaded")

        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)
        print("Model ready")

        print("Connecting to SQLite...")
        _con = sqlite3.connect(str(CHUNKS_DB_PATH), check_same_thread=False)
        _con.row_factory = sqlite3.Row

        # Build religion index
        for row in _con.execute("SELECT id, religion FROM chunks"):
            _religion_ids.setdefault(row["religion"], []).append(row["id"])
        print(f"Religions indexed: {list(_religion_ids.keys())}")

# ────────────────────────────────
# Fetch chunks
# ────────────────────────────────
def _fetch_chunks(ids):
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _con.execute(
        f"SELECT id, text, book, pitaka, source, religion, language FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {row["id"]: row for row in rows}

# ────────────────────────────────
# Search
# ────────────────────────────────
TOP_K = 10
SIMILARITY_THRESHOLD = 0.45

def search(query: str, religion="Buddhism", top_k=TOP_K, threshold=SIMILARITY_THRESHOLD, language="en"):
    _lazy_load()

    # Embed query
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    fetch_k = min(top_k * 10, index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)
    scores = scores[0]
    indices = indices[0]

    allowed_ids = set(_religion_ids.get(religion, []))
    candidate_ids = [
        int(idx) for score, idx in zip(scores, indices)
        if idx != -1 and int(idx) in allowed_ids and float(score) >= threshold
    ]
    if not candidate_ids:
        return []

    chunk_map = _fetch_chunks(candidate_ids)
    score_map = {int(idx): float(score) for score, idx in zip(scores, indices) if idx != -1}

    results = []
    for idx in candidate_ids:
        chunk = chunk_map.get(idx)
        if chunk is None or chunk["language"] != language:
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