import os
import sqlite3
import threading
import numpy as np
import faiss
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global state
index        = None
model        = None
_con         = None
_religion_ids = {}
_load_lock   = threading.Lock()   # prevents race on concurrent first requests

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
DATA_DIR       = Path("/tmp/religious-ai-data")
CHUNKS_DB_PATH = DATA_DIR / "chunks.db"
FAISS_PATH     = DATA_DIR / "faiss_index.bin"

# ────────────────────────────────────────────────────────────────
# Lazy loader
# Uses fastembed instead of sentence-transformers to avoid pulling
# in PyTorch (~400 MB). fastembed uses ONNX runtime (~150 MB total).
# ────────────────────────────────────────────────────────────────
def _lazy_load():
    global index, model, _con, _religion_ids
    if index is not None:
        return  # already loaded — fast path, no lock needed

    with _load_lock:
        if index is not None:
            return  # another thread loaded while we waited

        print("Loading FAISS index...")
        index = faiss.read_index(str(FAISS_PATH))
        print(f"{index.ntotal:,} vectors loaded")

        print("Loading embedding model (fastembed / ONNX)...")
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=MODEL_NAME)
        print("Model ready")

        print("Connecting to SQLite...")
        _con = sqlite3.connect(str(CHUNKS_DB_PATH), check_same_thread=False)
        _con.row_factory = sqlite3.Row

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
# Language normalisation
# ────────────────────────────────
TOP_K = 10
SIMILARITY_THRESHOLD = 0.45

_LANG_ALIASES = {
    "en": ["en", "english"],
    "si": ["si", "sinhala"],
    "ta": ["ta", "tamil"],
}

def _lang_matches(chunk_lang: str, requested_lang: str) -> bool:
    c = chunk_lang.lower().strip()
    r = requested_lang.lower().strip()
    aliases = _LANG_ALIASES.get(r, [r])
    return c in aliases or r in _LANG_ALIASES.get(c, [c])

# ────────────────────────────────
# Search
# ────────────────────────────────
def search(query: str, religion="Buddhism", top_k=TOP_K, threshold=SIMILARITY_THRESHOLD, language="en"):
    _lazy_load()

    # fastembed returns a generator — convert to numpy array
    query_vec = np.array(list(model.embed([query])), dtype=np.float32)
    faiss.normalize_L2(query_vec)

    fetch_k = min(top_k * 10, index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)
    scores  = scores[0]
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
        if chunk is None or not _lang_matches(chunk["language"], language):
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