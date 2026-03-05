import sqlite3
import threading
import numpy as np
import faiss
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global state
index         = None
_tokenizer    = None
_ort_session  = None
_con          = None
_religion_ids = {}
_load_lock    = threading.Lock()

# ────────────────────────────────────────────────────────────────
# Paths  — updated to bilingual -en-si files
# ────────────────────────────────────────────────────────────────
DATA_DIR       = Path("/tmp/religious-ai-data/buddhism")
CHUNKS_DB_PATH = DATA_DIR / "chunks-en-si.db"
FAISS_PATH     = DATA_DIR / "faiss_index-en-si.bin"
MODEL_DIR      = DATA_DIR / "onnx-model"

# ────────────────────────────────────────────────────────────────
# ONNX model download helper
# ────────────────────────────────────────────────────────────────
def _ensure_onnx_model() -> Path:
    from huggingface_hub import hf_hub_download
    import shutil
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "onnx/model.onnx",
    ]
    for filename in files:
        dest_name = filename.replace("/", "_")   # "onnx/model.onnx" → "onnx_model.onnx"
        dest = MODEL_DIR / dest_name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"  [model] Downloading {filename}...")
        tmp = hf_hub_download(
            repo_id=MODEL_NAME,
            filename=filename,
            cache_dir=str(MODEL_DIR / ".hf_cache"),
        )
        shutil.copy2(tmp, dest)
    return MODEL_DIR


# ────────────────────────────────────────────────────────────────
# Mean pooling
# ────────────────────────────────────────────────────────────────
def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


# ────────────────────────────────────────────────────────────────
# Encode query → unit-normalised numpy vector
# Always encode English text — Sinhala queries must be translated
# before calling _encode() because all-MiniLM-L6-v2 is English-only.
# ────────────────────────────────────────────────────────────────
def _encode(text: str) -> np.ndarray:
    inputs = _tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=256,
    )
    ort_inputs = {
        "input_ids":      inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in [i.name for i in _ort_session.get_inputs()]:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

    outputs = _ort_session.run(None, ort_inputs)
    vec = _mean_pool(outputs[0], inputs["attention_mask"])
    norm = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-9)
    return (vec / norm).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Lazy loader
# ────────────────────────────────────────────────────────────────
def _lazy_load():
    global index, _tokenizer, _ort_session, _con, _religion_ids
    if index is not None:
        return

    with _load_lock:
        if index is not None:
            return

        print("Loading FAISS index...")
        index = faiss.read_index(str(FAISS_PATH))
        print(f"{index.ntotal:,} vectors loaded")

        print("Downloading / loading ONNX embedding model...")
        model_dir = _ensure_onnx_model()

        from transformers import AutoTokenizer
        import onnxruntime as ort

        _tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            tokenizer_file=str(model_dir / "tokenizer.json"),
        )

        onnx_path = model_dir / "onnx_model.onnx"
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        _ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        print("Embedding model ready (ONNX / no PyTorch)")

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
SIMILARITY_THRESHOLD = 0.30   # lowered from 0.45 — Sinhala chunks may score lower

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


# ────────────────────────────────────────────────────────────────
# Search
#
# HOW SINHALA RETRIEVAL WORKS:
#   The FAISS index was built from English text only — Sinhala chunks
#   score near-zero against any query and never appear in top results.
#
#   For language="si":
#     1. Use FAISS to find the best *English* chunks (scores 0.7-0.8)
#     2. Collect the unique `source` values of those English chunks
#        (e.g. ["dn", "mn", "sn"])
#     3. Fetch Sinhala chunks from the same sources directly from the DB
#     4. Return those Sinhala chunks with the English score attached
#
#   For language="en": standard FAISS → filter pipeline unchanged.
# ────────────────────────────────────────────────────────────────
def search_sinhala_direct(
    en_query: str,
    religion: str = "Buddhism",
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Check whether Sinhala scripture chunks exist in the DB that are relevant
    to the question.

    Strategy:
      1. Embed the English translation of the query.
      2. Run FAISS to find top candidate chunk IDs (all languages).
      3. Among those candidates, look for chunks with language='si'.
      4. Return only Sinhala chunks that pass the score threshold.

    Returns an empty list when no relevant Sinhala chunks are found,
    which tells the caller to fall back to English retrieval + translation.
    """
    _lazy_load()

    query_vec = _encode(en_query)
    faiss.normalize_L2(query_vec)

    fetch_k = min(top_k * 20, index.ntotal)
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
        if chunk is None or not _lang_matches(chunk["language"], "si"):
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


def _fetch_sinhala_by_sources(sources: list[str], religion: str, top_k: int) -> list[dict]:
    """Fetch Sinhala chunks from the DB for the given source codes."""
    if not sources:
        return []
    placeholders = ",".join("?" * len(sources))
    rows = _con.execute(
        f"""SELECT id, text, book, pitaka, source, religion, language
            FROM chunks
            WHERE language='si' AND religion=? AND source IN ({placeholders})
            LIMIT ?""",
        [religion] + sources + [top_k * 4],
    ).fetchall()
    return [dict(r) for r in rows]


def search(
    query: str,
    religion: str = "Buddhism",
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    language: str = "en",
) -> list[dict]:
    """
    Search for English scripture chunks relevant to the query.
    For Sinhala retrieval use search_sinhala_direct() instead.
    """
    _lazy_load()

    query_vec = _encode(query)
    faiss.normalize_L2(query_vec)

    fetch_k = min(top_k * 20, index.ntotal)
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