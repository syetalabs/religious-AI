import sqlite3
import threading
import numpy as np
import faiss
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/tmp/religious-ai-data")

_RELIGION_PATHS = {
    "Buddhism": {
        "dir":        DATA_ROOT / "buddhism",
        "faiss":      DATA_ROOT / "buddhism" / "faiss_index-en-si.bin",
        "db":         DATA_ROOT / "buddhism" / "chunks-en-si.db",
    },
    "Christianity": {
        "dir":        DATA_ROOT / "christianity",
        "faiss":      DATA_ROOT / "christianity" / "faiss_index.bin",
        "db":         DATA_ROOT / "christianity" / "chunks.db",
    },
}

MODEL_DIR = DATA_ROOT / "buddhism" / "onnx-model"   # shared model cache

# ────────────────────────────────────────────────────────────────
# Per-religion state
# ────────────────────────────────────────────────────────────────
_indexes      = {}   # religion -> faiss.Index
_cons         = {}   # religion -> sqlite3.Connection
_religion_ids = {}   # religion -> {religion_name: [chunk_id, ...]}

# Shared embedding model
_tokenizer   = None
_ort_session = None

_load_lock = threading.Lock()
_loaded_religions: set = set()


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
        dest_name = filename.replace("/", "_")
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
    mask   = attention_mask[..., np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


# ────────────────────────────────────────────────────────────────
# Encode query → unit-normalised numpy vector
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
    vec  = _mean_pool(outputs[0], inputs["attention_mask"])
    norm = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-9)
    return (vec / norm).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Load embedding model (once, shared across religions)
# ────────────────────────────────────────────────────────────────
def _load_embedding_model():
    global _tokenizer, _ort_session
    if _ort_session is not None:
        return

    print("Downloading / loading ONNX embedding model...")
    model_dir = _ensure_onnx_model()

    from transformers import AutoTokenizer
    import onnxruntime as ort

    _tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        tokenizer_file=str(model_dir / "tokenizer.json"),
    )

    onnx_path  = model_dir / "onnx_model.onnx"
    sess_opts  = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    _ort_session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    print("Embedding model ready (ONNX / no PyTorch)")


# ────────────────────────────────────────────────────────────────
# Load a single religion's index + DB
# ────────────────────────────────────────────────────────────────
def _load_religion(religion: str) -> None:
    paths = _RELIGION_PATHS.get(religion)
    if paths is None:
        raise ValueError(f"Unknown religion: {religion}")

    faiss_path = paths["faiss"]
    db_path    = paths["db"]

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Chunks DB not found: {db_path}")

    print(f"Loading FAISS index for {religion}...")
    idx = faiss.read_index(str(faiss_path))
    print(f"  {religion}: {idx.ntotal:,} vectors loaded")

    print(f"Connecting to SQLite for {religion}...")
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row

    rel_ids: dict[str, list] = {}
    for row in con.execute("SELECT id, religion FROM chunks"):
        rel_ids.setdefault(row["religion"], []).append(row["id"])
    print(f"  {religion}: religions indexed = {list(rel_ids.keys())}")

    _indexes[religion]      = idx
    _cons[religion]         = con
    _religion_ids[religion] = rel_ids


# ────────────────────────────────────────────────────────────────
# Lazy loader — load one or all religions
# ────────────────────────────────────────────────────────────────
def _lazy_load(religion: str = None) -> None:
    """
    Load the embedding model and the specified religion's index/DB.
    If religion is None, load ALL configured religions.
    """
    religions_to_load = (
        list(_RELIGION_PATHS.keys()) if religion is None
        else [religion]
    )

    # Fast path — everything already loaded
    if _ort_session is not None and all(r in _loaded_religions for r in religions_to_load):
        return

    with _load_lock:
        # Re-check inside lock
        _load_embedding_model()

        for r in religions_to_load:
            if r not in _loaded_religions:
                _load_religion(r)
                _loaded_religions.add(r)


# ────────────────────────────────────────────────────────────────
# Convenience accessors (used by main.py health check)
# ────────────────────────────────────────────────────────────────
@property
def index():
    """Return the Buddhism index for backwards compatibility."""
    return _indexes.get("Buddhism")


# ────────────────────────────────────────────────────────────────
# Fetch chunks from a religion's DB
# ────────────────────────────────────────────────────────────────
def _fetch_chunks(religion: str, ids: list) -> dict:
    if not ids:
        return {}
    con          = _cons[religion]
    placeholders = ",".join("?" * len(ids))

    # Detect available columns — Buddhism has 'pitaka', Christianity has 'testament'/'genre'
    col_names   = {row[1] for row in con.execute("PRAGMA table_info(chunks)").fetchall()}
    pitaka_expr = "pitaka" if "pitaka" in col_names else "COALESCE(testament, '') AS pitaka"

    rows = con.execute(
        f"SELECT id, text, book, {pitaka_expr}, source, religion, language "
        f"FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {row["id"]: row for row in rows}


# ────────────────────────────────────────────────────────────────
# Language normalisation
# ────────────────────────────────────────────────────────────────
TOP_K                = 10
SIMILARITY_THRESHOLD = 0.30

_LANG_ALIASES = {
    "en": ["en", "english"],
    "si": ["si", "sinhala"],
    "ta": ["ta", "tamil"],
}

def _lang_matches(chunk_lang: str, requested_lang: str) -> bool:
    c       = chunk_lang.lower().strip()
    r       = requested_lang.lower().strip()
    aliases = _LANG_ALIASES.get(r, [r])
    return c in aliases or r in _LANG_ALIASES.get(c, [c])


# ────────────────────────────────────────────────────────────────
# Core search (English path — used for Buddhism EN and all Christianity)
# ────────────────────────────────────────────────────────────────
def search(
    query:     str,
    religion:  str   = "Buddhism",
    top_k:     int   = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    language:  str   = "en",
) -> list[dict]:
    _lazy_load(religion)

    idx = _indexes.get(religion)
    if idx is None:
        return []

    # For Christianity the DB stores religion as "Christianity"
    rel_name = religion
    allowed_ids = set(_religion_ids[religion].get(rel_name, []))

    query_vec = _encode(query)
    faiss.normalize_L2(query_vec)

    fetch_k = min(top_k * 20, idx.ntotal)
    scores, indices = idx.search(query_vec, fetch_k)
    scores  = scores[0]
    indices = indices[0]

    candidate_ids = [
        int(ix) for score, ix in zip(scores, indices)
        if ix != -1 and int(ix) in allowed_ids and float(score) >= threshold
    ]
    if not candidate_ids:
        return []

    chunk_map = _fetch_chunks(religion, candidate_ids)
    score_map = {int(ix): float(score) for score, ix in zip(scores, indices) if ix != -1}

    results = []
    for ix in candidate_ids:
        chunk = chunk_map.get(ix)
        if chunk is None or not _lang_matches(chunk["language"], language):
            continue
        results.append({
            "text":     chunk["text"],
            "book":     chunk["book"],
            "pitaka":   chunk["pitaka"],
            "source":   chunk["source"],
            "religion": chunk["religion"],
            "language": chunk["language"],
            "score":    score_map[ix],
        })
        if len(results) >= top_k:
            break

    return results


# ────────────────────────────────────────────────────────────────
# Sinhala-specific search (Buddhism only)
# ────────────────────────────────────────────────────────────────
def search_sinhala_direct(
    en_query:  str,
    religion:  str   = "Buddhism",
    top_k:     int   = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Find Sinhala scripture chunks relevant to an English query.
    Uses FAISS with the English query, then filters for language='si'.
    Falls back to empty list (triggers English-context path in rag_answer.py).
    Only applicable to Buddhism — returns [] for other religions.
    """
    if religion != "Buddhism":
        return []

    _lazy_load(religion)

    idx = _indexes.get(religion)
    if idx is None:
        return []

    query_vec = _encode(en_query)
    faiss.normalize_L2(query_vec)

    fetch_k     = min(top_k * 20, idx.ntotal)
    scores, indices = idx.search(query_vec, fetch_k)
    scores  = scores[0]
    indices = indices[0]

    allowed_ids = set(_religion_ids[religion].get(religion, []))

    candidate_ids = [
        int(ix) for score, ix in zip(scores, indices)
        if ix != -1 and int(ix) in allowed_ids and float(score) >= threshold
    ]
    if not candidate_ids:
        return []

    chunk_map = _fetch_chunks(religion, candidate_ids)
    score_map = {int(ix): float(score) for score, ix in zip(scores, indices) if ix != -1}

    results = []
    for ix in candidate_ids:
        chunk = chunk_map.get(ix)
        if chunk is None or not _lang_matches(chunk["language"], "si"):
            continue
        results.append({
            "text":     chunk["text"],
            "book":     chunk["book"],
            "pitaka":   chunk["pitaka"],
            "source":   chunk["source"],
            "religion": chunk["religion"],
            "language": chunk["language"],
            "score":    score_map[ix],
        })
        if len(results) >= top_k:
            break

    return results