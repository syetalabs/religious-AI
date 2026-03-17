import sqlite3
import threading
import numpy as np
import faiss
from pathlib import Path
import gc  # Added for strict memory management

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/tmp/religious-ai-data")

_RELIGION_PATHS = {
    "Buddhism": {
        "dir":   DATA_ROOT / "buddhism",
        "faiss": DATA_ROOT / "buddhism" / "faiss_index-en-si.bin",
        "db":    DATA_ROOT / "buddhism" / "chunks-en-si.db",
    },
    "Christianity": {
        "dir":   DATA_ROOT / "christianity",
        "faiss": DATA_ROOT / "christianity" / "faiss_index-en-si-ta.bin",
        "db":    DATA_ROOT / "christianity" / "chunks-en-si-ta.db",
    },
    "Hinduism": {
        "dir":   DATA_ROOT / "hinduism",
        "faiss": DATA_ROOT / "hinduism" / "faiss_index.bin",
        "db":    DATA_ROOT / "hinduism" / "chunks.db",
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

# CHANGED: RLock allows a thread to acquire the lock multiple times without freezing
_load_lock = threading.RLock()
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
    col_names = {row[1] for row in con.execute("PRAGMA table_info(chunks)").fetchall()}
    if "religion" in col_names:
        for row in con.execute("SELECT id, religion FROM chunks"):
            rel_ids.setdefault(row["religion"], []).append(row["id"])
    else:
        for row in con.execute("SELECT id FROM chunks"):
            rel_ids.setdefault(religion, []).append(row["id"])
    print(f"  {religion}: religions indexed = {list(rel_ids.keys())}")

    _indexes[religion]      = idx
    _cons[religion]         = con
    _religion_ids[religion] = rel_ids


# ────────────────────────────────────────────────────────────────
# Unload a religion to free RAM (Render 512MB limits)
# ────────────────────────────────────────────────────────────────
def _unload_religion(religion: str) -> None:
    """
    Release FAISS index and SQLite connection for a religion to free RAM.
    Buddhism is never unloaded — it is the base/default religion.
    """
    if religion == "Buddhism":
        return

    with _load_lock:
        if religion not in _loaded_religions:
            return

        con = _cons.pop(religion, None)
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

        # CHANGED: Explicitly delete the FAISS object and force garbage collection
        idx = _indexes.pop(religion, None)
        del idx  
        
        _religion_ids.pop(religion, None)
        _loaded_religions.discard(religion)

        gc.collect()

        print(f"  [retrieve] Unloaded {religion} from memory")


# ────────────────────────────────────────────────────────────────
# Lazy loader — load one or all religions
# ────────────────────────────────────────────────────────────────
def _lazy_load(religion: str = None) -> None:
    """
    Load the embedding model and the specified religion's index/DB.
    Enforces memory limits automatically by evicting inactive indices.
    """
    religions_to_load = (
        list(_RELIGION_PATHS.keys()) if religion is None
        else [religion]
    )

    if _ort_session is not None and all(r in _loaded_religions for r in religions_to_load):
        return

    with _load_lock:
        _load_embedding_model()

        # CHANGED: Eviction logic is now securely embedded in the loader
        for r in religions_to_load:
            if r != "Buddhism" and r not in _loaded_religions:
                # Unload any other loaded religion to make room
                for loaded in list(_loaded_religions):
                    if loaded != "Buddhism" and loaded != r:
                        print(f"  [retrieve] Evicting {loaded} to make room for {r}")
                        _unload_religion(loaded)

        for r in religions_to_load:
            if r not in _loaded_religions:
                _load_religion(r)
                _loaded_religions.add(r)


# ────────────────────────────────────────────────────────────────
# Convenience accessors (used by main.py health check)
# ────────────────────────────────────────────────────────────────
@property
def index():
    return _indexes.get("Buddhism")


# ────────────────────────────────────────────────────────────────
# Fetch chunks from a religion's DB
# ────────────────────────────────────────────────────────────────
def _fetch_chunks(religion: str, ids: list) -> dict:
    if not ids:
        return {}
    con          = _cons[religion]
    placeholders = ",".join("?" * len(ids))

    col_names = {row[1] for row in con.execute("PRAGMA table_info(chunks)").fetchall()}

    if "pitaka" in col_names:
        pitaka_expr = "pitaka"
    elif "testament" in col_names:
        pitaka_expr = "COALESCE(testament, '') AS pitaka"
    elif "section" in col_names:
        pitaka_expr = "COALESCE(section, '') AS pitaka"
    else:
        pitaka_expr = "'' AS pitaka"

    if "book" in col_names:
        book_expr = "book"
    elif "title" in col_names:
        book_expr = "title AS book"
    elif "scripture" in col_names:
        book_expr = "scripture AS book"
    else:
        book_expr = "'' AS book"

    source_expr = "source" if "source" in col_names else "'' AS source"
    religion_expr = "religion" if "religion" in col_names else f"'{religion}' AS religion"
    lang_expr = "language" if "language" in col_names else "'en' AS language"

    rows = con.execute(
        f"SELECT id, text, {book_expr}, {pitaka_expr}, {source_expr}, "
        f"{religion_expr}, {lang_expr} "
        f"FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {row["id"]: dict(row) for row in rows}


# ────────────────────────────────────────────────────────────────
# Language-match helper
# ────────────────────────────────────────────────────────────────
_LANG_ALIASES = {
    "en":      ["en", "english"],
    "si":      ["si", "sinhala", "sin"],
    "ta":      ["ta", "tamil", "tam"],
    "english": ["en", "english"],
    "sinhala": ["si", "sinhala", "sin"],
    "tamil":   ["ta", "tamil", "tam"],
}

TOP_K     = 5
THRESHOLD = 0.25


def _lang_matches(chunk_lang: str, requested_lang: str) -> bool:
    aliases = _LANG_ALIASES.get(requested_lang.lower(), [requested_lang.lower()])
    return chunk_lang.lower() in aliases


# ────────────────────────────────────────────────────────────────
# Main search
# ────────────────────────────────────────────────────────────────
def search(
    query:     str,
    religion:  str   = "Buddhism",
    language:  str   = "en",
    top_k:     int   = TOP_K,
    threshold: float = THRESHOLD,
) -> list[dict]:
    _lazy_load(religion)

    idx = _indexes.get(religion)
    if idx is None:
        return []

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
    threshold: float = 0.20,
) -> list[dict]:
    if religion != "Buddhism":
        return []

    _lazy_load(religion)

    idx = _indexes.get(religion)
    if idx is None:
        return []

    query_vec = _encode(en_query)
    faiss.normalize_L2(query_vec)

    fetch_k = min(top_k * 60, idx.ntotal)
    scores, indices = idx.search(query_vec, fetch_k)
    scores  = scores[0]
    indices = indices[0]

    allowed_ids = set(_religion_ids[religion].get(religion, []))

    candidate_ids = [
        int(ix) for score, ix in zip(scores, indices)
        if ix != -1 and int(ix) in allowed_ids and float(score) >= threshold
    ]
    if not candidate_ids:
        print(f"  [si-search] No candidates above threshold={threshold:.2f} "
              f"(fetch_k={fetch_k}, index_size={idx.ntotal})")
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

    print(f"  [si-search] Sinhala chunks returned: {len(results)} "
          f"(candidates scanned: {len(candidate_ids)}, fetch_k={fetch_k})")
    return results


# ────────────────────────────────────────────────────────────────
# Christianity SI/TA native-language search
# ────────────────────────────────────────────────────────────────
MIN_CHUNKS_FOR_NATIVE = 2

def search_christianity_native_lang(
    en_query:  str,
    language:  str,
    top_k:     int   = TOP_K,
    threshold: float = 0.20,
) -> list[dict]:
    _lazy_load("Christianity")

    idx = _indexes.get("Christianity")
    con = _cons.get("Christianity")
    if idx is None or con is None:
        return []

    query_vec = _encode(en_query)
    faiss.normalize_L2(query_vec)

    fetch_k = min(top_k * 60, idx.ntotal)
    scores, indices = idx.search(query_vec, fetch_k)
    scores  = scores[0]
    indices = indices[0]

    allowed_ids = set(_religion_ids["Christianity"].get("Christianity", []))

    candidate_ids = [
        int(ix) for score, ix in zip(scores, indices)
        if ix != -1 and int(ix) in allowed_ids and float(score) >= threshold
    ]
    if not candidate_ids:
        print(f"  [native-search] No candidates above threshold={threshold:.2f}")
        return []

    lang_aliases  = _LANG_ALIASES.get(language, [language])
    placeholders  = ",".join("?" * len(candidate_ids))
    lang_ph       = ",".join("?" * len(lang_aliases))
    score_map     = {int(ix): float(s) for s, ix in zip(scores, indices) if ix != -1}

    try:
        rows = con.execute(
            f"""
            SELECT id, text, book, source,
                   COALESCE(testament, '') AS pitaka,
                   religion, language
            FROM   chunks
            WHERE  id IN ({placeholders})
              AND  LOWER(language) IN ({lang_ph})
            """,
            candidate_ids + [a.lower() for a in lang_aliases],
        ).fetchall()
    except Exception as exc:
        print(f"  [native-search] DB query error: {exc}")
        return []

    results  = []
    seen     = set()
    for row in rows:
        text_hash = hash(row["text"][:200])
        if text_hash in seen:
            continue
        seen.add(text_hash)
        results.append({
            "text":     row["text"],
            "book":     row["book"],
            "pitaka":   row["pitaka"],
            "source":   row["source"],
            "religion": row["religion"],
            "language": row["language"],
            "score":    score_map.get(row["id"], 0.0),
        })
        if len(results) >= top_k:
            break

    results.sort(key=lambda r: -r["score"])
    print(
        f"  [native-search] language={language!r} chunks returned: {len(results)} "
        f"(candidates scanned: {len(candidate_ids)}, fetch_k={fetch_k})"
    )
    return results