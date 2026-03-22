import os
import threading
import httpx
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_fetcher import ensure_data_files

_ready      = False
_load_error = None

# Per-religion on-demand load state: "idle" | "loading" | "ready" | "error"
_religion_status: dict = {}
_religion_error:  dict = {}
_religion_lock = threading.Lock()


def _background_load():
    global _ready, _load_error
    try:
        print("=== Background: downloading Buddhism + Christianity data from HuggingFace ===")
        ensure_data_files(["Buddhism", "Christianity"])

        print("=== Background: loading Buddhism index + embedding model ===")
        from retrieve import _lazy_load
        _lazy_load("Buddhism")

        with _religion_lock:
            _religion_status["Buddhism"] = "ready"

        print("=== Background load complete. API is ready. ===")
        _ready = True
    except Exception as e:
        import traceback
        _load_error = str(e)
        print(f"=== Background load FAILED: {e} ===")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_background_load, daemon=True)
    t.start()
    print("=== Server started. Data loading in background... ===")
    yield
    print("=== Shutting down. ===")


# ════════════════════════════════════════════════════════════════
# App
# ════════════════════════════════════════════════════════════════
app = FastAPI(title="Multi-Religious Chatbot API", lifespan=lifespan)

_frontend_url    = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
_allowed_origins = ["http://localhost:5173", "http://localhost:3000"]
if _frontend_url:
    _allowed_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ════════════════════════════════════════════════════════════════
# Request models
# ════════════════════════════════════════════════════════════════
class QuestionRequest(BaseModel):
    question: str
    religion: str = "Buddhism"   # "Buddhism" | "Christianity" | "Hinduism"
    language: str = "en"         # "en"|"si"|"ta"  (Hinduism: "en" only)


class FeedbackRequest(BaseModel):
    question: str = ""
    answer:   str = ""
    rating:   str = ""
    comment:  str = ""
    religion: str = ""
    language: str = "English"


# ════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"message": "Multi-Religious Chatbot API is running"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    from retrieve import _indexes, _religion_ids

    files_info = {}
    for religion, paths in {
        "buddhism": {
            "faiss": Path("/tmp/religious-ai-data/buddhism/faiss_index-en-si.bin"),
            "db":    Path("/tmp/religious-ai-data/buddhism/chunks-en-si.db"),
        },
        "christianity": {
            "faiss": Path("/tmp/religious-ai-data/christianity/faiss_index-en-si-ta.bin"),
            "db":    Path("/tmp/religious-ai-data/christianity/chunks-en-si-ta.db"),
        },
        "hinduism": {
            "faiss": Path("/tmp/religious-ai-data/hinduism/faiss_index.bin"),
            "db":    Path("/tmp/religious-ai-data/hinduism/chunks.db"),
        },
    }.items():
        files_info[religion] = {
            "faiss_index": {
                "exists":  paths["faiss"].exists(),
                "size_mb": round(paths["faiss"].stat().st_size / 1_048_576, 1) if paths["faiss"].exists() else 0,
            },
            "chunks_db": {
                "exists":  paths["db"].exists(),
                "size_mb": round(paths["db"].stat().st_size / 1_048_576, 1) if paths["db"].exists() else 0,
            },
        }

    return {
        "status":         "ready" if _ready else ("error" if _load_error else "loading"),
        "load_error":     _load_error,
        "files":          files_info,
        "indexes_loaded": list(_indexes.keys()),
        "religions":      {r: list(ids.keys()) for r, ids in _religion_ids.items()},
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if _load_error:
        raise HTTPException(status_code=503, detail=f"Server failed to load data: {_load_error}")
    if not _ready:
        raise HTTPException(status_code=503, detail="Server is still loading data. Please retry in a moment.")

    supported = ["Buddhism", "Christianity", "Hinduism"]
    if request.religion not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported religion: {request.religion}. Supported: {supported}")

    language = request.language

    from retrieve import _lazy_load
    from rag_answer import answer_question

    # _lazy_load evicts any currently-loaded religion before loading the new one
    _lazy_load(request.religion)

    result = answer_question(
        question=request.question,
        religion=request.religion,
        language=language,
    )

    return {
        "answer":             result["answer"],
        "sources":            result.get("sources", []),
        "scores":             result.get("scores", []),
        "confidence_warning": result.get("low_confidence", False),
        "flagged":            result.get("flagged", False),
        "warnings":           result.get("warnings", []),
    }


# ════════════════════════════════════════════════════════════════
# On-demand religion preparation
# ════════════════════════════════════════════════════════════════
def _prepare_religion_bg(religion: str):
    """
    Download data files + load FAISS index for a religion in a background thread.
    Eviction of any previously loaded religion is handled inside _lazy_load automatically.
    """
    with _religion_lock:
        if _religion_status.get(religion) in ("loading", "ready"):
            return
        _religion_status[religion] = "loading"

    try:
        print(f"=== /prepare: downloading {religion} data ===")
        ensure_data_files([religion])

        print(f"=== /prepare: loading {religion} index ===")
        from retrieve import _lazy_load
        _lazy_load(religion)

        with _religion_lock:
            _religion_status[religion] = "ready"
        print(f"=== /prepare: {religion} ready ===")

    except Exception as e:
        with _religion_lock:
            _religion_status[religion] = "error"
            _religion_error[religion]  = str(e)
        print(f"=== /prepare: {religion} FAILED: {e} ===")


@app.post("/prepare/{religion}")
def prepare_religion(religion: str):
    """Trigger background download + index load for a religion."""
    supported = ["Buddhism", "Christianity", "Hinduism"]
    if religion not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported religion: {religion}")

    if _religion_status.get(religion) == "ready":
        return {"status": "ready"}

    if _religion_status.get(religion) != "loading":
        t = threading.Thread(target=_prepare_religion_bg, args=(religion,), daemon=True)
        t.start()

    return {"status": "loading"}


@app.get("/status/{religion}")
def religion_status(religion: str):
    """Poll the load status of a specific religion."""
    supported = ["Buddhism", "Christianity", "Hinduism"]
    if religion not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported religion: {religion}")

    status = _religion_status.get(religion, "idle")

    if status == "error":
        return {"status": "error", "error": _religion_error.get(religion, "Unknown error")}
    if status == "ready":
        return {"status": "ready"}
    if status == "idle":
        t = threading.Thread(target=_prepare_religion_bg, args=(religion,), daemon=True)
        t.start()
    return {"status": "loading"}


# ════════════════════════════════════════════════════════════════
# Memory debug — hit /memory before/after switching religions
# to verify unloading + RSS drop
# ════════════════════════════════════════════════════════════════
@app.get("/memory")
def memory_stats():
    import gc
    import psutil
    from retrieve import _indexes, _loaded_religions, _cons

    proc = psutil.Process(os.getpid())
    mem  = proc.memory_info()

    return {
        "rss_mb":           round(mem.rss / 1_048_576, 1),   # actual RAM (what Render sees)
        "vms_mb":           round(mem.vms / 1_048_576, 1),   # virtual memory
        "loaded_religions": list(_loaded_religions),
        "indexes_loaded":   {r: idx.ntotal for r, idx in _indexes.items()},
        "db_connections":   list(_cons.keys()),
        "gc_counts":        gc.get_count(),                   # (gen0, gen1, gen2)
    }


# ════════════════════════════════════════════════════════════════
# Feedback
# ════════════════════════════════════════════════════════════════
@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    notion_token = os.environ.get("NOTION_TOKEN", "").strip()
    notion_db_id = os.environ.get("NOTION_DB_ID", "").strip()

    if not notion_token or not notion_db_id:
        raise HTTPException(status_code=503, detail="Feedback not configured")

    # Notion rejects select with an empty name — only include Rating if set
    properties = {
        "Question": {"title":     [{"text": {"content": payload.question[:2000]}}]},
        "Answer":   {"rich_text": [{"text": {"content": payload.answer[:2000]}}]},
        "Comment":  {"rich_text": [{"text": {"content": payload.comment[:2000]}}]},
        "Religion": {"rich_text": [{"text": {"content": payload.religion}}]},
        "Language": {"rich_text": [{"text": {"content": payload.language}}]},
    }
    if payload.rating:
        properties["Rating"] = {"select": {"name": payload.rating}}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.notion.com/v1/pages",
                headers={
                    "Authorization":  f"Bearer {notion_token}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type":   "application/json",
                },
                json={
                    "parent":     {"database_id": notion_db_id},
                    "properties": properties,
                },
                timeout=8.0,   # stay under Render's gateway timeout
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Notion request timed out")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach Notion: {exc}")

    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Notion error {resp.status_code}: {resp.text[:500]}")

    return {"ok": True}