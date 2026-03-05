import os
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_fetcher import ensure_data_files
from retrieve import _lazy_load

_ready = False
_load_error = None

def _background_load():
    global _ready, _load_error
    try:
        print("=== Background: downloading data files from HuggingFace ===")
        from data_fetcher import ensure_data_files
        ensure_data_files()

        print("=== Background: loading FAISS index + embedding model ===")
        from retrieve import _lazy_load
        _lazy_load()

        print("=== Background load complete. API is ready to answer questions. ===")
        _ready = True
    except Exception as e:
        _load_error = str(e)
        print(f"=== Background load FAILED: {e} ===")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_background_load, daemon=True)
    t.start()
    print("=== Server started. Data loading in background... ===")
    yield
    print("=== Shutting down. ===")


# ════════════════════════════════════════════════
# App
# ════════════════════════════════════════════════
app = FastAPI(title="Multi-Religious Chatbot API", lifespan=lifespan)

_frontend_url = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
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


# ════════════════════════════════════════════════
# Request model
# ════════════════════════════════════════════════
class QuestionRequest(BaseModel):
    question: str
    religion: str = "Buddhism"
    language: str = "en"   # "en" | "si" | "ta" — passed directly to answer_question


# ════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════
@app.get("/")
def root():
    return {"message": "Multi-Religious Chatbot API is running"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    from retrieve import index, _religion_ids

    data_dir   = Path("/tmp/religious-ai-data/buddhism")
    faiss_path = data_dir / "faiss_index-en-si.bin"
    db_path    = data_dir / "chunks-en-si.db"

    return {
        "status":        "ready" if _ready else ("error" if _load_error else "loading"),
        "load_error":    _load_error,
        "files": {
            "faiss_index": {
                "exists":  faiss_path.exists(),
                "size_mb": round(faiss_path.stat().st_size / 1_048_576, 1) if faiss_path.exists() else 0,
            },
            "chunks_db": {
                "exists":  db_path.exists(),
                "size_mb": round(db_path.stat().st_size / 1_048_576, 1) if db_path.exists() else 0,
            },
        },
        "index_loaded":  index is not None,
        "vectors":       index.ntotal if index is not None else 0,
        "religions":     list(_religion_ids.keys()) if _religion_ids else [],
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if _load_error:
        raise HTTPException(status_code=503, detail=f"Server failed to load data: {_load_error}")
    if not _ready:
        raise HTTPException(status_code=503, detail="Server is still loading data. Please retry in a moment.")

    from retrieve import _lazy_load
    from rag_answer import answer_question
    _lazy_load()

    result = answer_question(
        question=request.question,
        religion=request.religion,
        language=request.language,
    )

    return {
        "answer":             result["answer"],
        "sources":            result.get("sources", []),
        "scores":             result.get("scores", []),
        "confidence_warning": result.get("low_confidence", False),
        "flagged":            result.get("flagged", False),
        "warnings":           result.get("warnings", []),
    }