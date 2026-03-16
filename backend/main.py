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

_ready       = False
_load_error  = None


def _background_load():
    global _ready, _load_error
    try:
        print("=== Background: downloading data files from HuggingFace ===")
        ensure_data_files()   # downloads both Buddhism and Christianity

        print("=== Background: loading FAISS indexes + embedding model ===")
        from retrieve import _lazy_load
        _lazy_load()          # loads all religions

        print("=== Background load complete. API is ready. ===")
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
    religion: str = "Buddhism"   # "Buddhism" | "Christianity"
    language: str = "en"         # "en" only for Christianity; "en"|"si"|"ta" for Buddhism


class FeedbackRequest(BaseModel):
    question: str = ""
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

    # Validate religion
    supported = ["Buddhism", "Christianity"]
    if request.religion not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported religion: {request.religion}. Supported: {supported}")

    # All supported languages are passed through; rag_answer.py handles
    # the translation path for Christianity Sinhala / Tamil.
    language = request.language

    from retrieve import _lazy_load
    from rag_answer import answer_question
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


@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    notion_token = os.environ.get("NOTION_TOKEN", "")
    notion_db_id = os.environ.get("NOTION_DB_ID", "")

    if not notion_token or not notion_db_id:
        raise HTTPException(status_code=503, detail="Feedback not configured")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization":  f"Bearer {notion_token}",
                "Notion-Version": "2022-06-28",
                "Content-Type":   "application/json",
            },
            json={
                "parent": {"database_id": notion_db_id},
                "properties": {
                    "Question": {"title":     [{"text": {"content": payload.question[:2000]}}]},
                    "Rating":   {"select":    {"name": payload.rating}},
                    "Comment":  {"rich_text": [{"text": {"content": payload.comment[:2000]}}]},
                    "Religion": {"rich_text": [{"text": {"content": payload.religion}}]},
                    "Language": {"rich_text": [{"text": {"content": payload.language}}]},
                },
            },
            timeout=10,
        )

    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Notion error: {resp.text}")

    return {"ok": True}