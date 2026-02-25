import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════

BASE_DIR                     = Path(__file__).parent
BUDDHISM_DIR                 = BASE_DIR.parent / "multi-religion" / "buddhism"
BUDDHISM_DATA_DIR            = BUDDHISM_DIR / "data"
BUDDHISM_SECTIONS_DIR        = BUDDHISM_DATA_DIR / "sections"
BUDDHISM_CHECKPOINT_PATH     = BUDDHISM_DATA_DIR / "checkpoint.json"

BUDDHISM_CORPUS_PATH         = BUDDHISM_DATA_DIR / "tipitaka_raw.json"
BUDDHISM_CHUNKS_DB_PATH      = BUDDHISM_DATA_DIR / "chunks.db"       # ← SQLite (runtime)
BUDDHISM_CHUNKS_JSON_PATH    = BUDDHISM_DATA_DIR / "chunks.json"     # ← JSON (build-time only)
BUDDHISM_EMBEDDINGS_PATH     = BUDDHISM_DATA_DIR / "embeddings.npy"
BUDDHISM_FAISS_PATH          = BUDDHISM_DATA_DIR / "faiss_index.bin"

def _ensure_buddhism_path():
    buddhism_str = str(BUDDHISM_DIR)
    if buddhism_str not in sys.path:
        sys.path.insert(0, buddhism_str)

_ensure_buddhism_path()

# ═══════════════════════════════════════════════════════════
# Startup helpers
# ═══════════════════════════════════════════════════════════

def _banner(title: str):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")


def _check_env_vars():
    from dotenv import load_dotenv
    load_dotenv()

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        print("  ❌  GROQ_API_KEY is not set.")
        print("      Create a .env file in your backend folder:")
        print("        GROQ_API_KEY=your_key_here")
        print("      Get a free key at https://console.groq.com\n")
        sys.exit(1)
    else:
        print(f"  ✅  GROQ_API_KEY is set.")


def _run_data_loader():
    _ensure_buddhism_path()
    print("\n  📥  Running data_loader.download_tipitaka()…")
    try:
        from data_loader import download_tipitaka
        download_tipitaka()
        print(f"  ✅  Corpus saved → {BUDDHISM_CORPUS_PATH.relative_to(BUDDHISM_DIR)}")
    except Exception as exc:
        print(f"  ❌  data_loader failed: {exc}")
        sys.exit(1)


def _run_chunk_and_embed():
    _ensure_buddhism_path()
    print("\n  🔢  Running chunk_and_embed pipeline…")
    try:
        import runpy
        runpy.run_path(str(BUDDHISM_DIR / "chunk_and_embed.py"), run_name="__main__")
        print("  ✅  chunks.json, embeddings.npy, faiss_index.bin created.")
    except Exception as exc:
        print(f"  ❌  chunk_and_embed failed: {exc}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════
# Main startup sequence
# ═══════════════════════════════════════════════════════════

def check_and_build():
    _banner("Startup Check")
    _check_env_vars()

    print("  🔄  Ensuring runtime data files from Hugging Face...")

    try:
        import retrieve  # this triggers ensure_data_files()
        print("  ✅  Runtime files ready.")
    except Exception as exc:
        print(f"  ❌  Startup failed: {exc}")
        sys.exit(1)

    print("\n  All checks passed. Starting chatbot... 🪷")
    print("═" * 55 + "\n")
# ═══════════════════════════════════════════════════════════
# FastAPI lifespan
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App starting...")
    yield
    print("\nAPI shutting down gracefully.")


# ═══════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Multi-Religious Chatbot API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════

class QuestionRequest(BaseModel):
    question: str
    religion: str = "Buddhism"
    language: str = "English"


# ═══════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Multi-Religious Chatbot API is running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    _ensure_buddhism_path()
    from rag_answer import answer_question
    from translator import translate_to_english, translate_from_english

    english_question = translate_to_english(request.question, request.language)

    result = answer_question(
        question=english_question,
        religion=request.religion,
        language="en",
    )

    translated_answer = translate_from_english(result["answer"], request.language)

    return {
        "answer":             translated_answer,
        "sources":            result.get("sources", []),
        "scores":             result.get("scores", []),
        "confidence_warning": result.get("low_confidence", False),
        "flagged":            result.get("flagged", False),
        "warnings":           result.get("warnings", []),
    }