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
BUDDHISM_CHUNKS_PATH         = BUDDHISM_DATA_DIR / "chunks.json"
BUDDHISM_EMBEDDINGS_PATH     = BUDDHISM_DATA_DIR / "embeddings.npy"
BUDDHISM_FAISS_PATH          = BUDDHISM_DATA_DIR / "faiss_index.bin"

# ── Inject path early and unconditionally ──────────────────
# Done twice: once at module level, once as a guaranteed
# runtime guard inside each helper — uvicorn's reloader
# spawns child processes where module-level inserts may not
# propagate reliably.
def _ensure_buddhism_path():
    buddhism_str = str(BUDDHISM_DIR)
    if buddhism_str not in sys.path:
        sys.path.insert(0, buddhism_str)

_ensure_buddhism_path()   # module-level call

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
    _ensure_buddhism_path()   # guaranteed before import
    print("\n  📥  Running data_loader.download_tipitaka()…")
    try:
        from data_loader import download_tipitaka
        download_tipitaka()
        print(f"  ✅  Corpus saved → {BUDDHISM_CORPUS_PATH.relative_to(BUDDHISM_DIR)}")
    except Exception as exc:
        print(f"  ❌  data_loader failed: {exc}")
        sys.exit(1)


def _run_chunk_and_embed():
    _ensure_buddhism_path()   # guaranteed before runpy
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

    # ── 0. Environment ──────────────────────────────────────
    _check_env_vars()

    # ── 1. Directories ──────────────────────────────────────
    BUDDHISM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    BUDDHISM_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ✅  data/     : {BUDDHISM_DATA_DIR.relative_to(BUDDHISM_DIR)}")
    print(f"  ✅  sections/ : {BUDDHISM_SECTIONS_DIR.relative_to(BUDDHISM_DIR)}")

    # ── 2. Raw corpus ───────────────────────────────────────
    corpus_ok = BUDDHISM_CORPUS_PATH.exists() and BUDDHISM_CORPUS_PATH.stat().st_size > 0
    if corpus_ok:
        size_mb = BUDDHISM_CORPUS_PATH.stat().st_size / 1_048_576
        print(f"  ✅  tipitaka_raw.json  ({size_mb:.1f} MB)")
    else:
        print(f"  ❌  tipitaka_raw.json  — not found or empty")

    # ── 3. Embedding artefacts ──────────────────────────────
    chunks_ok     = BUDDHISM_CHUNKS_PATH.exists()     and BUDDHISM_CHUNKS_PATH.stat().st_size > 0
    embeddings_ok = BUDDHISM_EMBEDDINGS_PATH.exists() and BUDDHISM_EMBEDDINGS_PATH.stat().st_size > 0
    faiss_ok      = BUDDHISM_FAISS_PATH.exists()      and BUDDHISM_FAISS_PATH.stat().st_size > 0

    for label, ok in [
        ("chunks.json",     chunks_ok),
        ("embeddings.npy",  embeddings_ok),
        ("faiss_index.bin", faiss_ok),
    ]:
        print(f"  {'✅' if ok else '❌'}  {label}")

    # ── 4. Build missing steps ──────────────────────────────
    if not corpus_ok:
        print("\n  ⚠️   Raw corpus missing — running data_loader…")
        _run_data_loader()
        corpus_ok = BUDDHISM_CORPUS_PATH.exists() and BUDDHISM_CORPUS_PATH.stat().st_size > 0
        if not corpus_ok:
            print("  ❌  Corpus still missing after download. Aborting.")
            sys.exit(1)

    if not (chunks_ok and embeddings_ok and faiss_ok):
        print("\n  ⚠️   Embedding artefacts missing — running chunk_and_embed…")
        _run_chunk_and_embed()

    # ── 5. Final check ──────────────────────────────────────
    all_present = all([
        BUDDHISM_CORPUS_PATH.exists(),
        BUDDHISM_CHUNKS_PATH.exists(),
        BUDDHISM_EMBEDDINGS_PATH.exists(),
        BUDDHISM_FAISS_PATH.exists(),
    ])

    if not all_present:
        print("\n  ❌  One or more required files are still missing. Aborting.")
        sys.exit(1)

    # ── 6. Pre-load retrieve.py ─────────────────────────────
    _ensure_buddhism_path()   # guaranteed before import
    print("\n  🔄  Pre-loading embedding model and FAISS index…")
    try:
        import retrieve  # noqa: F401
        print("  ✅  Model and index loaded into memory.")
    except Exception as exc:
        print(f"  ❌  Failed to pre-load retrieve.py: {exc}")
        sys.exit(1)

    print(f"\n  All checks passed. Starting chatbot... 🪷")
    print("═" * 55 + "\n")


# ═══════════════════════════════════════════════════════════
# FastAPI lifespan
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    check_and_build()
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
    language: str = "English"   # "English" | "Sinhala" | "Tamil"

# ═══════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Multi-Religious Chatbot API is running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    _ensure_buddhism_path()   # guaranteed before imports
    from rag_answer import answer_question
    from translator import translate_to_english, translate_from_english

    # Step 1 — translate question to English for the RAG pipeline
    english_question = translate_to_english(request.question, request.language)

    # Step 2 — run through RAG (always in English)
    result = answer_question(
        question=english_question,
        religion=request.religion,
        language="en",
    )

    # Step 3 — translate the answer back to the user's language
    translated_answer = translate_from_english(result["answer"], request.language)

    return {
        "answer":             translated_answer,
        "sources":            result.get("sources", []),
        "scores":             result.get("scores", []),
        "confidence_warning": result.get("low_confidence", False),
        "flagged":            result.get("flagged", False),
        "warnings":           result.get("warnings", []),
    }