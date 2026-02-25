import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieve import search, _lazy_load

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# FastAPI lifespan
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy loading happens on first search, so startup is light
    print("API starting... ready for requests.")
    yield
    print("\nAPI shutting down gracefully.")

# ════════════════════════════════════════════════
# App
# ════════════════════════════════════════════════
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

# ════════════════════════════════════════════════
# Request model
# ════════════════════════════════════════════════
class QuestionRequest(BaseModel):
    question: str
    religion: str = "Buddhism"
    language: str = "English"

# ════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════
@app.get("/")
def root():
    return {"message": "Multi-Religious Chatbot API is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Lazy-load FAISS and embedding model on first request
    _lazy_load()
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