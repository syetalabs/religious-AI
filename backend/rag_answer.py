import os
import re
import time
import requests
from dotenv import load_dotenv
from retrieve import search

# ────────────────────────────────────────────────────────────────
# Groq API settings
# ────────────────────────────────────────────────────────────────

load_dotenv()  # loads variables from .env file automatically

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "Create a .env file in your backend folder with:\n"
        "  GROQ_API_KEY=your_key_here\n"
        "Get your free key at https://console.groq.com"
    )

# ────────────────────────────────────────────────────────────────
# Persona prompt
# ────────────────────────────────────────────────────────────────

BUDDHIST_PERSONA = """You are a knowledgeable and compassionate Buddhist guide speaking to someone new to Buddhism.

Rules you must follow without exception:
- Answer ONLY using the scripture context provided. Do not use your own knowledge.
- If the context does not contain enough information, say exactly:
  "I do not have enough reliable scriptural context to answer this accurately."
- Use simple, clear, everyday language. Avoid Pali technical terms unless you explain them.
- Begin with the most human, relatable aspect of the teaching before going deeper.
- Keep answers concise — 3 to 5 sentences unless the question requires more.
- Do not provide personal opinions or moral judgments.
- Do not compare Buddhism with other religions.
- Do not mix teachings from other traditions.
- Maintain a calm, warm, and welcoming tone at all times.
- When referencing scripture, mention the source book naturally (e.g. "As taught in the Digha Nikaya...")."""

# ────────────────────────────────────────────────────────────────
# Moderation Layer 1 — Input (Pre-processing)
# ────────────────────────────────────────────────────────────────

_UNSAFE_PHRASES = [
    "which religion is correct", "which religion is better",
    "which religion is true", "which religion is superior",
    "best religion", "true religion", "only true religion",
    "hate", "kill", "destroy", "inferior religion", "false religion",
]
_COMPARATIVE_PATTERNS = [
    r"\b(buddhism|christianity|islam|hinduism)\b.{0,30}\b(better|worse|superior|inferior|correct|wrong|true|false)\b",
    r"\b(better|worse|superior|inferior|correct|wrong|true|false)\b.{0,30}\b(buddhism|christianity|islam|hinduism)\b",
    r"compare.{0,20}(religion|buddhism|christianity|islam|hinduism)",
    r"which (god|religion|faith|belief).{0,20}(right|correct|true|real|better)",
]
_HATE_PATTERNS = [
    r"\b(hate|despise|destroy|eliminate|eradicate).{0,20}(religion|muslim|christian|hindu|buddhist|jew)\b",
    r"\b(religion|faith).{0,20}(cancer|disease|evil|poison|lie|scam|fraud)\b",
]

def moderate_input(query: str) -> tuple[bool, str]:
    q = query.lower().strip()
    for phrase in _UNSAFE_PHRASES:
        if phrase in q:
            return False, "comparative_or_unsafe"
    for pattern in _COMPARATIVE_PATTERNS:
        if re.search(pattern, q):
            return False, "comparative_religion"
    for pattern in _HATE_PATTERNS:
        if re.search(pattern, q):
            return False, "hate_speech"
    return True, "ok"

_FALLBACK_MESSAGES = {
    "comparative_religion": (
        "Questions comparing religions fall outside the scope of this guide. "
        "Please ask about a specific Buddhist concept or practice."
    ),
    "hate_speech": (
        "This question contains language that is not appropriate for a respectful "
        "religious discussion. Please rephrase your question."
    ),
    "comparative_or_unsafe": (
        "This question may cause religious conflict and cannot be answered. "
        "Please ask about Buddhist teachings specifically."
    ),
}

# ────────────────────────────────────────────────────────────────
# Moderation Layer 3 — Output (Post-response validation)
# ────────────────────────────────────────────────────────────────

_OTHER_RELIGION_TERMS = [
    "christianity", "islam", "hinduism", "judaism", "sikhism",
    "quran", "bible", "torah", "vedas", "gita",
    "jesus", "allah", "brahma", "vishnu", "shiva", "moses",
    "church", "mosque",
]
_OPINION_SIGNALS = [
    r"\bi (think|believe|feel|personally|would say)\b",
    r"\bmy (view|opinion|take|perspective)\b",
]

def moderate_output(response: str) -> tuple[str, list[str]]:
    warnings = []
    r = response.lower()
    for term in _OTHER_RELIGION_TERMS:
        if term in r:
            warnings.append(f"cross_religion_term:{term}")
            return (
                "I was unable to generate a fully grounded response from the "
                "Buddhist scripture context. Please rephrase your question.",
                warnings,
            )
    for pattern in _OPINION_SIGNALS:
        if re.search(pattern, r):
            warnings.append("personal_opinion_detected")
            break
    if len(response.strip()) < 30:
        warnings.append("response_too_short")
        return (
            "I do not have enough reliable scriptural context to answer this accurately.",
            warnings,
        )
    return response, warnings

# ────────────────────────────────────────────────────────────────
# Groq API call
# ────────────────────────────────────────────────────────────────

def _call_groq(system_prompt: str, user_message: str) -> str:
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "temperature": 0.3,
                "max_tokens":  512,
            },
            timeout=30,
        )
        if resp.status_code == 401:
            return "[ERROR] Invalid Groq API key. Check the key at console.groq.com"
        if resp.status_code == 429:
            print("  [info] Rate limit hit, waiting 10s...")
            time.sleep(10)
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    "temperature": 0.3,
                    "max_tokens":  512,
                },
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Groq. Check your internet connection."
    except Exception as e:
        return f"[ERROR] Groq call failed: {e}"
    
# ────────────────────────────────────────────────────────────────
# Retrieval Post-Processing
# ────────────────────────────────────────────────────────────────
def _refine_results(results: list[dict]) -> list[dict]:
    seen_books = set()
    seen_text_hashes = set()
    refined = []

    # 1️⃣ Doctrinal prioritization
    # Sutta Pitaka first, then others
    results_sorted = sorted(
        results,
        key=lambda r: (
            0 if r.get("pitaka", "").lower() == "sutta pitaka" else 1,
            -r["score"]
        )
    )

    for r in results_sorted:
        text_hash = hash(r["text"][:200])  # avoid full long hashing

        # 2️⃣ Remove duplicate books
        if r["book"] in seen_books:
            continue

        # 3️⃣ Remove near-duplicate chunks
        if text_hash in seen_text_hashes:
            continue

        seen_books.add(r["book"])
        seen_text_hashes.add(text_hash)
        refined.append(r)

        # Limit to max 4 strong sources
        if len(refined) >= 4:
            break

    return refined

# ────────────────────────────────────────────────────────────────
# Core answer function
# ────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    religion: str = "Buddhism",
    language: str = "en",
) -> dict:

    # Layer 1 — Input moderation
    is_safe, reason = moderate_input(question)
    if not is_safe:
        return {
            "answer":         _FALLBACK_MESSAGES.get(reason, "This question cannot be answered."),
            "sources":        [], 
            "scores":        [],
            "flagged":        True,
            "low_confidence": False,
            "warnings":       [reason],
        }

    # Layer 2 — Retrieval guardrail
    results = search(question, religion=religion, language=language)
    if not results:
        return {
            "answer":         "I do not have enough reliable scriptural context to answer this accurately.",
            "sources":        [], "scores":        [],
            "flagged":        False,
            "low_confidence": True,
            "warnings":       [],
        }
    
    results = _refine_results(results)

    # Context injection
    context = "\n\n---\n\n".join(
        f"[Source: {r['book']} | {r['pitaka']}]\n{r['text']}"
        for r in results
    )
    user_message = f"""Scripture context:
{context}

Question: {question}

Instructions:
- Start with a simple, human explanation a beginner can understand.
- Then support it with what the scripture says, citing the source naturally.
- If the teaching has a practical dimension, briefly mention it.
- Do not use bullet points or lists. Write in flowing, warm prose.

Answer:"""

    # LLM generation
    raw_answer = _call_groq(BUDDHIST_PERSONA, user_message)

    # Layer 3 — Output moderation
    final_answer, warnings = moderate_output(raw_answer)

    return {
        "answer":         final_answer,
        "sources":        [r["book"]  for r in results],
        "scores":         [r["score"] for r in results],
        "flagged":        False,
        "low_confidence": False,
        "warnings":       warnings,
    }

# ────────────────────────────────────────────────────────────────
# CLI runner
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Buddhist RAG Chatbot  —  Groq ({GROQ_MODEL})")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            q = input("\nAsk: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not q:
            continue

        print("\nThinking...")
        result = answer_question(q)

        print("\n--- Answer ---")
        print(result["answer"])

        if result["sources"]:
            print("\n--- Sources ---")
            for book, score in zip(result["sources"], result["scores"]):
                print(f"  {book}  (similarity: {score:.3f})")

        if result["warnings"]:
            print(f"\n[Warnings: {', '.join(result['warnings'])}]")
        if result["low_confidence"]:
            print("\n[Low confidence — no chunks met the similarity threshold]")
        if result["flagged"]:
            print("\n[Input was flagged by moderation]")