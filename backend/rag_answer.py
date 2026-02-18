import os
import re
import time
import requests
from dotenv import load_dotenv
from retrieve import search

# ────────────────────────────────────────────────────────────────
# Groq API settings
# ────────────────────────────────────────────────────────────────

load_dotenv()

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
- Answer ONLY using the scripture context provided between [Source: ...] tags. Do not use your own knowledge under any circumstances.
- If the context does not contain enough information, say exactly:
  "I do not have enough reliable scriptural context to answer this accurately."
- Use simple, clear, everyday language. Avoid Pali technical terms unless you explain them.
- Begin with the most human, relatable aspect of the teaching before going deeper.
- Keep answers concise — 3 to 5 sentences unless the question requires more.
- Do not provide personal opinions or moral judgments.
- Do not compare Buddhism with other religions.
- Do not mix teachings from other traditions.
- Maintain a calm, warm, and welcoming tone at all times.
- When referencing scripture, mention the source book naturally (e.g. "As taught in the Digha Nikaya...").

CRITICAL — Quoting rules:
- Do NOT quote any specific verse numbers or references (e.g. SN 56.11, MN 22) unless those exact numbers appear word-for-word inside the provided scripture context.
- Do NOT reproduce or paraphrase text that is not present in the provided context.
- If you are unsure whether a quote exists in the context, do not include it.
- Never invent or recall quotes from memory — only use what is explicitly in the context."""

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

# Matches verse references like SN 56.11 / MN 22 / DN 16 / AN 3.65 etc.
_VERSE_REF_PATTERN = re.compile(
    r"\b(SN|MN|DN|AN|KN|Dhp|Ud|Iti|Snp|Thag|Thig|Ja|Cp|Mil|Pv|Vv|Vb|Ds|Kv|Pp|Ps|Ya)\s*\d+[\.\d]*\b",
    re.IGNORECASE,
)

def _check_fabricated_references(response: str, context: str) -> list[str]:
    """
    Detect verse references in the response that do not appear in the
    retrieved context — a strong signal of hallucination.
    """
    warnings = []
    found_refs = _VERSE_REF_PATTERN.findall(response)
    for ref in found_refs:
        # Search the full match, not just the abbreviation
        pass

    # Re-find full matches (abbreviation + number)
    full_refs = _VERSE_REF_PATTERN.finditer(response)
    for match in full_refs:
        ref_str = match.group(0)
        if ref_str.lower() not in context.lower():
            warnings.append(f"fabricated_reference:{ref_str}")

    return warnings

def moderate_output(response: str, context: str = "") -> tuple[str, list[str]]:
    warnings = []
    r = response.lower()

    # Cross-religion term check
    for term in _OTHER_RELIGION_TERMS:
        if term in r:
            warnings.append(f"cross_religion_term:{term}")
            return (
                "I was unable to generate a fully grounded response from the "
                "Buddhist scripture context. Please rephrase your question.",
                warnings,
            )

    # Personal opinion check
    for pattern in _OPINION_SIGNALS:
        if re.search(pattern, r):
            warnings.append("personal_opinion_detected")
            break

    # Response too short
    if len(response.strip()) < 30:
        warnings.append("response_too_short")
        return (
            "I do not have enough reliable scriptural context to answer this accurately.",
            warnings,
        )

    # Fabricated verse reference check
    ref_warnings = _check_fabricated_references(response, context)
    if ref_warnings:
        warnings.extend(ref_warnings)
        # Strip the fabricated references from the response rather than blocking entirely
        cleaned = _VERSE_REF_PATTERN.sub("", response)
        # Clean up orphaned parentheses like ( ) or ()
        cleaned = re.sub(r"\(\s*\)", "", cleaned).strip()
        return cleaned, warnings

    return response, warnings

# ────────────────────────────────────────────────────────────────
# Groq API call
# ────────────────────────────────────────────────────────────────

def _call_groq(system_prompt: str, user_message: str) -> str:
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.1,   # lowered from 0.3 to reduce creative generation
        "max_tokens":  512,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)

        if resp.status_code == 401:
            return "[ERROR] Invalid Groq API key. Check the key at console.groq.com"

        if resp.status_code == 429:
            print("  [info] Rate limit hit, waiting 10s...")
            time.sleep(10)
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)

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

    # Sutta Pitaka first, then others — within each group sort by score
    results_sorted = sorted(
        results,
        key=lambda r: (
            0 if r.get("pitaka", "").lower() == "sutta pitaka" else 1,
            -r["score"]
        )
    )

    for r in results_sorted:
        text_hash = hash(r["text"][:200])

        if r["book"] in seen_books:
            continue
        if text_hash in seen_text_hashes:
            continue

        seen_books.add(r["book"])
        seen_text_hashes.add(text_hash)
        refined.append(r)

        if len(refined) >= 4:
            break

    return refined

# ────────────────────────────────────────────────────────────────
# Intent detection
# ────────────────────────────────────────────────────────────────

_LIST_PATTERNS = [
    r"\b(list|enumerate|show|give me|what are|name|which books?|which texts?|which suttas?)\b",
    r"\b(all the|all books|books (about|on|that))\b",
]

def _is_list_request(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _LIST_PATTERNS)

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
            "sources":        [],
            "scores":         [],
            "flagged":        False,
            "low_confidence": True,
            "warnings":       [],
        }
    
    results = _refine_results(results)

    # Build context string (also passed to output moderation for reference checking)
    context = "\n\n---\n\n".join(
        f"[Source: {r['book']} | {r['pitaka']}]\n{r['text']}"
        for r in results
    )

    if _is_list_request(question):
        format_instructions = """- The user is asking for a list. Respond with a clear, direct list of book/source names found in the context.
- Format: one book per line, with a one-sentence description of what it covers regarding the topic.
- Do not write long prose. Be concise and direct.
- Only include books that are present in the provided scripture context."""
    else:
        format_instructions = """- Start with a simple, human explanation a beginner can understand.
- Then support it with what the scripture says, citing the source book name naturally (e.g. "As found in the Samyutta Nikaya...").
- If the teaching has a practical dimension, briefly mention it.
- Do not use bullet points or lists. Write in flowing, warm prose."""

    user_message = f"""Scripture context:
{context}

Question: {question}

Instructions:
{format_instructions}
- Do NOT cite specific verse numbers (like SN 56.11) unless those exact numbers appear in the context above.

Answer:"""

    # LLM generation
    raw_answer = _call_groq(BUDDHIST_PERSONA, user_message)

    # Layer 3 — Output moderation (pass context for fabrication check)
    final_answer, warnings = moderate_output(raw_answer, context=context)

    # Source reconciliation — only show sources the LLM actually mentioned.
    # Falls back to all retrieved results if none are detected in the answer.
    answer_lower = final_answer.lower()
    matched = [r for r in results if r["book"].lower() in answer_lower]
    display_results = matched if matched else results

    return {
        "answer":         final_answer,
        "sources":        [r["book"]  for r in display_results],
        "scores":         [r["score"] for r in display_results],
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