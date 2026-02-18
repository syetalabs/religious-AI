import json
import re
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "backend" / "data"

CHUNKS_PATH = DATA_DIR / "chunks.json"

# ────────────────────────────────────────────────────────────────
# Load chunks once at import (cached)
# ────────────────────────────────────────────────────────────────

print("Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"  {len(chunks):,} chunks loaded")
print("  Keyword search mode enabled")

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────

TOP_K = 100  # return many matches
_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "and",
    "to", "in", "on", "for", "with", "that",
    "does", "do", "are", "was", "were", "be",
    "this", "these", "those", "it", "as",
}

# ────────────────────────────────────────────────────────────────
# Keyword extraction
# ────────────────────────────────────────────────────────────────

def _extract_keywords(query: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]

# ────────────────────────────────────────────────────────────────
# Search
# ────────────────────────────────────────────────────────────────

def search(
    query: str,
    religion: str = "Buddhism",
    top_k: int = TOP_K,
    language: str = "en",
) -> list[dict]:
    """
    Keyword-based retrieval.
    Returns ALL chunks containing the keywords from the question.
    """

    keywords = _extract_keywords(query)

    if not keywords:
        return []

    results = []

    for chunk in chunks:

        # 1️⃣ Religion filter
        if chunk.get("religion") != religion:
            continue

        # 2️⃣ Language filter
        if chunk.get("language", "en") != language:
            continue

        text_lower = chunk["text"].lower()

        # 3️⃣ Keyword match count
        match_count = sum(1 for k in keywords if k in text_lower)

        if match_count == 0:
            continue

        # Simple relevance score
        score = match_count / len(keywords)

        results.append({
            "text":     chunk["text"],
            "book":     chunk["book"],
            "pitaka":   chunk.get("pitaka", ""),
            "source":   chunk["source"],
            "religion": chunk["religion"],
            "language": chunk.get("language", "en"),
            "score":    float(score),
        })

    # Sort by keyword density
    results.sort(key=lambda r: r["score"], reverse=True)

    return results[:top_k]

# ────────────────────────────────────────────────────────────────
# CLI test
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nKeyword Retrieval Test  (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            q = input("\nQuery: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if q.lower() in ("quit", "exit", "q"):
            break

        if not q:
            continue

        res = search(q)

        if not res:
            print("No matching chunks found.")
            continue

        for r in res:
            print(f"\n[{r['book']} | {r['pitaka']}]  score={r['score']:.3f}")
            print(r["text"][:300] + ("…" if len(r["text"]) > 300 else ""))
