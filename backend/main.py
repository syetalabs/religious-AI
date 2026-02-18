import os
import sys
import subprocess
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CORPUS_PATH     = DATA_DIR / "tipitaka_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks.json"
FAISS_PATH      = DATA_DIR / "faiss_index.bin"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

# ═══════════════════════════════════════════════════════════════════════════════
# Step helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _env_check():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print(
            "\n  ℹ  GITHUB_TOKEN not set.\n"
            "     File downloads use raw.githubusercontent.com (unmetered).\n"
            "     Set GITHUB_TOKEN only if you run this script many times per hour.\n"
        )

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        # Try loading from .env manually as a fallback hint
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            print("  ℹ  .env file found — GROQ_API_KEY will be loaded from it.\n")
        else:
            print(
                "\n  ⚠  GROQ_API_KEY is not set and no .env file found.\n"
                "     Create a .env file in this folder with:\n"
                "       GROQ_API_KEY=your_key_here\n"
                "     Get your free key at https://console.groq.com\n"
            )
            sys.exit(1)


def _step_download():
    """Download corpus if not already present."""
    if CORPUS_PATH.exists() and CORPUS_PATH.stat().st_size > 1000:
        print(f"  ✓ Corpus already downloaded → {CORPUS_PATH}")
        return

    print("\n━━━  STEP 1: Downloading Tipitaka Corpus  ━━━\n")
    from data_loader import download_tipitaka
    download_tipitaka()
    print(f"  ✓ Corpus saved → {CORPUS_PATH}")


def _step_chunk_and_embed():
    """Build chunks + FAISS index if not already present."""
    if CHUNKS_PATH.exists() and FAISS_PATH.exists():
        import json
        chunk_count = len(json.loads(CHUNKS_PATH.read_text(encoding="utf-8")))
        print(f"  ✓ Chunks already built ({chunk_count:,} chunks) → {CHUNKS_PATH}")
        print(f"  ✓ FAISS index already built → {FAISS_PATH}")
        return

    print("\n━━━  STEP 2: Chunking & Embedding  ━━━\n")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "chunk_and_embed.py")],
        check=True
    )
    print("  ✓ Chunks and FAISS index built successfully.")


def _print_banner():
    print("\n" + "═" * 55)
    print("   🪷  Buddhist AI Chatbot  🪷")
    print("   Powered by: Groq  (llama-3.1-8b-instant)")
    print("   Knowledge:  Pali Canon (Sutta, Vinaya, Abhidhamma)")
    print("═" * 55)
    print("   Type your question and press Enter.")
    print("   Type 'quit' or 'exit' to end the session.")
    print("═" * 55 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _env_check()

    # ── Step 1: Download corpus ───────────────────────────────────────────────
    _step_download()

    # ── Step 2: Chunk + embed ─────────────────────────────────────────────────
    _step_chunk_and_embed()

    # ── Step 3: Launch chatbot ────────────────────────────────────────────────
    print("\n━━━  STEP 3: Starting Chatbot  ━━━")

    from rag_answer import answer_question, GROQ_MODEL

    _print_banner()

    while True:
        try:
            question = input("Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye. May you be well. 🙏")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye. May you be well. 🙏")
            break

        print("\n  Searching scriptures...\n")

        result = answer_question(question)

        print("─" * 55)
        print(result["answer"])
        print("─" * 55)

        if result["sources"]:
            print("\n📖 Sources:")
            for book, score in zip(result["sources"], result["scores"]):
                print(f"   • {book}  (relevance: {score:.3f})")

        if result.get("flagged"):
            print(f"\n⚠  Input flagged: {', '.join(result['warnings'])}")

        if result.get("low_confidence"):
            print("\n⚠  Low confidence — no strong scriptural match found.")

        if result.get("warnings") and not result.get("flagged"):
            print(f"\n[Warnings: {', '.join(result['warnings'])}]")

        print()


if __name__ == "__main__":
    main()