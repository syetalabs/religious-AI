import os
import sys

from data_loader import (
    DATA_PATH,
    SECTIONS_DIR,
    SC_API_SECTIONS,
    VINAYA_SECTIONS,
    LEGACY_SECTIONS,
    SECTION_LABELS,
    download_tipitaka,
    load_all_texts,
    load_section,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _section_summary():
    """Print a table showing how many segments were saved per section."""
    all_sections = (
        list(SC_API_SECTIONS)
        + list(VINAYA_SECTIONS)
        + list(LEGACY_SECTIONS)
    )
    print(f"\n{'─'*50}")
    print(f"  {'Section':<22}  {'Label':<26}  Segments")
    print(f"{'─'*50}")
    for section in all_sections:
        segs  = load_section(section)
        label = SECTION_LABELS.get(section, section.upper())
        mark  = "✓" if segs else "✗"
        print(f"  {mark} {section:<20}  {label:<26}  {len(segs):>7,}")
    print(f"{'─'*50}")


def _env_check():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print(
            "\n  ℹ  GITHUB_TOKEN not set.  The downloader will still work — Phase 2\n"
            "     uses at most ~4 GitHub API calls (well within the 60/hr limit).\n"
            "     A token is only needed if you run this script many times in an hour.\n"
            "     To set one:  export GITHUB_TOKEN=<your-pat>\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _env_check()

    # ── Step 1: Download / resume ────────────────────────────────────────────
    print("Starting full Tipitaka download …")
    print("(Already-completed sections will be skipped via checkpoint)\n")

    all_texts = download_tipitaka()

    # ── Step 2: Per-section summary ──────────────────────────────────────────
    _section_summary()

    # ── Step 3: Load merged corpus into memory ───────────────────────────────
    print(f"\n  Loading merged corpus from {DATA_PATH} …")
    corpus = load_all_texts()
    print(f"  Total segments in memory: {len(corpus):,}")

    # ── Step 4: Quick sanity checks ──────────────────────────────────────────
    if not corpus:
        print("\n  [ERROR] Corpus is empty — check warnings above.", file=sys.stderr)
        sys.exit(1)

    non_empty = [s for s in corpus if s.strip()]
    avg_len   = sum(len(s) for s in non_empty) / max(len(non_empty), 1)
    print(f"  Non-empty segments:       {len(non_empty):,}")
    print(f"  Average segment length:   {avg_len:.0f} characters")
    print(f"\n  Sample segments (first 3):")
    for i, seg in enumerate(non_empty[:3], 1):
        preview = seg[:120].replace("\n", " ")
        print(f"    [{i}] {preview}{'…' if len(seg) > 120 else ''}")

    print("\n  ✓ Pipeline complete.  corpus variable is ready for use.\n")
    return corpus


if __name__ == "__main__":
    corpus = main()