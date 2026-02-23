import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_PATH       = DATA_DIR / "bible_raw.json"
SECTIONS_DIR    = DATA_DIR / "sections"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Bible API settings
# ═══════════════════════════════════════════════════════════════════════════════

BIBLE_API_BASE  = "https://bible-api.com"
TRANSLATION     = "kjv"          # King James Version — public domain, clean text
REQUEST_DELAY   = 1.5            # bible-api.com allows ~40 req/min; 1.5s is safe
TIMEOUT         = 15
MAX_RETRIES     = 6              # more retries to ride out 429 bursts

_session = requests.Session()
_session.headers.update({"User-Agent": "MultiReligiousChatbot/3.0 (educational use)"})

# ═══════════════════════════════════════════════════════════════════════════════
# Bible canon — all 66 books with chapter counts
# ═══════════════════════════════════════════════════════════════════════════════

# Format: "Book Name": num_chapters
# Names match bible-api.com's accepted query strings
OLD_TESTAMENT: dict[str, int] = {
    "Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numbers": 36,
    "Deuteronomy": 34, "Joshua": 24, "Judges": 21, "Ruth": 4,
    "1 Samuel": 31, "2 Samuel": 24, "1 Kings": 22, "2 Kings": 25,
    "1 Chronicles": 29, "2 Chronicles": 36, "Ezra": 10, "Nehemiah": 13,
    "Esther": 10, "Job": 42, "Psalms": 150, "Proverbs": 31,
    "Ecclesiastes": 12, "Song of Solomon": 8, "Isaiah": 66, "Jeremiah": 52,
    "Lamentations": 5, "Ezekiel": 48, "Daniel": 12, "Hosea": 14,
    "Joel": 3, "Amos": 9, "Obadiah": 1, "Jonah": 4,
    "Micah": 7, "Nahum": 3, "Habakkuk": 3, "Zephaniah": 3,
    "Haggai": 2, "Zechariah": 14, "Malachi": 4,
}

NEW_TESTAMENT: dict[str, int] = {
    "Matthew": 28, "Mark": 16, "Luke": 24, "John": 21,
    "Acts": 28, "Romans": 16, "1 Corinthians": 16, "2 Corinthians": 13,
    "Galatians": 6, "Ephesians": 6, "Philippians": 4, "Colossians": 4,
    "1 Thessalonians": 5, "2 Thessalonians": 3, "1 Timothy": 6, "2 Timothy": 4,
    "Titus": 3, "Philemon": 1, "Hebrews": 13, "James": 5,
    "1 Peter": 5, "2 Peter": 3, "1 John": 5, "2 John": 1,
    "3 John": 1, "Jude": 1, "Revelation": 22,
}

ALL_BOOKS: dict[str, int] = {**OLD_TESTAMENT, **NEW_TESTAMENT}

# Human-readable section keys (used as section IDs in chunks)
SECTION_LABELS: dict[str, str] = {
    book: book for book in ALL_BOOKS
}

TESTAMENT_MAP: dict[str, str] = {
    **{book: "Old Testament" for book in OLD_TESTAMENT},
    **{book: "New Testament" for book in NEW_TESTAMENT},
}

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_replace(src: str, dst: str):
    """
    Atomically replace `dst` with `src`.
    Falls back through three strategies to handle Windows PermissionError (WinError 5).
    """
    src_p = Path(src)
    dst_p = Path(dst)
    # Strategy 1: standard atomic replace (Linux/macOS always works)
    try:
        src_p.replace(dst_p)
        return
    except PermissionError:
        pass
    # Strategy 2: delete target first, then rename
    try:
        if dst_p.exists():
            dst_p.unlink()
        src_p.rename(dst_p)
        return
    except OSError:
        pass
    # Strategy 3: plain copy + remove (non-atomic but always works)
    try:
        dst_p.write_bytes(src_p.read_bytes())
        src_p.unlink(missing_ok=True)
    except Exception as exc:
        tqdm.write(f"  [warn] _safe_replace failed ({src} -> {dst}): {exc}")


def _get_json(url: str) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            # 429 Too Many Requests — honour Retry-After or use exponential backoff
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 0))
                wait = retry_after if retry_after > 0 else min(15 * attempt, 120)
                tqdm.write(f"  [429] rate-limited — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue   # retry the same attempt counter
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES:
                tqdm.write(f"  [timeout] {url}")
            else:
                time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code
            if 500 <= code < 600 and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  [http {code}] {url}")
                return None
        except Exception as exc:
            tqdm.write(f"  [error] {exc}")
            return None
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# Bible API fetcher
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_chapter(book: str, chapter: int) -> list[str]:
    """
    Fetch all verses from a single chapter via bible-api.com.
    Returns a list of verse strings, each prefixed with the verse number.
    e.g. ["1 In the beginning God created...", "2 And the earth was..."]
    """
    # bible-api.com query format: "John 3" or "1+Corinthians+3"
    query = f"{book} {chapter}".replace(" ", "+")
    url   = f"{BIBLE_API_BASE}/{query}?translation={TRANSLATION}"
    data  = _get_json(url)

    if not data or "verses" not in data:
        return []

    verses = []
    for v in data["verses"]:
        text = v.get("text", "").strip()
        verse_num = v.get("verse", "")
        if text:
            verses.append(f"{verse_num} {text}")
    return verses

# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint & section I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> tuple[set[str], dict]:
    if not CHECKPOINT_PATH.exists():
        return set(), {}
    try:
        cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(cp.get("completed", [])), cp.get("stats", {})
    except Exception:
        return set(), {}


def save_checkpoint(completed: set[str], stats: dict):
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps({"completed": sorted(completed), "stats": stats},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(CHECKPOINT_PATH))


def load_section(book: str) -> list[str]:
    safe_name = book.replace(" ", "_")
    p = SECTIONS_DIR / f"{safe_name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_section(book: str, verses: list[str]):
    safe_name = book.replace(" ", "_")
    p   = SECTIONS_DIR / f"{safe_name}.json"
    tmp = str(p) + ".tmp"
    Path(tmp).write_text(
        json.dumps(verses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(p))


def merge_sections() -> list[dict]:
    """Merge all section files into bible_raw.json with metadata."""
    all_texts: list[dict] = []
    for book in ALL_BOOKS:
        safe_name = book.replace(" ", "_")
        p = SECTIONS_DIR / f"{safe_name}.json"
        if not p.exists():
            continue
        try:
            verses = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for verse in verses:
            all_texts.append({
                "text":      verse,
                "section":   book,
                "testament": TESTAMENT_MAP.get(book, "Unknown"),
            })

    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps(all_texts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(DATA_PATH))
    return all_texts

# ═══════════════════════════════════════════════════════════════════════════════
# Download pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _download_bible(completed: set[str], stats: dict):
    print(f"\n{'═' * 60}")
    print("  Downloading Bible (KJV) via bible-api.com")
    print(f"{'═' * 60}")

    for book, num_chapters in tqdm(ALL_BOOKS.items(), desc="Books", unit="book"):
        testament = TESTAMENT_MAP[book]
        book_verses = load_section(book)
        prev  = stats.get(book, {"ok": 0, "warn": 0, "verses": 0})
        ok    = prev.get("ok", 0)
        warn  = prev.get("warn", 0)

        chapters_done = set(
            c for c in prev.get("chapters_done", [])
        )
        pending_chapters = [
            ch for ch in range(1, num_chapters + 1)
            if f"{book}:{ch}" not in completed
        ]

        if not pending_chapters:
            tqdm.write(f"  ✓ {book} ({testament}) — already complete")
            continue

        tqdm.write(f"\n── {book}  [{testament}]  {num_chapters} chapters")

        for chapter in pending_chapters:
            key = f"{book}:{chapter}"
            verses = fetch_chapter(book, chapter)
            if verses:
                book_verses.extend(verses)
                ok += 1
            else:
                tqdm.write(f"  [warn] empty: {book} {chapter}")
                warn += 1

            completed.add(key)
            stats[book] = {
                "ok": ok, "warn": warn,
                "verses": len(book_verses),
                "chapters_done": sorted(
                    [c for c in stats.get(book, {}).get("chapters_done", [])]
                    + [chapter]
                ),
            }
            save_checkpoint(completed, stats)

        save_section(book, book_verses)
        tqdm.write(f"  → {len(book_verses)} verses saved")


def download_bible() -> list[dict]:
    completed, stats = load_checkpoint()

    _download_bible(completed, stats)

    print(f"\n  Merging all sections → {DATA_PATH} …")
    all_texts = merge_sections()

    total_verses = sum(s.get("verses", 0) for s in stats.values())
    print(f"\n{'═' * 60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  {len(all_texts):,} entries  |  ~{total_verses:,} verses")
    print(f"  Output: {DATA_PATH}")
    print(f"{'═' * 60}")
    return all_texts


if __name__ == "__main__":
    download_bible()