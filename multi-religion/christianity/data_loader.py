import io
import json
import re
import time
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup
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
# Bible API settings (English KJV via bible-api.com)
# ═══════════════════════════════════════════════════════════════════════════════

BIBLE_API_BASE  = "https://bible-api.com"
TRANSLATION     = "kjv"
REQUEST_DELAY   = 1.5
TIMEOUT         = 30
MAX_RETRIES     = 6

_session = requests.Session()
_session.headers.update({"User-Agent": "MultiReligiousChatbot/3.0 (educational use)"})

# ═══════════════════════════════════════════════════════════════════════════════
# Wordproject offline zips — no API key, no login required
# To add more languages, just add an entry: "lang_code": "zip_url"
# ═══════════════════════════════════════════════════════════════════════════════

# Maps our language key -> Wordproject zip URL
WP_LANGUAGES: dict[str, str] = {
    "si": "https://www.wordpocket.org/bibles/download_zip/si_new.zip",
    "ta": "https://www.wordpocket.org/bibles/download_zip/tm_new.zip",
}

# Maps our language key -> folder name INSIDE the zip (may differ from our key)
WP_LANG_FOLDER: dict[str, str] = {
    "si": "si",
    "ta": "tm",   # Wordproject uses "tm" for Tamil internally
}

def _wp_zip_path(lang: str) -> Path:
    return DATA_DIR / f"{lang}_new.zip"

def _wp_html_dir(lang: str) -> Path:
    return DATA_DIR / f"{lang}_html"

# ═══════════════════════════════════════════════════════════════════════════════
# Bible canon — all 66 books with chapter counts
# ═══════════════════════════════════════════════════════════════════════════════

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

TESTAMENT_MAP: dict[str, str] = {
    **{book: "Old Testament" for book in OLD_TESTAMENT},
    **{book: "New Testament" for book in NEW_TESTAMENT},
}

# Wordproject uses a 2-digit numeric folder per book (01=Genesis … 66=Revelation)
_BOOK_TO_WP_NUM: dict[str, str] = {
    book: str(i).zfill(2)
    for i, book in enumerate(ALL_BOOKS.keys(), start=1)
}

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_replace(src: str, dst: str):
    src_p, dst_p = Path(src), Path(dst)
    try:
        src_p.replace(dst_p); return
    except PermissionError:
        pass
    try:
        if dst_p.exists(): dst_p.unlink()
        src_p.rename(dst_p); return
    except OSError:
        pass
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
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 0)) or min(15 * attempt, 120)
                print(f"  [429] waiting {wait}s (attempt {attempt}/{MAX_RETRIES})", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            try:
                return resp.json()
            except Exception:
                return None
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES: time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError as exc:
            print(f"  [connection-error] {exc}", flush=True)
            if attempt < MAX_RETRIES: time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code
            if 500 <= code < 600 and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}", flush=True)
            return None
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# English KJV fetcher (bible-api.com)
# ═══════════════════════════════════════════════════════════════════════════════

_BOOK_SLUG: dict[str, str] = {
    "1 Samuel": "1samuel", "2 Samuel": "2samuel",
    "1 Kings": "1kings", "2 Kings": "2kings",
    "1 Chronicles": "1chronicles", "2 Chronicles": "2chronicles",
    "1 Corinthians": "1corinthians", "2 Corinthians": "2corinthians",
    "1 Thessalonians": "1thessalonians", "2 Thessalonians": "2thessalonians",
    "1 Timothy": "1timothy", "2 Timothy": "2timothy",
    "1 Peter": "1peter", "2 Peter": "2peter",
    "1 John": "1john", "2 John": "2john", "3 John": "3john",
    "Song of Solomon": "songofsolomon",
}

_SINGLE_CHAPTER_VERSE_COUNT: dict[str, int] = {
    "Obadiah": 21, "Philemon": 25, "2 John": 13, "3 John": 14, "Jude": 25,
}


def _build_url(book: str, chapter: int) -> str:
    slug = _BOOK_SLUG.get(book, book)
    path = f"{slug}+{chapter}".replace(" ", "+")
    return f"{BIBLE_API_BASE}/{path}?translation={TRANSLATION}"


def _build_verse_url(book: str, chapter: int, verse: int) -> str:
    slug = _BOOK_SLUG.get(book, book)
    path = f"{slug}+{chapter}:{verse}".replace(" ", "+")
    return f"{BIBLE_API_BASE}/{path}?translation={TRANSLATION}"


def _parse_verses(data: dict) -> list[str]:
    return [
        f"{v.get('verse', '')} {v.get('text', '').strip()}"
        for v in data.get("verses", [])
        if v.get("text", "").strip()
    ]


def fetch_chapter_en(book: str, chapter: int, total_chapters: int) -> list[str]:
    if total_chapters == 1:
        n_verses = _SINGLE_CHAPTER_VERSE_COUNT.get(book, 30)
        all_verses = []
        for v_num in range(1, n_verses + 1):
            data = _get_json(_build_verse_url(book, chapter, v_num))
            if data and "verses" in data:
                all_verses.extend(_parse_verses(data))
            else:
                break
        return all_verses
    else:
        data = _get_json(_build_url(book, chapter))
        return _parse_verses(data) if data and "verses" in data else []

# ═══════════════════════════════════════════════════════════════════════════════
# Wordproject offline zip fetcher — works for any language (si, ta, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

def _download_wp_zip(lang: str):
    """Download a Wordproject Bible zip once and cache it locally."""
    zip_path = _wp_zip_path(lang)
    zip_url  = WP_LANGUAGES[lang]
    if zip_path.exists() and zip_path.stat().st_size > 100_000:
        print(f"  ✓ {lang.upper()} zip already downloaded ({zip_path.name})")
        return
    print(f"  📥 Downloading {lang.upper()} Bible zip from Wordproject…")
    print(f"     URL: {zip_url}")
    resp = _session.get(zip_url, timeout=120, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"  Downloading ({lang})"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  ✓ Saved → {zip_path}")


def _extract_wp_zip(lang: str):
    """Extract the zip into DATA_DIR/<lang>_html/ if not already done."""
    html_dir = _wp_html_dir(lang)
    if html_dir.exists() and any(html_dir.rglob("*.htm")):
        print(f"  ✓ {lang.upper()} HTML files already extracted")
        return
    print(f"  📂 Extracting {lang.upper()} Bible zip…")
    html_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_wp_zip_path(lang), "r") as zf:
        zf.extractall(html_dir)
    htm_count = len(list(html_dir.rglob("*.htm")))
    print(f"  ✓ Extracted {htm_count} HTML files → {html_dir}")


def _parse_wp_chapter_html(html_content: str) -> list[str]:
    """
    Parse a Wordproject chapter HTML file and return verse strings.
    Format: "VERSE_NUM verse text here"

    Structure (confirmed from actual files):
      <div id="textBody">
        <p>
          <span class="verse" id="1">1 </span>verse text...
          <br/>
          <span class="verse" id="2">2 </span>verse text...
          ...
        </p>
      </div>

    Verses are <span class="verse"> nodes whose immediately following
    NavigableString siblings (up to the next <span class="verse">) are the text.
    """
    from bs4 import NavigableString, Tag

    soup = BeautifulSoup(html_content, "html.parser")

    body = (
        soup.find("div", id="textBody")
        or soup.find("div", class_="textBody")
        or soup.find("div", id="content")
    )
    if not body:
        return []

    verses: list[str] = []

    for span in body.find_all("span", class_="verse"):
        verse_num = span.get_text(strip=True).strip(".")
        if not verse_num.isdigit():
            continue

        text_parts: list[str] = []
        for sibling in span.next_siblings:
            if isinstance(sibling, NavigableString):
                t = str(sibling).strip()
                if t:
                    text_parts.append(t)
            elif isinstance(sibling, Tag):
                if sibling.name == "span" and "verse" in (sibling.get("class") or []):
                    break
                if sibling.name == "br":
                    continue
                t = sibling.get_text(strip=True)
                if t:
                    text_parts.append(t)

        text = " ".join(text_parts).strip()
        text = re.sub(r"^\d+\s*", "", text).strip()
        if text:
            verses.append(f"{verse_num} {text}")

    return verses


def _find_chapter_html(lang: str, book: str, chapter: int) -> Path | None:
    """Locate the HTML file for a given lang+book+chapter inside the extracted zip."""
    book_num   = _BOOK_TO_WP_NUM[book]
    html_dir   = _wp_html_dir(lang)
    wp_folder  = WP_LANG_FOLDER.get(lang, lang)  # internal folder name inside zip

    # Primary: <wp_folder>/<book_num>/<chapter>.htm
    primary = html_dir / wp_folder / book_num / f"{chapter}.htm"
    if primary.exists():
        return primary

    # Alternative flat structure
    alt = html_dir / book_num / f"{chapter}.htm"
    if alt.exists():
        return alt

    # Recursive search as last resort
    candidates = [
        p for p in html_dir.rglob(f"{chapter}.htm")
        if book_num in p.parts
    ]
    return candidates[0] if candidates else None


def fetch_chapter_wp(lang: str, book: str, chapter: int) -> list[str]:
    """Return parsed verses for a chapter from the locally extracted zip."""
    html_path = _find_chapter_html(lang, book, chapter)
    if not html_path:
        tqdm.write(f"  [warn] {lang.upper()} HTML not found: {book} ch{chapter}")
        return []
    content = html_path.read_text(encoding="utf-8", errors="replace")
    return _parse_wp_chapter_html(content)

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


def _section_path(book: str, lang: str) -> Path:
    safe_name = book.replace(" ", "_")
    return SECTIONS_DIR / f"{safe_name}_{lang}.json"


def load_section(book: str, lang: str = "en") -> list[str]:
    p = _section_path(book, lang)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_section(book: str, verses: list[str], lang: str = "en"):
    p   = _section_path(book, lang)
    tmp = str(p) + ".tmp"
    Path(tmp).write_text(
        json.dumps(verses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(p))


def merge_sections() -> list[dict]:
    """Merge all section files (en + all WP languages) into bible_raw.json."""
    all_langs = ["en"] + list(WP_LANGUAGES.keys())
    all_texts: list[dict] = []
    for lang in all_langs:
        for book in ALL_BOOKS:
            p = _section_path(book, lang)
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
                    "language":  lang,
                })

    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps(all_texts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(DATA_PATH))
    return all_texts

# ═══════════════════════════════════════════════════════════════════════════════
# Download pipelines
# ═══════════════════════════════════════════════════════════════════════════════

def _download_english(completed: set[str], stats: dict):
    print(f"\n{'═' * 60}")
    print("  Downloading Bible (KJV) via bible-api.com")
    print(f"{'═' * 60}")

    for book, num_chapters in tqdm(ALL_BOOKS.items(), desc="Books (EN)", unit="book"):
        testament   = TESTAMENT_MAP[book]
        book_verses = load_section(book, "en")
        prev        = stats.get(f"{book}:en", {"ok": 0, "warn": 0, "verses": 0})
        ok, warn    = prev.get("ok", 0), prev.get("warn", 0)

        pending = [ch for ch in range(1, num_chapters + 1)
                   if f"{book}:{ch}:en" not in completed]

        if not pending:
            tqdm.write(f"  ✓ {book} ({testament}) EN — already complete")
            continue

        tqdm.write(f"\n── {book}  [{testament}]  {num_chapters} chapters")

        for chapter in pending:
            key    = f"{book}:{chapter}:en"
            verses = fetch_chapter_en(book, chapter, num_chapters)
            if verses:
                book_verses.extend(verses)
                ok += 1
            else:
                tqdm.write(f"  [warn] empty EN: {book} {chapter}")
                warn += 1

            completed.add(key)
            stats[f"{book}:en"] = {"ok": ok, "warn": warn, "verses": len(book_verses)}
            save_checkpoint(completed, stats)

        save_section(book, book_verses, "en")
        tqdm.write(f"  → {len(book_verses)} EN verses saved")


def _download_wp_language(lang: str, completed: set[str], stats: dict):
    """Generic Wordproject language downloader — works for si, ta, and any future language."""
    lang_names = {"si": "Sinhala", "ta": "Tamil"}
    label = lang_names.get(lang, lang.upper())

    print(f"\n{'═' * 60}")
    print(f"  Downloading Bible ({label}) from Wordproject zip")
    print(f"{'═' * 60}")

    _download_wp_zip(lang)
    _extract_wp_zip(lang)

    for book, num_chapters in tqdm(ALL_BOOKS.items(), desc=f"Books ({lang.upper()})", unit="book"):
        key_book = f"{book}:{lang}"
        if key_book in completed:
            tqdm.write(f"  ✓ {book} {lang.upper()} — already complete")
            continue

        book_verses = load_section(book, lang)
        new_verses  = []

        for chapter in range(1, num_chapters + 1):
            verses = fetch_chapter_wp(lang, book, chapter)
            new_verses.extend(verses)

        if new_verses:
            book_verses.extend(new_verses)
            save_section(book, book_verses, lang)
            tqdm.write(f"  ✓ {book}: {len(new_verses)} {lang.upper()} verses")
        else:
            tqdm.write(f"  [warn] {book}: no {label} verses found")

        completed.add(key_book)
        stats[key_book] = {"verses": len(book_verses)}
        save_checkpoint(completed, stats)


def _reset_lang_checkpoint(lang: str, completed: set[str], stats: dict):
    """Remove checkpoint entries and section files for a language so parser re-runs."""
    suffix = f":{lang}"
    for k in [k for k in list(completed) if k.endswith(suffix)]:
        completed.discard(k)
    for k in [k for k in list(stats.keys()) if k.endswith(suffix)]:
        stats.pop(k, None)
    for p in SECTIONS_DIR.glob(f"*_{lang}.json"):
        p.unlink(missing_ok=True)


def download_bible() -> list[dict]:
    completed, stats = load_checkpoint()

    # Auto-repair: if a WP language was "completed" but has 0 verses, reset it
    for lang in WP_LANGUAGES:
        lang_completed = sum(1 for k in completed if k.endswith(f":{lang}"))
        lang_verses    = sum(
            v.get("verses", 0) for k, v in stats.items()
            if k.endswith(f":{lang}") and isinstance(v, dict)
        )
        if lang_completed > 0 and lang_verses == 0:
            print(f"  [repair] {lang.upper()} verse count is 0 — resetting for re-extraction…")
            _reset_lang_checkpoint(lang, completed, stats)
            save_checkpoint(completed, stats)

    # Repair single-chapter English books if under-populated
    for book in [b for b, n in ALL_BOOKS.items() if n == 1]:
        expected = _SINGLE_CHAPTER_VERSE_COUNT.get(book, 0)
        actual   = stats.get(f"{book}:en", {}).get("verses", 0)
        if actual < expected:
            tqdm.write(f"  [repair] {book} EN: {actual} verses, expected {expected} — re-queuing")
            completed.discard(f"{book}:1:en")
            stats.pop(f"{book}:en", None)
            p = _section_path(book, "en")
            if p.exists(): p.unlink()
    save_checkpoint(completed, stats)

    _download_english(completed, stats)

    for lang in WP_LANGUAGES:
        _download_wp_language(lang, completed, stats)

    print(f"\n  Merging all sections → {DATA_PATH} …")
    all_texts = merge_sections()

    print(f"\n{'═' * 60}")
    print(f"  DOWNLOAD COMPLETE")
    for lang in ["en"] + list(WP_LANGUAGES.keys()):
        count = sum(1 for t in all_texts if t["language"] == lang)
        label = {"en": "English", "si": "Sinhala", "ta": "Tamil"}.get(lang, lang.upper())
        print(f"  {label}: {count:,} verses")
    print(f"  Total:  {len(all_texts):,} entries")
    print(f"  Output: {DATA_PATH}")
    print(f"{'═' * 60}")
    return all_texts


if __name__ == "__main__":
    download_bible()