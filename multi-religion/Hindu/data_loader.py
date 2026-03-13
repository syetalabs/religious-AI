"""
Hindu Scripture Downloader
==========================
Sources:
  Bhagavad Gita  → bhagavadgita.io  (free JSON API, no key)
  Upanishads     → wisdomlib.org    (HTML, chapter-list scrape)
  Rig Veda       → wisdomlib.org    (HTML, chapter-list scrape)
  Mahabharata    → wisdomlib.org    (HTML, chapter-list scrape)
  Ramayana       → wisdomlib.org    (HTML, chapter-list scrape)
  Yoga Sutras    → wisdomlib.org    (HTML, chapter-list scrape)
  Laws of Manu   → wisdomlib.org    (HTML, chapter-list scrape)

All wisdomlib.org URLs have been verified from live search results.

Usage:
  python download_hindu.py            # normal run (resumes from checkpoint)
  python download_hindu.py --reset    # wipe checkpoint and start fresh
  python download_hindu.py --debug    # show HTTP status + HTML snippet per request
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_PATH       = DATA_DIR / "hindu_raw.json"
SECTIONS_DIR    = DATA_DIR / "sections"
CHECKPOINT_PATH = DATA_DIR / "checkpoint_hindu.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

DEBUG         = False
REQUEST_DELAY = 2.0      # seconds between requests (be polite)
TIMEOUT       = 30
MAX_RETRIES   = 5

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection":      "keep-alive",
})

# ─────────────────────────────────────────────────────────────────────────────
# Scripture canon
# ─────────────────────────────────────────────────────────────────────────────
# Types:
#   "gita_api"     → bhagavadgita.io JSON API
#   "chapter_list" → wisdomlib index page → crawl sub-pages

HINDU_SCRIPTURES: dict[str, dict] = {

    # ── Bhagavad Gita (JSON API) ──────────────────────────────────────────────
    "Bhagavad Gita": {
        "category": "Epics",
        "type":     "gita_api",
    },

    # ── Principal Upanishads (wisdomlib.org — verified URLs) ──────────────────
    # URL pattern: /hinduism/book/<full-book-slug>
    # Sub-pages:   /hinduism/book/<slug>/d/doc<id>.html   (crawled automatically)

    "Isha Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/ishavasya-upanishad-shankara-bhashya",
    },
    "Kena Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/kena-upanishad-shankara-bhashya",
    },
    "Katha Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/katha-upanishad-shankara-bhashya",
    },
    "Mundaka Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/mundaka-upanishad-shankara-bhashya",
    },
    "Mandukya Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-mandukya-upanishad",
    },
    "Taittiriya Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-taittiriya-upanishad",
    },
    "Brihadaranyaka Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-brihadaranyaka-upanishad",
    },
    "Prashna Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/prashna-upanishad-shankara-bhashya",
    },
    "Chandogya Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/chandogya-upanishad-english",
    },
    "Thirty Minor Upanishads": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/thirty-minor-upanishads",
    },

    # ── Rig Veda (wisdomlib — H.H. Wilson translation) ────────────────────────
    "Rig Veda": {
        "category": "Vedas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/rig-veda-wilson",
    },

    # ── Yoga / Philosophy ─────────────────────────────────────────────────────
    "Yoga Sutras of Patanjali": {
        "category": "Yoga / Philosophy",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-yoga-sutras-of-patanjali",
    },

    # ── Dharmashastra ─────────────────────────────────────────────────────────
    "Laws of Manu": {
        "category": "Dharmashastra",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/manusmriti-with-the-commentary-of-medhatithi",
    },

    # ── Epics ─────────────────────────────────────────────────────────────────
    "Ramayana of Valmiki": {
        "category": "Epics",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-ramayana-of-valmiki",
    },
    "Mahabharata (Ganguli)": {
        "category": "Epics",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-mahabharata-of-krishna-dwaipayana-vyasa",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _dbg(msg: str):
    if DEBUG:
        tqdm.write(f"  [debug] {msg}")


def _safe_replace(src: str, dst: str):
    src_p, dst_p = Path(src), Path(dst)
    for attempt in (1, 2, 3):
        try:
            src_p.replace(dst_p)
            return
        except PermissionError:
            pass
        try:
            if dst_p.exists():
                dst_p.unlink()
            src_p.rename(dst_p)
            return
        except OSError:
            pass
    try:
        dst_p.write_bytes(src_p.read_bytes())
        src_p.unlink(missing_ok=True)
    except Exception as exc:
        tqdm.write(f"  [warn] _safe_replace: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_html(url: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=TIMEOUT)
            _dbg(f"GET {url}  →  {resp.status_code}  ({len(resp.content):,} bytes)")

            if resp.status_code == 404:
                tqdm.write(f"    [404] {url}")
                return None
            if resp.status_code == 403:
                tqdm.write(f"    [403 Forbidden] {url}")
                return None
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 0)) or min(20 * attempt, 120)
                tqdm.write(f"    [429] rate-limited — waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            enc  = resp.apparent_encoding or resp.encoding or "utf-8"
            html = resp.content.decode(enc, errors="replace")

            if DEBUG:
                snippet = html[:400].replace("\n", " ").strip()
                tqdm.write(f"  [debug] snippet: {snippet}")

            time.sleep(REQUEST_DELAY)
            return html

        except requests.exceptions.ConnectionError as exc:
            tqdm.write(f"    [conn-error] attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            tqdm.write(f"    [timeout] attempt {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response else 0
            tqdm.write(f"    [http {code}] {url}")
            if 500 <= code < 600 and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as exc:
            tqdm.write(f"    [error] {type(exc).__name__}: {exc}")
            return None
    return None


def _get_json(url: str) -> dict | list | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=TIMEOUT)
            _dbg(f"GET {url}  →  {resp.status_code}")
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 0)) or min(15 * attempt, 60)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except Exception as exc:
            tqdm.write(f"    [json-error] attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Bhagavad Gita JSON API  (bhagavadgita.io — free, no key)
# ─────────────────────────────────────────────────────────────────────────────

_GITA_BASE     = "https://bhagavadgita.io/api/v1"
_GITA_CHAPTERS = 18


def fetch_gita_api() -> list[str]:
    verses_out: list[str] = []
    for ch in tqdm(range(1, _GITA_CHAPTERS + 1), desc="      Gita chapters", leave=False):
        data = _get_json(f"{_GITA_BASE}/chapters/{ch}/verses/")
        if not isinstance(data, list):
            tqdm.write(f"    [warn] Gita ch{ch}: no data")
            continue
        for v in data:
            num  = v.get("verse_number", "")
            text = (v.get("text") or "").strip()
            if text:
                verses_out.append(f"Chapter {ch}, Verse {num}: {text}")
    return verses_out

# ─────────────────────────────────────────────────────────────────────────────
# wisdomlib.org HTML parser
# ─────────────────────────────────────────────────────────────────────────────

_NAV_RE = re.compile(
    r"^(next|previous|index|contents?|back|forward|home|wisdomlib"
    r"|copyright|all rights|return to|buy this|let'?s grow|read more"
    r"|support|donate|patreon|newsletter|already donated)\b",
    re.IGNORECASE,
)

# wisdomlib wraps scripture text inside <div class="doc-body"> or similar
_CONTENT_SELECTORS = [
    {"class": re.compile(r"doc[-_]?body", re.I)},
    {"class": re.compile(r"content[-_]?body", re.I)},
    {"id":    re.compile(r"main[-_]?content", re.I)},
    {"class": re.compile(r"entry[-_]?content", re.I)},
]


def _extract_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup.find_all(["script", "style", "noscript", "iframe", "nav",
                               "header", "footer"]):
        tag.decompose()

    # Find content container
    container = None
    for sel in _CONTENT_SELECTORS:
        container = soup.find("div", sel)
        if container:
            break
    if not container:
        container = soup.find("body") or soup

    paragraphs: list[str] = []
    for elem in container.find_all(["p", "blockquote"]):
        text = re.sub(r"\s+", " ", elem.get_text(" ", strip=True)).strip()
        text = re.sub(r"\[p[g\.\s]*\d+\]", "", text, flags=re.IGNORECASE).strip()

        if len(text) < 40:
            continue
        if _NAV_RE.match(text):
            continue
        # Skip paragraphs that are mostly hyperlinks (nav blocks)
        link_chars = sum(len(a.get_text()) for a in elem.find_all("a"))
        if link_chars > 0.75 * len(text):
            continue

        paragraphs.append(text)

    _dbg(f"extracted {len(paragraphs)} paragraphs")
    return paragraphs


def _collect_chapter_links(html: str, base_url: str) -> list[str]:
    """
    Collect all /d/doc*.html sub-page links from a wisdomlib book index.
    wisdomlib chapter pages follow the pattern:
        /hinduism/book/<slug>/d/doc<id>.html
    """
    soup = BeautifulSoup(html, "html.parser")
    host = "wisdomlib.org"
    links: list[str] = []
    seen:  set[str]  = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "javascript:")):
            continue

        abs_url = href if href.startswith("http") else f"https://www.wisdomlib.org{href}"

        if host not in abs_url:
            continue
        # Only follow /d/doc*.html chapter pages
        if "/d/doc" not in abs_url:
            continue

        key = abs_url.lower().split("?")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        links.append(abs_url)

    _dbg(f"found {len(links)} chapter links")
    return links

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrators
# ─────────────────────────────────────────────────────────────────────────────

def fetch_chapter_list(index_url: str) -> list[str]:
    index_html = _get_html(index_url)
    if not index_html:
        tqdm.write(f"    [warn] Could not fetch index: {index_url}")
        return []

    links = _collect_chapter_links(index_html, index_url)

    if not links:
        tqdm.write(f"    [info] No /d/doc sub-links found — using index page text")
        return _extract_paragraphs(index_html)

    all_paras: list[str] = []
    for link in tqdm(links, desc="      pages", leave=False, unit="pg"):
        html = _get_html(link)
        if html:
            all_paras.extend(_extract_paragraphs(html))
    return all_paras

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint & section I/O
# ─────────────────────────────────────────────────────────────────────────────

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


def _section_path(scripture: str) -> Path:
    safe = re.sub(r"[^\w]", "_", scripture)
    return SECTIONS_DIR / f"hindu_{safe}.json"


def save_section(scripture: str, paragraphs: list[str]):
    p   = _section_path(scripture)
    tmp = str(p) + ".tmp"
    Path(tmp).write_text(
        json.dumps(paragraphs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(p))


def merge_sections() -> list[dict]:
    all_texts: list[dict] = []
    for scripture, meta in HINDU_SCRIPTURES.items():
        p = _section_path(scripture)
        if not p.exists():
            continue
        try:
            paragraphs = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for para in paragraphs:
            all_texts.append({
                "text":     para,
                "section":  scripture,
                "category": meta.get("category", "Unknown"),
                "religion": "Hinduism",
                "language": "en",
            })

    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps(all_texts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(DATA_PATH))
    return all_texts

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def download_hindu() -> list[dict]:
    completed, stats = load_checkpoint()

    print(f"\n{'═' * 60}")
    print("  Downloading Hindu Scriptures")
    print(f"  Bhagavad Gita  → bhagavadgita.io  (JSON API)")
    print(f"  All others     → wisdomlib.org     (HTML scrape)")
    print(f"{'═' * 60}\n")

    for scripture, meta in tqdm(HINDU_SCRIPTURES.items(), desc="Scriptures", unit="text"):
        if scripture in completed:
            n = stats.get(scripture, {}).get("paragraphs", 0)
            tqdm.write(f"  ✓ {scripture} — already done ({n} items)")
            continue

        tqdm.write(f"\n── {scripture}  [{meta['category']}]")

        paragraphs: list[str] = []
        try:
            if meta["type"] == "gita_api":
                paragraphs = fetch_gita_api()
            elif meta["type"] == "chapter_list":
                paragraphs = fetch_chapter_list(meta["url"])
            else:
                tqdm.write(f"    [warn] Unknown type '{meta['type']}'")
        except Exception as exc:
            tqdm.write(f"    [error] {type(exc).__name__}: {exc}")

        if paragraphs:
            save_section(scripture, paragraphs)
            tqdm.write(f"  → {len(paragraphs)} items saved")
        else:
            tqdm.write(
                f"  [warn] 0 items retrieved for: {scripture}\n"
                f"         Run with --debug to inspect HTTP status and HTML."
            )

        completed.add(scripture)
        stats[scripture] = {"paragraphs": len(paragraphs)}
        save_checkpoint(completed, stats)

    print(f"\n  Merging all sections → {DATA_PATH} …")
    all_texts = merge_sections()

    print(f"\n{'═' * 60}")
    print("  DOWNLOAD COMPLETE")
    print(f"  Total entries : {len(all_texts):,}")
    by_cat: dict[str, int] = {}
    for e in all_texts:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"    {cat:<34} {count:>6,}")
    print(f"  Output : {DATA_PATH}")
    print(f"{'═' * 60}")
    return all_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hindu scripture data")
    parser.add_argument("--debug", action="store_true",
                        help="Print HTTP status + HTML snippet per request")
    parser.add_argument("--reset", action="store_true",
                        help="Clear checkpoint and section files, re-download all")
    args = parser.parse_args()

    if args.debug:
        DEBUG = True
        print("  [debug mode ON]")

    if args.reset:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        for p in SECTIONS_DIR.glob("hindu_*.json"):
            p.unlink()
        print("  [reset] Checkpoint and section files cleared.")

    download_hindu()