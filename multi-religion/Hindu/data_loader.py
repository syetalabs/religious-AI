"""
Hindu Scripture Downloader  (data_loader.py)
=============================================
Sources:
  Bhagavad Gita  -> GitHub (praneshp1org/Bhagavad-Gita-JSON-data)
  Rig Veda       -> Local section file only (run patch_rigveda.py separately)
  All others     -> wisdomlib.org (HTML scrape, /d/doc links)

Usage:
  python data_loader.py              # resume from checkpoint
  python data_loader.py --reset      # wipe checkpoint, re-download all
  python data_loader.py --patch      # only re-run entries with 0 items
  python data_loader.py --force "X"  # force re-download one scripture
  python data_loader.py --clean      # strip Hindi/Sanskrit from hindu_raw.json
  python data_loader.py --remerge    # re-merge all section files into hindu_raw.json
  python data_loader.py --debug      # show HTTP status + HTML per request

Note: Rig Veda is NOT downloaded by this script. Run patch_rigveda.py
      once to generate data/sections/hindu_Rig_Veda.json, then use
      --remerge to include it in hindu_raw.json.
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
REQUEST_DELAY = 2.0
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
# Language detection
# ─────────────────────────────────────────────────────────────────────────────

_NON_ENGLISH_RANGES = [
    (0x0900, 0x097F),   # Devanagari
    (0x0980, 0x09FF),   # Bengali
    (0x0A00, 0x0A7F),   # Gurmukhi
    (0x0A80, 0x0AFF),   # Gujarati
    (0x0B00, 0x0B7F),   # Oriya
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0C80, 0x0CFF),   # Kannada
    (0x0D00, 0x0D7F),   # Malayalam
]

def _non_english_ratio(text: str) -> float:
    if not text:
        return 0.0
    count = sum(
        1 for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in _NON_ENGLISH_RANGES)
    )
    return count / len(text)

def is_english(text: str, threshold: float = 0.05) -> bool:
    return _non_english_ratio(text) < threshold

# ─────────────────────────────────────────────────────────────────────────────
# Hindu scripture canon
#
# "Rig Veda" is intentionally NOT listed here.
# It is fetched separately via patch_rigveda.py which saves:
#   data/sections/hindu_Rig_Veda.json
# merge_sections() always picks up that file automatically.
# ─────────────────────────────────────────────────────────────────────────────

HINDU_SCRIPTURES: dict[str, dict] = {

    # Bhagavad Gita (GitHub JSON)
    "Bhagavad Gita": {
        "category": "Epics",
        "type":     "gita_github",
    },

    # Upanishads
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

    # Yoga / Philosophy
    "Yoga Sutras of Patanjali": {
        "category": "Yoga / Philosophy",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/yoga-sutras-study",
    },

    # Dharmashastra
    "Laws of Manu": {
        "category": "Dharmashastra",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/manusmriti-with-the-commentary-of-medhatithi",
    },

    # Epics
    "Ramayana of Valmiki": {
        "category": "Epics",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-ramayana-of-valmiki",
    },
    "Mahabharata (English)": {
        "category": "Epics",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-mahabharata-mohan",
    },
    "Mahabharata (Summary)": {
        "category": "Epics",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/mahabharata-english-summary",
    },

    # ── Puranas (Vaishnavism) ─────────────────────────────────────────────────
    # The Bhagavata Purana is the most important Purana for devotion to Vishnu/Krishna.
    # It contains the Bhagavata philosophy, the life of Krishna (Book 10), and
    # core teachings on bhakti, cosmology, and liberation.
    "Bhagavata Purana": {
        "category": "Puranas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/the-bhagavata-purana",
    },
    # The Vishnu Purana (Wilson translation) covers Vishnu cosmology,
    # the ten avatars, duties of the four varnas, and the Yuga cycles.
    "Vishnu Purana": {
        "category": "Puranas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/vishnu-purana-wilson",
    },

    # ── Puranas (Shaivism) ────────────────────────────────────────────────────
    # The Shiva Purana covers Shiva worship, the Shaiva philosophy,
    # Shiva's forms, festivals, and the nature of liberation through Shiva.
    "Shiva Purana": {
        "category": "Puranas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/shiva-purana-english",
    },
    # The Linga Purana focuses on the Shiva Linga as a symbol of the divine,
    # containing cosmological accounts and Shaiva rituals.
    "Linga Purana": {
        "category": "Puranas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/linga-purana-english",
    },

    # ── Puranas (Shaktism) ────────────────────────────────────────────────────
    # The Devi Bhagavata Purana is the central text of Shaktism —
    # worship of the Goddess (Devi/Shakti) as the supreme reality.
    "Devi Bhagavata Purana": {
        "category": "Puranas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/devi-bhagavata-purana",
    },
    # The Markandeya Purana contains the Devi Mahatmya (Durga Saptashati),
    # the most widely recited Shakta scripture glorifying the Goddess.
    "Markandeya Purana": {
        "category": "Puranas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/markandeya-purana-english",
    },

    # ── Puranas (Garuda / Afterlife) ──────────────────────────────────────────
    # The Garuda Purana covers the afterlife, death rituals, karma,
    # and the journey of the soul — commonly referenced for death-related questions.
    "Garuda Purana": {
        "category": "Puranas",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/garuda-purana-saroddhara",
    },

    # ── Additional Upanishads (replacements for thin entries) ─────────────────
    # The Katha Upanishad previously returned only 40 entries due to a URL issue.
    # This uses the full English translation edition.
    "Katha Upanishad (Gambhirananda)": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/katha-upanishad-gambhirananda",
    },
    # The Svetasvatara Upanishad covers personal theism, the nature of Brahman,
    # devotion, and liberation — important for theistic Vedanta.
    "Svetasvatara Upanishad": {
        "category": "Upanishads",
        "type":     "chapter_list",
        "url":      "https://www.wisdomlib.org/hinduism/book/shvetashvatara-upanishad-gambhirananda",
    },
}

# Rig Veda metadata — used only by merge_sections() to tag its entries.
# The actual data comes from the section file written by patch_rigveda.py.
_RIG_VEDA_META = {"category": "Vedas", "section": "Rig Veda"}

# ─────────────────────────────────────────────────────────────────────────────
# Bhagavad Gita — GitHub raw JSON
# ─────────────────────────────────────────────────────────────────────────────

_GITA_VERSE_URL = (
    "https://raw.githubusercontent.com/praneshp1org/"
    "Bhagavad-Gita-JSON-data/main/verse.json"
)
_GITA_TRANS_URL = (
    "https://raw.githubusercontent.com/praneshp1org/"
    "Bhagavad-Gita-JSON-data/main/translation.json"
)
_PREFERRED_AUTHORS = {"swami sivananda", "dr. s. sankaranarayan", "shri purohit swami"}


def _fetch_raw_json(url: str, label: str) -> list | None:
    tqdm.write(f"    Fetching {label} ...")
    for attempt in range(1, 4):
        try:
            resp = _session.get(url, timeout=TIMEOUT)
            tqdm.write(f"      HTTP {resp.status_code}  ({len(resp.content):,} bytes)")
            if resp.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            text = resp.content.decode("utf-8-sig")
            data = json.loads(text)
            tqdm.write(f"      {len(data)} records")
            return data
        except Exception as exc:
            tqdm.write(f"      Error attempt {attempt}/3: {exc}")
            time.sleep(2 ** attempt)
    return None


def fetch_gita_github() -> list[str]:
    verse_data = _fetch_raw_json(_GITA_VERSE_URL, "verse.json")
    if not verse_data:
        tqdm.write("    [warn] Could not fetch verse.json")
        return []

    trans_data = _fetch_raw_json(_GITA_TRANS_URL, "translation.json")
    if not trans_data:
        tqdm.write("    [warn] Could not fetch translation.json")
        return []

    meta: dict[str, tuple] = {}
    for v in verse_data:
        vid = str(v.get("verse_id") or v.get("id", "")).strip()
        ch  = v.get("chapter_number") or v.get("chapter_id")
        vn  = v.get("verse_number")
        if vid and ch is not None and vn is not None:
            meta[vid] = (int(ch), int(vn))

    best: dict[str, str] = {}
    for t in trans_data:
        vid  = str(t.get("verse_id") or t.get("id", "")).strip()
        lang = str(t.get("lang") or t.get("language", "")).lower()
        desc = str(t.get("description") or t.get("translation") or "").strip()

        if not vid or not desc:
            continue
        if lang and "english" not in lang and lang not in ("en", "1", ""):
            continue
        if not is_english(desc):
            continue

        if vid not in best:
            best[vid] = desc
        elif str(t.get("authorName", t.get("author_name", ""))).lower() in _PREFERRED_AUTHORS:
            best[vid] = desc

    tqdm.write(f"    English translations matched: {len(best)}")

    rows: list[tuple] = []
    for vid, text in best.items():
        if vid in meta:
            ch, vn = meta[vid]
            rows.append((ch, vn, text))

    rows.sort(key=lambda x: (x[0], x[1]))
    verses = [f"Chapter {ch}, Verse {vn}: {text}" for ch, vn, text in rows]
    tqdm.write(f"    {len(verses)} verses assembled")
    return verses

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _dbg(msg: str):
    if DEBUG:
        tqdm.write(f"  [debug] {msg}")


def _safe_replace(src: str, dst: str):
    src_p, dst_p = Path(src), Path(dst)
    try:
        src_p.replace(dst_p); return
    except PermissionError:
        pass
    try:
        if dst_p.exists():
            dst_p.unlink()
        src_p.rename(dst_p); return
    except OSError:
        pass
    dst_p.write_bytes(src_p.read_bytes())
    src_p.unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_html(url: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=TIMEOUT)
            _dbg(f"GET {url}  ->  {resp.status_code}  ({len(resp.content):,} B)")

            if resp.status_code == 404:
                tqdm.write(f"    [404] {url}")
                return None
            if resp.status_code == 403:
                tqdm.write(f"    [403] {url}")
                return None
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 0)) or min(20 * attempt, 120)
                tqdm.write(f"    [429] waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            enc  = resp.apparent_encoding or resp.encoding or "utf-8"
            html = resp.content.decode(enc, errors="replace")

            if DEBUG:
                tqdm.write(f"  [debug] snippet: {html[:400].replace(chr(10), ' ')}")

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

# ─────────────────────────────────────────────────────────────────────────────
# wisdomlib HTML parser
# ─────────────────────────────────────────────────────────────────────────────

_NAV_RE = re.compile(
    r"^(next|previous|index|contents?|back|forward|home|wisdomlib"
    r"|copyright|all rights|return to|buy (this|now)|let'?s grow"
    r"|support|donate|patreon|newsletter|already donated|read more)\b",
    re.IGNORECASE,
)

_BOILERPLATE = {
    "Disclaimer: These are translations of Sanskrit texts and are not necessarily "
    "approved by everyone associated with the traditions connected to these texts. "
    "Consult the source and original scripture in case of doubt.",
    "I humbly request your help to keep doing what I do best: provide the world with "
    "unbiased sources, definitions and images. Your donation direclty influences the "
    "quality and quantity of knowledge, wisdom and spiritual insight the world is "
    "exposed to.",
    "Let's make the world a better place together!",
}

_BOILERPLATE_FRAGMENTS = [
    "humbly request your help to keep doing what i do best",
    "your donation direclty influences the quality",
    "isbn-10:",
    "isbn-13:",
    "words | isbn",
]

_CONTENT_SELECTORS = [
    {"class": re.compile(r"doc[-_]?body",      re.I)},
    {"class": re.compile(r"content[-_]?body",  re.I)},
    {"id":    re.compile(r"main[-_]?content",  re.I)},
    {"class": re.compile(r"entry[-_]?content", re.I)},
]


def _is_boilerplate(text: str) -> bool:
    if text in _BOILERPLATE:
        return True
    tl = text.lower()
    return any(frag.lower() in tl for frag in _BOILERPLATE_FRAGMENTS)


def _extract_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "noscript", "iframe",
                               "nav", "header", "footer"]):
        tag.decompose()

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
        if _is_boilerplate(text):
            continue
        link_chars = sum(len(a.get_text()) for a in elem.find_all("a"))
        if link_chars > 0.75 * len(text):
            continue
        if not is_english(text):
            continue

        paragraphs.append(text)

    _dbg(f"extracted {len(paragraphs)} paragraphs")
    return paragraphs


def _collect_chapter_links(html: str, base_url: str) -> list[str]:
    soup  = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen:  set[str]  = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "javascript:")):
            continue

        abs_url = (href if href.startswith("http")
                   else f"https://www.wisdomlib.org{href}")

        if "wisdomlib.org" not in abs_url:
            continue
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
        tqdm.write(f"    [warn] Could not fetch: {index_url}")
        return []

    links = _collect_chapter_links(index_html, index_url)

    if not links:
        tqdm.write("    [info] No /d/doc sub-links found -- using index page text")
        return _extract_paragraphs(index_html)

    all_paras: list[str] = []
    seen_paras: set[str] = set()

    for link in tqdm(links, desc="      pages", leave=False, unit="pg"):
        html = _get_html(link)
        if not html:
            continue
        for para in _extract_paragraphs(html):
            if para not in seen_paras:
                seen_paras.add(para)
                all_paras.append(para)

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
    """
    Merge all section files into hindu_raw.json.
    Includes HINDU_SCRIPTURES entries + the Rig Veda section file
    (written by patch_rigveda.py) if it exists.
    """
    all_texts: list[dict] = []

    # Build the full list of (scripture_name, category) pairs to merge.
    # HINDU_SCRIPTURES covers everything except Rig Veda.
    # Rig Veda is added here so it gets picked up from its section file.
    merge_list: list[tuple[str, str]] = [
        (name, meta["category"]) for name, meta in HINDU_SCRIPTURES.items()
    ]
    merge_list.append(("Rig Veda", "Vedas"))   # always include even if not in canon

    for scripture, category in merge_list:
        p = _section_path(scripture)
        if not p.exists():
            continue
        try:
            paragraphs = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        seen_in_section: set[str] = set()
        for para in paragraphs:
            if not is_english(para):
                continue
            if _is_boilerplate(para):
                continue
            if para in seen_in_section:
                continue
            seen_in_section.add(para)
            all_texts.append({
                "text":     para,
                "section":  scripture,
                "category": category,
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
# --clean
# ─────────────────────────────────────────────────────────────────────────────

def clean_existing_raw():
    if not DATA_PATH.exists():
        print("  hindu_raw.json not found -- nothing to clean.")
        return

    data: list[dict] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    before = len(data)

    kept    = [e for e in data
               if is_english(e.get("text", ""))
               and not _is_boilerplate(e.get("text", ""))]
    removed = before - len(kept)

    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps(kept, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(DATA_PATH))

    print(f"  Cleaned hindu_raw.json:")
    print(f"    Before : {before:,} entries")
    print(f"    Removed: {removed:,} non-English entries")
    print(f"    After  : {len(kept):,} entries")

    cleaned_sections = 0
    for p in SECTIONS_DIR.glob("hindu_*.json"):
        try:
            paras: list[str] = json.loads(p.read_text(encoding="utf-8"))
            clean = [t for t in paras if is_english(t) and not _is_boilerplate(t)]
            if len(clean) < len(paras):
                tmp = str(p) + ".tmp"
                Path(tmp).write_text(
                    json.dumps(clean, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                _safe_replace(tmp, str(p))
                cleaned_sections += 1
        except Exception:
            pass

    if cleaned_sections:
        print(f"    Also cleaned {cleaned_sections} section file(s)")

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def download_hindu() -> list[dict]:
    completed, stats = load_checkpoint()

    # Check for Rig Veda section file and report its status
    rv_path = _section_path("Rig Veda")
    if rv_path.exists():
        try:
            rv_count = len(json.loads(rv_path.read_text(encoding="utf-8")))
            print(f"  [info] Rig Veda section file found: {rv_count:,} entries")
        except Exception:
            print(f"  [info] Rig Veda section file found but unreadable")
    else:
        print(f"  [info] Rig Veda section file not found.")
        print(f"         Run 'python patch_rigveda.py' to generate it.")

    print(f"\n{'=' * 60}")
    print("  Downloading Hindu Scriptures")
    print(f"  Bhagavad Gita  -> GitHub raw JSON (praneshp1org)")
    print(f"  All others     -> wisdomlib.org   (HTML scrape)")
    print(f"  Rig Veda       -> Local file only (patch_rigveda.py)")
    print(f"{'=' * 60}\n")

    for scripture, meta in tqdm(HINDU_SCRIPTURES.items(),
                                desc="Scriptures", unit="text"):
        if scripture in completed:
            n = stats.get(scripture, {}).get("paragraphs", 0)
            tqdm.write(f"  [skip] {scripture} -- already done ({n} items)")
            continue

        tqdm.write(f"\n-- {scripture}  [{meta['category']}]")

        paragraphs: list[str] = []
        try:
            if meta["type"] == "gita_github":
                paragraphs = fetch_gita_github()
            elif meta["type"] == "chapter_list":
                paragraphs = fetch_chapter_list(meta["url"])
            else:
                tqdm.write(f"    [warn] Unknown type '{meta['type']}'")
        except Exception as exc:
            tqdm.write(f"    [error] {type(exc).__name__}: {exc}")

        if paragraphs:
            save_section(scripture, paragraphs)
            tqdm.write(f"  -> {len(paragraphs)} items saved")
        else:
            tqdm.write(f"  [warn] 0 items for: {scripture}  (run --debug to diagnose)")

        completed.add(scripture)
        stats[scripture] = {"paragraphs": len(paragraphs)}
        save_checkpoint(completed, stats)

    print(f"\n  Merging all sections -> {DATA_PATH} ...")
    all_texts = merge_sections()

    print(f"\n{'=' * 60}")
    print("  DOWNLOAD COMPLETE")
    print(f"  Total entries : {len(all_texts):,}")
    by_cat: dict[str, int] = {}
    for e in all_texts:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"    {cat:<34} {count:>7,}")
    print(f"  Output : {DATA_PATH}")
    print(f"{'=' * 60}")
    return all_texts

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hindu scripture data")
    parser.add_argument("--debug", action="store_true",
                        help="Print HTTP status + HTML snippet per request")
    parser.add_argument("--reset", action="store_true",
                        help="Clear ALL checkpoint + section files, re-download everything")
    parser.add_argument("--patch", action="store_true",
                        help="Re-run only entries that previously returned 0 items")
    parser.add_argument("--force", metavar="SCRIPTURE",
                        help='Force re-download one scripture, e.g. --force "Bhagavad Gita"')
    parser.add_argument("--clean", action="store_true",
                        help="Strip Hindi/Sanskrit entries from existing hindu_raw.json")
    parser.add_argument("--remerge", action="store_true",
                        help="Re-merge all section files into hindu_raw.json (no download)")
    args = parser.parse_args()

    if args.debug:
        DEBUG = True
        print("  [debug mode ON]")

    if args.clean:
        clean_existing_raw()
        raise SystemExit(0)

    if args.remerge:
        print("  Re-merging all section files...")
        all_texts = merge_sections()
        print(f"  Done. {len(all_texts):,} total entries -> {DATA_PATH}")
        by_cat: dict[str, int] = {}
        for e in all_texts:
            by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
        for cat, count in sorted(by_cat.items()):
            print(f"    {cat:<34} {count:>7,}")
        raise SystemExit(0)

    if args.reset:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        for p in SECTIONS_DIR.glob("hindu_*.json"):
            p.unlink()
        print("  [reset] Checkpoint and all section files cleared.")

    elif args.force:
        name = args.force
        if name == "Rig Veda":
            print("  [info] Rig Veda is managed by patch_rigveda.py, not this script.")
            print("         Run: python patch_rigveda.py")
            print("         Then: python data_loader.py --remerge")
            raise SystemExit(0)
        if name not in HINDU_SCRIPTURES:
            close = [k for k in HINDU_SCRIPTURES if name.lower() in k.lower()]
            if close:
                print(f"  [force] Did you mean: {close}")
            else:
                print(f"  [force] Not found. Available scriptures:")
                for k in HINDU_SCRIPTURES:
                    print(f"    - {k}")
                print(f"    (Rig Veda is managed separately via patch_rigveda.py)")
            raise SystemExit(1)
        completed, stats = load_checkpoint()
        completed.discard(name)
        stats.pop(name, None)
        sec = re.sub(r"[^\w]", "_", name)
        fp  = SECTIONS_DIR / f"hindu_{sec}.json"
        if fp.exists():
            fp.unlink()
        save_checkpoint(completed, stats)
        print(f"  [force] Queued '{name}' for re-download.")

    elif args.patch:
        completed, stats = load_checkpoint()
        patched = [k for k, v in stats.items()
                   if isinstance(v, dict) and v.get("paragraphs", 0) == 0
                   and k != "Rig Veda"]  # never auto-patch Rig Veda
        for k in patched:
            completed.discard(k)
            stats.pop(k, None)
            sec = re.sub(r"[^\w]", "_", k)
            fp  = SECTIONS_DIR / f"hindu_{sec}.json"
            if fp.exists():
                fp.unlink()
        save_checkpoint(completed, stats)
        if patched:
            print(f"  [patch] Queued {len(patched)} failed entries for retry:")
            for k in patched:
                print(f"    - {k}")
        else:
            print("  [patch] Nothing to patch -- all entries already have data.")

    download_hindu()