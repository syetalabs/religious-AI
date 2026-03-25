"""
Islam Data Loader  —  Comprehensive Islamic Knowledge Base
==========================================================
Downloads and merges multiple Islamic text sources into quran_raw.json.

Sources included:
  1. Quran           — 6,236 verses (Sahih International, alquran.cloud)
                       + Ibn Kathir tafsir per verse (spa5k CDN)
  2. Sahih Bukhari   — ~7,563 hadiths  (fawazahmed0 CDN, eng-bukhari)
  3. Sahih Muslim    — ~3,033 hadiths  (fawazahmed0 CDN, eng-muslim)
  4. 40 Hadith Nawawi— 42 hadiths      (fawazahmed0 CDN, eng-nawawi40)
  5. Sunan Abu Dawud — ~5,274 hadiths  (fawazahmed0 CDN, eng-abudawud)
  6. Riyad as-Salihin— ~1,900 hadiths  (fawazahmed0 CDN, eng-riyadussalihin)

All sources are free, no API key required.

Expected output size : ~35,000-50,000 KB  (comparable to other religions)
Expected entry count : ~24,000+ passages

Output:
  data/quran_raw.json              — all entries (Quran + Hadith) for chunk_and_embed
  data/sections/islam_*.json       — per-surah Quran section files (checkpoint)
  data/hadith/islam_hadith_*.json  — per-collection hadith files (checkpoint)
  data/checkpoint_islam.json       — download progress checkpoint

Usage:
  python data_loader.py              # download all (resumes from checkpoint)
  python data_loader.py --reset      # wipe everything, re-download all
  python data_loader.py --remerge    # re-merge existing files, no download
  python data_loader.py --patch      # retry surahs that returned 0 verses
  python data_loader.py --force 2    # force re-download one surah by number
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_PATH       = DATA_DIR / "quran_raw.json"
SECTIONS_DIR    = DATA_DIR / "sections"
HADITH_DIR      = DATA_DIR / "hadith"
CHECKPOINT_PATH = DATA_DIR / "checkpoint_islam.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
HADITH_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_DELAY = 0.5
TIMEOUT       = 60
MAX_RETRIES   = 3
GROUP_SIZE    = 1     # 1 verse per passage — chunk_and_embed handles context window

# ─────────────────────────────────────────────────────────────────────────────
# HTTP session
# ─────────────────────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})

# ─────────────────────────────────────────────────────────────────────────────
# Source URLs
# ─────────────────────────────────────────────────────────────────────────────

# Quran translation
_ALQURAN_CLOUD_URL = "https://api.alquran.cloud/v1/surah/{n}/en.sahih"
_SEMARKETIR_URL    = "https://raw.githubusercontent.com/semarketir/quranjson/master/source/translation/en/en_translation_{n}.json"
_QURANAPI_URL      = "https://quranapi.pages.dev/api/{n}.json"

# Quran tafsir — Ibn Kathir English
_TAFSIR_CDN_URL       = "https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/en-tafisr-ibn-kathir/{surah}/{ayah}.json"
_QURAN_COM_TAFSIR_URL = "https://api.quran.com/api/v4/tafsirs/169/by_ayah/{surah}:{ayah}"

# Hadith — fawazahmed0 CDN (entire collection in one bulk JSON, no API key)
_HADITH_BASE = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{edition}.min.json"

# Collections: (edition_id, display_name, book_key)
HADITH_COLLECTIONS = [
    ("eng-bukhari",  "Sahih al-Bukhari",  "bukhari"),
    ("eng-muslim",   "Sahih Muslim",      "muslim"),
    ("eng-abudawud", "Sunan Abu Dawud",   "abudawud"),
    ("eng-ibnmajah", "Sunan Ibn Majah",   "ibnmajah"),   # replaces nawawi40 (not on CDN)
    ("eng-malik",    "Muwatta Malik",     "malik"),       # replaces riyadussalihin (not on CDN)
]

# ─────────────────────────────────────────────────────────────────────────────
# Surah metadata (all 114)
# ─────────────────────────────────────────────────────────────────────────────

SURAH_INFO: dict[int, dict] = {
    1:   {"name": "Al-Fatihah",       "revelation": "Meccan"},
    2:   {"name": "Al-Baqarah",       "revelation": "Medinan"},
    3:   {"name": "Ali 'Imran",       "revelation": "Medinan"},
    4:   {"name": "An-Nisa",          "revelation": "Medinan"},
    5:   {"name": "Al-Ma'idah",       "revelation": "Medinan"},
    6:   {"name": "Al-An'am",         "revelation": "Meccan"},
    7:   {"name": "Al-A'raf",         "revelation": "Meccan"},
    8:   {"name": "Al-Anfal",         "revelation": "Medinan"},
    9:   {"name": "At-Tawbah",        "revelation": "Medinan"},
    10:  {"name": "Yunus",            "revelation": "Meccan"},
    11:  {"name": "Hud",              "revelation": "Meccan"},
    12:  {"name": "Yusuf",            "revelation": "Meccan"},
    13:  {"name": "Ar-Ra'd",          "revelation": "Medinan"},
    14:  {"name": "Ibrahim",          "revelation": "Meccan"},
    15:  {"name": "Al-Hijr",          "revelation": "Meccan"},
    16:  {"name": "An-Nahl",          "revelation": "Meccan"},
    17:  {"name": "Al-Isra",          "revelation": "Meccan"},
    18:  {"name": "Al-Kahf",          "revelation": "Meccan"},
    19:  {"name": "Maryam",           "revelation": "Meccan"},
    20:  {"name": "Ta-Ha",            "revelation": "Meccan"},
    21:  {"name": "Al-Anbiya",        "revelation": "Meccan"},
    22:  {"name": "Al-Hajj",          "revelation": "Medinan"},
    23:  {"name": "Al-Mu'minun",      "revelation": "Meccan"},
    24:  {"name": "An-Nur",           "revelation": "Medinan"},
    25:  {"name": "Al-Furqan",        "revelation": "Meccan"},
    26:  {"name": "Ash-Shu'ara",      "revelation": "Meccan"},
    27:  {"name": "An-Naml",          "revelation": "Meccan"},
    28:  {"name": "Al-Qasas",         "revelation": "Meccan"},
    29:  {"name": "Al-'Ankabut",      "revelation": "Meccan"},
    30:  {"name": "Ar-Rum",           "revelation": "Meccan"},
    31:  {"name": "Luqman",           "revelation": "Meccan"},
    32:  {"name": "As-Sajdah",        "revelation": "Meccan"},
    33:  {"name": "Al-Ahzab",         "revelation": "Medinan"},
    34:  {"name": "Saba",             "revelation": "Meccan"},
    35:  {"name": "Fatir",            "revelation": "Meccan"},
    36:  {"name": "Ya-Sin",           "revelation": "Meccan"},
    37:  {"name": "As-Saffat",        "revelation": "Meccan"},
    38:  {"name": "Sad",              "revelation": "Meccan"},
    39:  {"name": "Az-Zumar",         "revelation": "Meccan"},
    40:  {"name": "Ghafir",           "revelation": "Meccan"},
    41:  {"name": "Fussilat",         "revelation": "Meccan"},
    42:  {"name": "Ash-Shura",        "revelation": "Meccan"},
    43:  {"name": "Az-Zukhruf",       "revelation": "Meccan"},
    44:  {"name": "Ad-Dukhan",        "revelation": "Meccan"},
    45:  {"name": "Al-Jathiyah",      "revelation": "Meccan"},
    46:  {"name": "Al-Ahqaf",         "revelation": "Meccan"},
    47:  {"name": "Muhammad",         "revelation": "Medinan"},
    48:  {"name": "Al-Fath",          "revelation": "Medinan"},
    49:  {"name": "Al-Hujurat",       "revelation": "Medinan"},
    50:  {"name": "Qaf",              "revelation": "Meccan"},
    51:  {"name": "Adh-Dhariyat",     "revelation": "Meccan"},
    52:  {"name": "At-Tur",           "revelation": "Meccan"},
    53:  {"name": "An-Najm",          "revelation": "Meccan"},
    54:  {"name": "Al-Qamar",         "revelation": "Meccan"},
    55:  {"name": "Ar-Rahman",        "revelation": "Medinan"},
    56:  {"name": "Al-Waqi'ah",       "revelation": "Meccan"},
    57:  {"name": "Al-Hadid",         "revelation": "Medinan"},
    58:  {"name": "Al-Mujadila",      "revelation": "Medinan"},
    59:  {"name": "Al-Hashr",         "revelation": "Medinan"},
    60:  {"name": "Al-Mumtahanah",    "revelation": "Medinan"},
    61:  {"name": "As-Saf",           "revelation": "Medinan"},
    62:  {"name": "Al-Jumu'ah",       "revelation": "Medinan"},
    63:  {"name": "Al-Munafiqun",     "revelation": "Medinan"},
    64:  {"name": "At-Taghabun",      "revelation": "Medinan"},
    65:  {"name": "At-Talaq",         "revelation": "Medinan"},
    66:  {"name": "At-Tahrim",        "revelation": "Medinan"},
    67:  {"name": "Al-Mulk",          "revelation": "Meccan"},
    68:  {"name": "Al-Qalam",         "revelation": "Meccan"},
    69:  {"name": "Al-Haqqah",        "revelation": "Meccan"},
    70:  {"name": "Al-Ma'arij",       "revelation": "Meccan"},
    71:  {"name": "Nuh",              "revelation": "Meccan"},
    72:  {"name": "Al-Jinn",          "revelation": "Meccan"},
    73:  {"name": "Al-Muzzammil",     "revelation": "Meccan"},
    74:  {"name": "Al-Muddaththir",   "revelation": "Meccan"},
    75:  {"name": "Al-Qiyamah",       "revelation": "Meccan"},
    76:  {"name": "Al-Insan",         "revelation": "Medinan"},
    77:  {"name": "Al-Mursalat",      "revelation": "Meccan"},
    78:  {"name": "An-Naba",          "revelation": "Meccan"},
    79:  {"name": "An-Nazi'at",       "revelation": "Meccan"},
    80:  {"name": "Abasa",            "revelation": "Meccan"},
    81:  {"name": "At-Takwir",        "revelation": "Meccan"},
    82:  {"name": "Al-Infitar",       "revelation": "Meccan"},
    83:  {"name": "Al-Mutaffifin",    "revelation": "Meccan"},
    84:  {"name": "Al-Inshiqaq",      "revelation": "Meccan"},
    85:  {"name": "Al-Buruj",         "revelation": "Meccan"},
    86:  {"name": "At-Tariq",         "revelation": "Meccan"},
    87:  {"name": "Al-A'la",          "revelation": "Meccan"},
    88:  {"name": "Al-Ghashiyah",     "revelation": "Meccan"},
    89:  {"name": "Al-Fajr",          "revelation": "Meccan"},
    90:  {"name": "Al-Balad",         "revelation": "Meccan"},
    91:  {"name": "Ash-Shams",        "revelation": "Meccan"},
    92:  {"name": "Al-Layl",          "revelation": "Meccan"},
    93:  {"name": "Ad-Duha",          "revelation": "Meccan"},
    94:  {"name": "Ash-Sharh",        "revelation": "Meccan"},
    95:  {"name": "At-Tin",           "revelation": "Meccan"},
    96:  {"name": "Al-'Alaq",         "revelation": "Meccan"},
    97:  {"name": "Al-Qadr",          "revelation": "Meccan"},
    98:  {"name": "Al-Bayyinah",      "revelation": "Medinan"},
    99:  {"name": "Az-Zalzalah",      "revelation": "Medinan"},
    100: {"name": "Al-'Adiyat",       "revelation": "Meccan"},
    101: {"name": "Al-Qari'ah",       "revelation": "Meccan"},
    102: {"name": "At-Takathur",      "revelation": "Meccan"},
    103: {"name": "Al-'Asr",          "revelation": "Meccan"},
    104: {"name": "Al-Humazah",       "revelation": "Meccan"},
    105: {"name": "Al-Fil",           "revelation": "Meccan"},
    106: {"name": "Quraysh",          "revelation": "Meccan"},
    107: {"name": "Al-Ma'un",         "revelation": "Meccan"},
    108: {"name": "Al-Kawthar",       "revelation": "Meccan"},
    109: {"name": "Al-Kafirun",       "revelation": "Meccan"},
    110: {"name": "An-Nasr",          "revelation": "Medinan"},
    111: {"name": "Al-Masad",         "revelation": "Meccan"},
    112: {"name": "Al-Ikhlas",        "revelation": "Meccan"},
    113: {"name": "Al-Falaq",         "revelation": "Meccan"},
    114: {"name": "An-Nas",           "revelation": "Meccan"},
}

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: int = TIMEOUT) -> dict | list | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=timeout)
            if resp.status_code == 200:
                time.sleep(REQUEST_DELAY)
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 0)) or min(20 * attempt, 60)
                print(f"    [429] waiting {wait}s ...")
                time.sleep(wait)
                continue
            if resp.status_code in (403, 404):
                return None
            time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError:
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
        except Exception as exc:
            print(f"    [error] {type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Quran — per-source parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_semarketir(data: dict) -> list[dict]:
    verses = []
    for k, v in data.items():
        try:
            ayah = int(k)
            text = str(v.get("text", "")).strip()
            if text:
                verses.append({"ayah": ayah, "text": text})
        except (ValueError, TypeError):
            continue
    verses.sort(key=lambda x: x["ayah"])
    return verses


def _parse_alquran_cloud(data: dict) -> list[dict]:
    ayahs = (data.get("data") or {}).get("ayahs") or []
    verses = []
    for item in ayahs:
        ayah = item.get("numberInSurah") or item.get("number")
        text = str(item.get("text", "")).strip()
        if ayah and text:
            verses.append({"ayah": int(ayah), "text": text})
    verses.sort(key=lambda x: x["ayah"])
    return verses


def _parse_quranapi_pages(data: dict) -> list[dict]:
    english = data.get("english") or data.get("translation") or []
    verses  = []
    for i, text in enumerate(english, start=1):
        t = str(text).strip()
        if t:
            verses.append({"ayah": i, "text": t})
    return verses

# ─────────────────────────────────────────────────────────────────────────────
# Quran — tafsir fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_tafsir(surah_num: int, ayah_num: int) -> str:
    """Fetch Ibn Kathir tafsir (English) for one verse. Returns '' on failure."""
    url = _TAFSIR_CDN_URL.format(surah=surah_num, ayah=ayah_num)
    data = _get_json(url)
    if data:
        text = (data.get("text") or data.get("tafsir") or "").strip()
        if text:
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

    url2 = _QURAN_COM_TAFSIR_URL.format(surah=surah_num, ayah=ayah_num)
    data2 = _get_json(url2)
    if data2:
        tafsirs = data2.get("tafsirs") or []
        if tafsirs:
            text = (tafsirs[0].get("text") or "").strip()
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

    return ""

# ─────────────────────────────────────────────────────────────────────────────
# Quran — fetch one surah (translation + tafsir per verse)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_surah(surah_num: int) -> tuple[list[dict], str]:
    sources = [
        ("semarketir",     _SEMARKETIR_URL.format(n=surah_num),    _parse_semarketir),
        ("alquran.cloud",  _ALQURAN_CLOUD_URL.format(n=surah_num), _parse_alquran_cloud),
        ("quranapi.pages", _QURANAPI_URL.format(n=surah_num),      _parse_quranapi_pages),
    ]
    for source_name, url, parser in sources:
        data = _get_json(url)
        if data is None:
            continue
        verses = parser(data)
        if verses:
            print(f"    Fetching tafsir for {len(verses)} verses ...", flush=True)
            for v in verses:
                v["tafsir"] = _fetch_tafsir(surah_num, v["ayah"])
            return verses, source_name
    return [], ""


def _group_verses(verses: list[dict], group_size: int = GROUP_SIZE) -> list[dict]:
    """
    GROUP_SIZE=1: one verse per passage.
    Stored text = "[ayah] <translation>\nTafsir: <commentary>"
    """
    passages = []
    for i in range(0, len(verses), group_size):
        group      = verses[i : i + group_size]
        start_ayah = group[0]["ayah"]
        parts = []
        for v in group:
            line   = f"[{v['ayah']}] {v['text']}"
            tafsir = v.get("tafsir", "").strip()
            if tafsir:
                line += f"\nTafsir: {tafsir}"
            parts.append(line)
        passages.append({"ayah": start_ayah, "text": "  ".join(parts)})
    return passages

# ─────────────────────────────────────────────────────────────────────────────
# Hadith — bulk download entire collection in one request
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_hadith_collection(edition: str, display_name: str) -> list[dict]:
    """
    Download the entire hadith collection as a single bulk JSON file.
    fawazahmed0 CDN hosts complete collections — no pagination needed.
    Returns list of normalised hadith dicts.
    """
    url  = _HADITH_BASE.format(edition=edition)
    print(f"  Downloading {display_name} ({edition}) ...", flush=True)
    data = _get_json(url, timeout=120)

    if not data:
        print(f"  [warn] {display_name}: download failed")
        return []

    raw_hadiths = data.get("hadiths") or []
    if not raw_hadiths:
        print(f"  [warn] {display_name}: empty response")
        return []

    entries = []
    for h in raw_hadiths:
        text = str(h.get("text") or "").strip()
        if not text or len(text) < 20:
            continue
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        hadith_num = h.get("hadithnumber") or h.get("number") or 0
        grades     = h.get("grades") or []
        grade      = grades[0].get("grade", "") if grades else ""
        book_info  = h.get("book") or {}
        book_name  = book_info.get("bookname") or book_info.get("arabicname") or ""

        entries.append({
            "hadith_number": hadith_num,
            "text":          text,
            "grade":         grade,
            "book_name":     book_name,
        })

    print(f"    -> {len(entries):,} hadiths loaded")
    return entries


def _hadith_to_entry(h: dict, display_name: str, book_key: str) -> dict:
    """Convert normalised hadith dict to the standard quran_raw.json entry format."""
    hadith_num = h["hadith_number"]
    text       = h["text"]
    grade      = h.get("grade", "")
    book_name  = h.get("book_name", "")

    header = f"[Hadith {hadith_num}]"
    if book_name:
        header += f" {book_name}."
    if grade:
        header += f" Grade: {grade}."
    full_text = f"{header}\n{text}"

    return {
        "text":     full_text,
        "section":  display_name,
        "category": "Hadith",
        "surah":    0,
        "ayah":     hadith_num,
        "religion": "Islam",
        "language": "en",
        "source":   book_key,
    }

# ─────────────────────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section_path(surah_num: int) -> Path:
    return SECTIONS_DIR / f"islam_surah_{surah_num:03d}.json"


def _hadith_path(book_key: str) -> Path:
    return HADITH_DIR / f"islam_hadith_{book_key}.json"


def _save_json(path: Path, data) -> None:
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        Path(tmp).replace(path)
    except Exception:
        path.write_bytes(Path(tmp).read_bytes())
        Path(tmp).unlink(missing_ok=True)


def _load_checkpoint() -> tuple[set[int], set[str], dict]:
    if not CHECKPOINT_PATH.exists():
        return set(), set(), {}
    try:
        cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return (
            set(int(x) for x in cp.get("completed_surahs", [])),
            set(cp.get("completed_hadiths", [])),
            cp.get("stats", {}),
        )
    except Exception:
        return set(), set(), {}


def _save_checkpoint(completed_surahs: set[int], completed_hadiths: set[str], stats: dict):
    _save_json(CHECKPOINT_PATH, {
        "completed_surahs":  sorted(completed_surahs),
        "completed_hadiths": sorted(completed_hadiths),
        "stats":             stats,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Merge all downloaded files → quran_raw.json
# ─────────────────────────────────────────────────────────────────────────────

def merge_all() -> list[dict]:
    all_entries: list[dict] = []

    # 1. Quran sections
    for surah_num in range(1, 115):
        p = _section_path(surah_num)
        if not p.exists():
            continue
        try:
            passages = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] {p.name}: {exc}")
            continue
        meta = SURAH_INFO.get(surah_num, {})
        for passage in passages:
            all_entries.append({
                "text":     passage["text"],
                "section":  meta.get("name", f"Surah {surah_num}"),
                "category": meta.get("revelation", "Meccan"),
                "surah":    surah_num,
                "ayah":     passage["ayah"],
                "religion": "Islam",
                "language": "en",
                "source":   "quran",
            })

    quran_count = len(all_entries)

    # 2. Hadith collections
    for _, display_name, book_key in HADITH_COLLECTIONS:
        p = _hadith_path(book_key)
        if not p.exists():
            continue
        try:
            hadiths = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] {p.name}: {exc}")
            continue
        for h in hadiths:
            all_entries.append(_hadith_to_entry(h, display_name, book_key))

    hadith_count = len(all_entries) - quran_count
    print(f"  Merged: {quran_count:,} Quran + {hadith_count:,} Hadith = {len(all_entries):,} total")
    _save_json(DATA_PATH, all_entries)
    return all_entries

# ─────────────────────────────────────────────────────────────────────────────
# Source probe
# ─────────────────────────────────────────────────────────────────────────────

def _probe_sources() -> None:
    for name, url in [
        ("semarketir",    _SEMARKETIR_URL.format(n=1)),
        ("alquran.cloud", _ALQURAN_CLOUD_URL.format(n=1)),
        ("quranapi",      _QURANAPI_URL.format(n=1)),
    ]:
        print(f"  Probing {name} ... ", end="", flush=True)
        try:
            resp = _session.get(url, timeout=10)
            print("OK v" if resp.status_code == 200 else f"HTTP {resp.status_code}")
        except Exception as exc:
            print(f"failed ({exc})")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_entries: list[dict]) -> None:
    by_source: dict[str, int] = {}
    for e in all_entries:
        s = e.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
    print(f"\n{'=' * 60}")
    print("  DOWNLOAD COMPLETE")
    print(f"  Total entries : {len(all_entries):,}")
    print(f"\n  By source:")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src:<20} {count:>7,}")
    print(f"\n  Output: {DATA_PATH}")
    print(f"{'=' * 60}")

# ─────────────────────────────────────────────────────────────────────────────
# Main download pipeline
# ─────────────────────────────────────────────────────────────────────────────

def download_islam() -> list[dict]:
    print(f"\n{'=' * 60}")
    print("  Islam Comprehensive Download Pipeline")
    print("  Quran (translation + Ibn Kathir tafsir)")
    print("  + Sahih Bukhari, Sahih Muslim, 40 Nawawi,")
    print("    Sunan Abu Dawud, Riyad as-Salihin")
    print(f"{'=' * 60}\n")

    _probe_sources()

    completed_surahs, completed_hadiths, stats = _load_checkpoint()

    # ── Phase 1: Quran ───────────────────────────────────────────────────────
    remaining = sorted(set(range(1, 115)) - completed_surahs)
    if remaining:
        print(f"\n[Phase 1] Quran — {len(remaining)} surahs remaining ...\n")
        for surah_num in remaining:
            meta = SURAH_INFO[surah_num]
            name = meta["name"]
            rev  = meta["revelation"]

            verses, source_used = _fetch_surah(surah_num)
            passages = _group_verses(verses) if verses else []

            if passages:
                _save_json(_section_path(surah_num), passages)
            else:
                print(f"  [warn] Surah {surah_num} ({name}): all sources failed")

            completed_surahs.add(surah_num)
            stats[f"surah_{surah_num}"] = {
                "name": name, "passages": len(passages),
                "verses": len(verses), "source": source_used,
            }
            _save_checkpoint(completed_surahs, completed_hadiths, stats)

            status = f"{len(verses):>3}v -> {len(passages)}p  [{source_used}]" if verses else "EMPTY"
            print(f"  [{surah_num:3d}/114] {name:<28} {rev:<8}  {status}")
    else:
        print("[Phase 1] Quran already complete v")

    # ── Phase 2: Hadith ──────────────────────────────────────────────────────
    print(f"\n[Phase 2] Hadith collections ...\n")
    for edition, display_name, book_key in HADITH_COLLECTIONS:
        if book_key in completed_hadiths:
            print(f"  {display_name}: already downloaded v")
            continue
        hadiths = _fetch_hadith_collection(edition, display_name)
        if hadiths:
            _save_json(_hadith_path(book_key), hadiths)
            completed_hadiths.add(book_key)
            stats[f"hadith_{book_key}"] = {"count": len(hadiths)}
            _save_checkpoint(completed_surahs, completed_hadiths, stats)
        else:
            print(f"  [warn] {display_name}: failed — re-run to retry")

    # ── Phase 3: Merge ───────────────────────────────────────────────────────
    print(f"\n[Phase 3] Merging all sources -> {DATA_PATH.name} ...")
    all_entries = merge_all()
    _print_summary(all_entries)
    return all_entries

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download comprehensive Islam data")
    parser.add_argument("--reset",   action="store_true",
                        help="Clear all checkpoints and re-download everything")
    parser.add_argument("--patch",   action="store_true",
                        help="Retry surahs that returned 0 passages")
    parser.add_argument("--remerge", action="store_true",
                        help="Re-merge existing section files, no download")
    parser.add_argument("--force",   metavar="SURAH", type=int,
                        help="Force re-download one surah (1-114)")
    args = parser.parse_args()

    if args.reset:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        removed = sum(1 for p in SECTIONS_DIR.glob("islam_surah_*.json") if p.unlink() or True)
        removed += sum(1 for p in HADITH_DIR.glob("islam_hadith_*.json") if p.unlink() or True)
        print(f"  [reset] Cleared checkpoint + {removed} section/hadith files.")

    elif args.remerge:
        entries = merge_all()
        _print_summary(entries)
        raise SystemExit(0)

    elif args.force:
        n = args.force
        if not 1 <= n <= 114:
            print(f"  [error] Surah number must be 1-114 (got {n})")
            raise SystemExit(1)
        cs, ch, st = _load_checkpoint()
        cs.discard(n)
        st.pop(f"surah_{n}", None)
        fp = _section_path(n)
        if fp.exists():
            fp.unlink()
        _save_checkpoint(cs, ch, st)
        print(f"  [force] Queued surah {n} ({SURAH_INFO[n]['name']}) for re-download.")

    elif args.patch:
        cs, ch, st = _load_checkpoint()
        patched = [int(k.replace("surah_", "")) for k, v in st.items()
                   if k.startswith("surah_") and isinstance(v, dict) and v.get("passages", 0) == 0]
        for n in patched:
            cs.discard(n)
            st.pop(f"surah_{n}", None)
            fp = _section_path(n)
            if fp.exists():
                fp.unlink()
        _save_checkpoint(cs, ch, st)
        print(f"  [patch] Queued {len(patched)} empty surahs: {patched}" if patched else "  [patch] Nothing to patch.")

    download_islam()