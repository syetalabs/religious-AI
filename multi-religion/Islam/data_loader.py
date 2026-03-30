"""
Islam Data Loader  —  Comprehensive Islamic Knowledge Base
==========================================================
Downloads and merges multiple Islamic text sources into quran_raw.json.

Sources included:
  English:
    1. Quran           — 6,236 verses (Sahih International, alquran.cloud)
                         + Ibn Kathir tafsir per verse (spa5k CDN)
    2. Sahih Bukhari   — ~7,563 hadiths  (fawazahmed0 CDN, eng-bukhari)
    3. Sahih Muslim    — ~3,033 hadiths  (fawazahmed0 CDN, eng-muslim)
    4. Sunan Abu Dawud — ~5,274 hadiths  (fawazahmed0 CDN, eng-abudawud)
    5. Sunan Ibn Majah — ~4,341 hadiths  (fawazahmed0 CDN, eng-ibnmajah)
    6. Muwatta Malik   — ~1,900 hadiths  (fawazahmed0 CDN, eng-malik)

  Sinhala (sin):
    7. Quran           — 6,236 verses    (fawazahmed0 CDN, sin-*)
       NOTE: No Sinhala hadith collections exist on fawazahmed0/hadith-api.

  Tamil (tam):
    8. Quran           — 6,236 verses    (fawazahmed0 CDN, tam-*)
    9. Sahih Bukhari   — ~7,563 hadiths  (fawazahmed0 CDN, tam-bukhari)
       NOTE: Only Bukhari is available in Tamil on fawazahmed0/hadith-api.

All sources are free, no API key required.

Quran editions used (fawazahmed0/quran-api CDN bulk download):
  Sinhala : sin-translationwmabdulhamee  (W.M. Abdul Hameed)
  Tamil   : tam-janturstfoundat          (Jan Trust Foundation)
            tam-abdulhameedbaqa          (Abdul Hameed Baqavi) — secondary fallback

Bulk download URL chain (tried in order until one succeeds):
  1. https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/{edition}.min.json
  2. https://raw.githubusercontent.com/fawazahmed0/quran-api/1/editions/{edition}.min.json
  3. Per-surah alquran.cloud  (e.g. https://api.alquran.cloud/v1/surah/{n}/sin.abdulhameed)

Output:
  data/quran_raw.json              — all entries (all languages) for chunk_and_embed
  data/sections/islam_*.json       — per-surah Quran section files (checkpoint)
  data/hadith/islam_hadith_*.json  — per-collection hadith files (checkpoint)
  data/checkpoint_islam.json       — download progress checkpoint

Usage:
  python data_loader.py              # download all (resumes from checkpoint)
  python data_loader.py --reset      # wipe everything, re-download all
  python data_loader.py --remerge    # re-merge existing files, no download
  python data_loader.py --patch      # retry surahs that returned 0 verses
  python data_loader.py --force 2    # force re-download one surah (en only)
  python data_loader.py --force 2 --lang sin   # force re-download one surah for sinhala
  python data_loader.py --force 2 --lang tam   # force re-download one surah for tamil
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
GROUP_SIZE    = 1     # 1 verse per passage

# ─────────────────────────────────────────────────────────────────────────────
# Languages to download
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (lang_code, quran_edition_id, display_label)
# The fawazahmed0/quran-api bulk endpoint returns the full Quran in one JSON:
#   https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/{edition}.min.json
#
# Sinhala editions confirmed in editions.json:
#   sin-translationwmabdulhamee  (W.M. Abdul Hameed — most complete)
# Tamil editions confirmed in editions.json:
#   tam-johntrustbookshop        (John Trust Book Shop)
#   tam-abdulhameedbaghavi       (alternative, also available)

QURAN_EXTRA_LANGS: list[tuple[str, str, str, str]] = [
    # (lang_code, fawaz_edition,                  display_label, alquran_cloud_edition)
    ("sin", "sin-translationwmabdulhamee", "Sinhala", "sin.abdulhameed"),
    ("tam", "tam-janturstfoundat",         "Tamil",   "ta.tamil"),
]

# Secondary fawaz edition to try if primary bulk download returns empty verses
QURAN_EXTRA_FALLBACK_EDITION: dict[str, str] = {
    "tam": "tam-abdulhameedbaqa",
}

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

# English Quran (per-surah sources, used as fallback chain)
_ALQURAN_CLOUD_URL = "https://api.alquran.cloud/v1/surah/{n}/en.sahih"
_SEMARKETIR_URL    = "https://raw.githubusercontent.com/semarketir/quranjson/master/source/translation/en/en_translation_{n}.json"
_QURANAPI_URL      = "https://quranapi.pages.dev/api/{n}.json"

# fawazahmed0/quran-api  — bulk full-Quran download per edition (no auth)
# Multiple mirrors tried in order; jsdelivr and raw.githubusercontent are independent domains
_FAWAZ_QURAN_BULK_URLS = [
    "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/{edition}.min.json",
    "https://raw.githubusercontent.com/fawazahmed0/quran-api/1/editions/{edition}.min.json",
]

# alquran.cloud per-surah endpoint — final fallback when bulk CDN is blocked
# edition examples: "sin.abdulhameed", "ta.tamil"
_ALQURAN_CLOUD_LANG_URL = "https://api.alquran.cloud/v1/surah/{n}/{edition}"

# Quran tafsir — Ibn Kathir English (English only)
_TAFSIR_CDN_URL       = "https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/en-tafisr-ibn-kathir/{surah}/{ayah}.json"
_QURAN_COM_TAFSIR_URL = "https://api.quran.com/api/v4/tafsirs/169/by_ayah/{surah}:{ayah}"

# Hadith — fawazahmed0 CDN (entire collection in one bulk JSON)
_HADITH_BASE = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{edition}.min.json"

# ─────────────────────────────────────────────────────────────────────────────
# Hadith collections per language
# Each entry: (edition_id, display_name, book_key, lang_code)
# ─────────────────────────────────────────────────────────────────────────────

HADITH_COLLECTIONS: list[tuple[str, str, str, str]] = [
    # English
    ("eng-bukhari",  "Sahih al-Bukhari",  "bukhari",  "en"),
    ("eng-muslim",   "Sahih Muslim",      "muslim",   "en"),
    ("eng-abudawud", "Sunan Abu Dawud",   "abudawud", "en"),
    ("eng-ibnmajah", "Sunan Ibn Majah",   "ibnmajah", "en"),
    ("eng-malik",    "Muwatta Malik",     "malik",    "en"),
    # Tamil  (only Bukhari is available in Tamil on fawazahmed0/hadith-api)
    ("tam-bukhari",  "Sahih al-Bukhari",  "bukhari",  "tam"),
    # Sinhala — no hadith collections exist on fawazahmed0/hadith-api
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
# English Quran — per-source parsers
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
    verses = []
    for i, text in enumerate(english, start=1):
        t = str(text).strip()
        if t:
            verses.append({"ayah": i, "text": t})
    return verses

# ─────────────────────────────────────────────────────────────────────────────
# fawazahmed0 bulk Quran parser (Sinhala / Tamil / any language)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fawaz_bulk(data: dict, surah_num: int) -> list[dict]:
    """
    Parse a fawazahmed0/quran-api bulk edition file for one surah.

    Handles all known response formats:

    Format B — standard fawaz bulk (most editions):
        { "quran": [ { "chapter": 1, "verse": { "1": "text", "2": "text", ... } }, ... ] }

    Format C — malformed/placeholder bulk (some Tamil editions):
        { "quran": [ { "chapter": 1, "verse": { "1": 1, "2": 2, ... } } ] }
        Verse values are ints, not text. Detected and returns [] so fallback runs.

    Format D — flat key map:
        { "1:1": "text", "1:2": "text", ... }

    Returns [] on any format that contains no usable text, triggering the
    alquran.cloud per-surah fallback in _fetch_surah_lang.
    """
    # Format B / C: top-level "quran" list
    quran_list = data.get("quran")
    if quran_list and isinstance(quran_list, list):
        for chapter in quran_list:
            if not isinstance(chapter, dict):
                continue
            if chapter.get("chapter") == surah_num:
                verse_map = chapter.get("verse") or {}
                if not isinstance(verse_map, dict):
                    return []  # unexpected structure
                verses = []
                for k, v in verse_map.items():
                    # Format C guard: skip entries where value is not a string
                    if not isinstance(v, str):
                        return []  # whole edition is non-text; bail immediately
                    text = v.strip()
                    if not text:
                        continue
                    try:
                        ayah = int(k)
                    except (ValueError, TypeError):
                        continue
                    verses.append({"ayah": ayah, "text": text})
                verses.sort(key=lambda x: x["ayah"])
                return verses
        return []

    # Format D: flat "chapter:verse" key map  e.g. {"1:1": "text", ...}
    verses = []
    prefix = f"{surah_num}:"
    for k, v in data.items():
        if not isinstance(k, str) or not k.startswith(prefix):
            continue
        if not isinstance(v, str):
            continue
        text = v.strip()
        if not text:
            continue
        try:
            ayah = int(k.split(":")[1])
        except (ValueError, IndexError):
            continue
        verses.append({"ayah": ayah, "text": text})
    if verses:
        verses.sort(key=lambda x: x["ayah"])
        return verses

    return []


# Cache for bulk downloads to avoid re-fetching the same file per surah
_bulk_cache: dict[str, dict] = {}


def _fetch_bulk_edition(edition: str) -> dict | None:
    """
    Download the full Quran for a given edition (cached).
    Tries all mirrors in _FAWAZ_QURAN_BULK_URLS in order.
    Returns None only if every mirror fails.
    """
    if edition in _bulk_cache:
        return _bulk_cache[edition]
    for url_template in _FAWAZ_QURAN_BULK_URLS:
        url  = url_template.format(edition=edition)
        print(f"    Trying bulk: {url}", flush=True)
        data = _get_json(url, timeout=120)
        if data:
            _bulk_cache[edition] = data
            print(f"    Bulk loaded: {url}")
            return data
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Quran tafsir (English only)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_tafsir(surah_num: int, ayah_num: int) -> str:
    url  = _TAFSIR_CDN_URL.format(surah=surah_num, ayah=ayah_num)
    data = _get_json(url)
    if data:
        text = (data.get("text") or data.get("tafsir") or "").strip()
        if text:
            text = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", text).strip()
    url2  = _QURAN_COM_TAFSIR_URL.format(surah=surah_num, ayah=ayah_num)
    data2 = _get_json(url2)
    if data2:
        tafsirs = data2.get("tafsirs") or []
        if tafsirs:
            text = (tafsirs[0].get("text") or "").strip()
            text = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", text).strip()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# English Quran — fetch one surah (translation + tafsir per verse)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_surah_en(surah_num: int) -> tuple[list[dict], str]:
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

# ─────────────────────────────────────────────────────────────────────────────
# Sinhala / Tamil Quran — fetch one surah from fawazahmed0 bulk file
# ─────────────────────────────────────────────────────────────────────────────

def _parse_alquran_cloud_lang(data: dict) -> list[dict]:
    """Parse alquran.cloud per-surah response for any non-English language."""
    ayahs = (data.get("data") or {}).get("ayahs") or []
    verses = []
    for item in ayahs:
        ayah = item.get("numberInSurah") or item.get("number")
        text = str(item.get("text", "")).strip()
        if ayah and text:
            verses.append({"ayah": int(ayah), "text": text})
    verses.sort(key=lambda x: x["ayah"])
    return verses


def _fetch_surah_lang(surah_num: int, edition: str, lang_label: str,
                      alquran_edition: str = "") -> tuple[list[dict], str]:
    """
    Fetch one surah for a non-English language. Strategy:
      1. Try fawazahmed0 bulk file (all mirrors in _FAWAZ_QURAN_BULK_URLS).
      2. If bulk returned data but 0 verses, try secondary fawaz edition if configured.
      3. Fall back to alquran.cloud per-surah request (different domain, avoids CDN blocks).
    Returns (verses, source_name).
    """
    # Step 1: bulk fawaz (mirrors tried inside _fetch_bulk_edition)
    bulk = _fetch_bulk_edition(edition)
    if bulk:
        verses = _parse_fawaz_bulk(bulk, surah_num)
        if verses:
            return verses, f"fawaz/{edition}"

    # Step 2: secondary fawaz edition (e.g. tam-abdulhameedbaqa for Tamil)
    fallback_ed = QURAN_EXTRA_FALLBACK_EDITION.get(
        next((lc for lc, ed, _, _ in QURAN_EXTRA_LANGS if ed == edition), ""), ""
    )
    if fallback_ed and fallback_ed != edition:
        bulk2 = _fetch_bulk_edition(fallback_ed)
        if bulk2:
            verses = _parse_fawaz_bulk(bulk2, surah_num)
            if verses:
                return verses, f"fawaz/{fallback_ed}"

    # Step 3: alquran.cloud per-surah (completely different domain)
    if alquran_edition:
        url  = _ALQURAN_CLOUD_LANG_URL.format(n=surah_num, edition=alquran_edition)
        data = _get_json(url)
        if data:
            verses = _parse_alquran_cloud_lang(data)
            if verses:
                return verses, f"alquran.cloud/{alquran_edition}"

    print(f"    [warn] {lang_label} Surah {surah_num}: all sources failed")
    return [], ""

# ─────────────────────────────────────────────────────────────────────────────
# Group verses into passages
# ─────────────────────────────────────────────────────────────────────────────

def _group_verses(verses: list[dict], include_tafsir: bool = False,
                  group_size: int = GROUP_SIZE) -> list[dict]:
    passages = []
    for i in range(0, len(verses), group_size):
        group      = verses[i : i + group_size]
        start_ayah = group[0]["ayah"]
        parts      = []
        for v in group:
            line   = f"[{v['ayah']}] {v['text']}"
            if include_tafsir:
                tafsir = v.get("tafsir", "").strip()
                if tafsir:
                    line += f"\nTafsir: {tafsir}"
            parts.append(line)
        passages.append({"ayah": start_ayah, "text": "  ".join(parts)})
    return passages

# ─────────────────────────────────────────────────────────────────────────────
# Hadith — bulk download entire collection
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_hadith_collection(edition: str, display_name: str) -> list[dict]:
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
        text       = re.sub(r"<[^>]+>", " ", text)
        text       = re.sub(r"\s+", " ", text).strip()
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


def _hadith_to_entry(h: dict, display_name: str, book_key: str, lang: str) -> dict:
    hadith_num = h["hadith_number"]
    text       = h["text"]
    grade      = h.get("grade", "")
    book_name  = h.get("book_name", "")
    header     = f"[Hadith {hadith_num}]"
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
        "language": lang,
        "source":   book_key,
    }

# ─────────────────────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section_path(surah_num: int, lang: str = "en") -> Path:
    return SECTIONS_DIR / f"islam_{lang}_surah_{surah_num:03d}.json"


def _hadith_path(book_key: str, lang: str = "en") -> Path:
    return HADITH_DIR / f"islam_hadith_{lang}_{book_key}.json"


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


def _load_checkpoint() -> tuple[dict[str, set[int]], set[str], dict]:
    """
    Returns:
      completed_surahs  : {lang_code -> set of completed surah numbers}
      completed_hadiths : set of "{lang}_{book_key}" strings
      stats             : dict
    """
    if not CHECKPOINT_PATH.exists():
        return {}, set(), {}
    try:
        cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        raw_cs = cp.get("completed_surahs", {})
        # Support old single-lang format: {"completed_surahs": [1,2,...]}
        if isinstance(raw_cs, list):
            raw_cs = {"en": raw_cs}
        completed_surahs = {lang: set(int(x) for x in nums) for lang, nums in raw_cs.items()}
        return (
            completed_surahs,
            set(cp.get("completed_hadiths", [])),
            cp.get("stats", {}),
        )
    except Exception:
        return {}, set(), {}


def _save_checkpoint(completed_surahs: dict[str, set[int]],
                     completed_hadiths: set[str],
                     stats: dict):
    _save_json(CHECKPOINT_PATH, {
        "completed_surahs":  {lang: sorted(nums) for lang, nums in completed_surahs.items()},
        "completed_hadiths": sorted(completed_hadiths),
        "stats":             stats,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Merge all downloaded files → quran_raw.json
# ─────────────────────────────────────────────────────────────────────────────

def merge_all() -> list[dict]:
    all_entries: list[dict] = []

    # 1. Quran sections (all languages)
    all_lang_codes = ["en"] + [lc for lc, _, _, _ in QURAN_EXTRA_LANGS]
    for lang in all_lang_codes:
        lang_count = 0
        for surah_num in range(1, 115):
            p = _section_path(surah_num, lang)
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
                    "language": lang,
                    "source":   "quran",
                })
                lang_count += 1
        if lang_count:
            print(f"    Quran [{lang}]: {lang_count:,} passages")

    quran_count = len(all_entries)

    # 2. Hadith collections
    for edition, display_name, book_key, lang in HADITH_COLLECTIONS:
        p = _hadith_path(book_key, lang)
        if not p.exists():
            continue
        try:
            hadiths = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] {p.name}: {exc}")
            continue
        count = 0
        for h in hadiths:
            all_entries.append(_hadith_to_entry(h, display_name, book_key, lang))
            count += 1
        print(f"    Hadith [{lang}] {display_name}: {count:,}")

    hadith_count = len(all_entries) - quran_count
    print(f"  Merged: {quran_count:,} Quran + {hadith_count:,} Hadith = {len(all_entries):,} total")
    _save_json(DATA_PATH, all_entries)
    return all_entries

# ─────────────────────────────────────────────────────────────────────────────
# Source probe
# ─────────────────────────────────────────────────────────────────────────────

def _probe_sources() -> None:
    probes = [
        ("semarketir",    _SEMARKETIR_URL.format(n=1)),
        ("alquran.cloud", _ALQURAN_CLOUD_URL.format(n=1)),
        ("quranapi",      _QURANAPI_URL.format(n=1)),
    ]
    for lang, edition, label, alquran_ed in QURAN_EXTRA_LANGS:
        probes.append((f"jsdelivr/{lang}",
                       _FAWAZ_QURAN_BULK_URLS[0].format(edition=edition)))
        probes.append((f"rawgithub/{lang}",
                       _FAWAZ_QURAN_BULK_URLS[1].format(edition=edition)))
        if alquran_ed:
            probes.append((f"alquran.cloud/{lang}",
                           _ALQURAN_CLOUD_LANG_URL.format(n=1, edition=alquran_ed)))
    for name, url in probes:
        print(f"  Probing {name} ... ", end="", flush=True)
        try:
            resp = _session.get(url, timeout=15)
            print("OK" if resp.status_code == 200 else f"HTTP {resp.status_code}")
        except Exception as exc:
            print(f"failed ({exc})")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_entries: list[dict]) -> None:
    by_lang: dict[str, int] = {}
    for e in all_entries:
        key = f"{e.get('language','?')}:{e.get('source','?')}"
        by_lang[key] = by_lang.get(key, 0) + 1
    print(f"\n{'=' * 60}")
    print("  DOWNLOAD COMPLETE")
    print(f"  Total entries : {len(all_entries):,}")
    print(f"\n  By language:source:")
    for k, count in sorted(by_lang.items(), key=lambda x: -x[1]):
        print(f"    {k:<30} {count:>7,}")
    print(f"\n  Output: {DATA_PATH}")
    print(f"{'=' * 60}")

# ─────────────────────────────────────────────────────────────────────────────
# Main download pipeline
# ─────────────────────────────────────────────────────────────────────────────

def download_islam() -> list[dict]:
    print(f"\n{'=' * 60}")
    print("  Islam Comprehensive Download Pipeline")
    print("  Languages: English | Sinhala | Tamil")
    print("  Quran (EN + tafsir) + Quran (SIN, TAM)")
    print("  + Hadith EN (Bukhari/Muslim/AbuDawud/IbnMajah/Malik)")
    print("  + Hadith TAM (Bukhari only)")
    print(f"{'=' * 60}\n")

    _probe_sources()

    completed_surahs, completed_hadiths, stats = _load_checkpoint()

    # Ensure every language has an entry in the checkpoint dict
    for lang in ["en"] + [lc for lc, _, _, _ in QURAN_EXTRA_LANGS]:
        completed_surahs.setdefault(lang, set())

    # ── Phase 1: English Quran (translation + tafsir) ────────────────────────
    # Cross-check checkpoint against actual files on disk so a partial/wiped
    # data/sections directory doesn't silently skip re-download.
    for n in list(completed_surahs["en"]):
        if not _section_path(n, "en").exists():
            completed_surahs["en"].discard(n)  # file missing — re-queue
    en_remaining = sorted(set(range(1, 115)) - completed_surahs["en"])
    if en_remaining:
        print(f"\n[Phase 1] English Quran — {len(en_remaining)} surahs remaining ...\n")
        for surah_num in en_remaining:
            meta = SURAH_INFO[surah_num]
            name, rev = meta["name"], meta["revelation"]
            verses, source_used = _fetch_surah_en(surah_num)
            passages = _group_verses(verses, include_tafsir=True) if verses else []
            if passages:
                _save_json(_section_path(surah_num, "en"), passages)
            else:
                print(f"  [warn] EN Surah {surah_num} ({name}): all sources failed")
            completed_surahs["en"].add(surah_num)
            stats[f"en_surah_{surah_num}"] = {
                "name": name, "passages": len(passages),
                "verses": len(verses), "source": source_used,
            }
            _save_checkpoint(completed_surahs, completed_hadiths, stats)
            status = f"{len(verses):>3}v -> {len(passages)}p  [{source_used}]" if verses else "EMPTY"
            print(f"  EN [{surah_num:3d}/114] {name:<28} {rev:<8}  {status}")
    else:
        print("[Phase 1] English Quran already complete ✓")

    # ── Phase 2: Sinhala & Tamil Quran ───────────────────────────────────────
    for lang, edition, label, alquran_ed in QURAN_EXTRA_LANGS:
        # Cross-check checkpoint against actual files on disk
        for n in list(completed_surahs[lang]):
            if not _section_path(n, lang).exists():
                completed_surahs[lang].discard(n)
        remaining = sorted(set(range(1, 115)) - completed_surahs[lang])
        if not remaining:
            print(f"\n[Phase 2/{label}] Already complete ✓")
            continue

        print(f"\n[Phase 2/{label}] {label} Quran ({edition}) — {len(remaining)} surahs remaining ...")
        # Pre-fetch the bulk file once for this edition
        print(f"  Pre-loading bulk edition file ...", flush=True)
        bulk = _fetch_bulk_edition(edition)
        if not bulk:
            print(f"  [warn] Bulk CDN unavailable for {edition}. Will use alquran.cloud per-surah fallback ({alquran_ed}).")
        else:
            print(f"  Bulk file loaded. Processing surahs ...")

        for surah_num in remaining:
            meta = SURAH_INFO[surah_num]
            name, rev = meta["name"], meta["revelation"]
            verses, source_used = _fetch_surah_lang(surah_num, edition, label, alquran_ed)
            passages = _group_verses(verses, include_tafsir=False) if verses else []
            if passages:
                _save_json(_section_path(surah_num, lang), passages)
            else:
                print(f"  [warn] {label} Surah {surah_num} ({name}): 0 verses extracted")
            completed_surahs[lang].add(surah_num)
            stats[f"{lang}_surah_{surah_num}"] = {
                "name": name, "passages": len(passages), "verses": len(verses),
            }
            _save_checkpoint(completed_surahs, completed_hadiths, stats)
            status = f"{len(verses):>3}v -> {len(passages)}p" if verses else "EMPTY"
            print(f"  {label} [{surah_num:3d}/114] {name:<28} {rev:<8}  {status}")

    # ── Phase 3: Hadith (all languages) ──────────────────────────────────────
    print(f"\n[Phase 3] Hadith collections ...\n")
    for edition, display_name, book_key, lang in HADITH_COLLECTIONS:
        cp_key = f"{lang}_{book_key}"
        if cp_key in completed_hadiths:
            print(f"  [{lang}] {display_name}: already downloaded ✓")
            continue
        hadiths = _fetch_hadith_collection(edition, f"[{lang}] {display_name}")
        if hadiths:
            _save_json(_hadith_path(book_key, lang), hadiths)
            completed_hadiths.add(cp_key)
            stats[f"hadith_{lang}_{book_key}"] = {"count": len(hadiths)}
            _save_checkpoint(completed_surahs, completed_hadiths, stats)
        else:
            print(f"  [warn] [{lang}] {display_name}: failed — re-run to retry")

    # ── Phase 4: Merge ────────────────────────────────────────────────────────
    print(f"\n[Phase 4] Merging all sources -> {DATA_PATH.name} ...")
    all_entries = merge_all()
    _print_summary(all_entries)
    return all_entries

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download comprehensive Islam data (EN/SIN/TAM)")
    parser.add_argument("--reset",   action="store_true",
                        help="Clear all checkpoints and re-download everything")
    parser.add_argument("--patch",   action="store_true",
                        help="Retry surahs that returned 0 passages")
    parser.add_argument("--remerge", action="store_true",
                        help="Re-merge existing section files, no download")
    parser.add_argument("--force",   metavar="SURAH", type=int,
                        help="Force re-download one surah (1-114)")
    parser.add_argument("--lang",    metavar="LANG", default="en",
                        help="Language for --force / --patch (en, sin, tam). Default: en")
    args = parser.parse_args()

    if args.reset:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        removed = sum(1 for p in SECTIONS_DIR.glob("islam_*_surah_*.json") if p.unlink() or True)
        removed += sum(1 for p in HADITH_DIR.glob("islam_hadith_*.json") if p.unlink() or True)
        print(f"  [reset] Cleared checkpoint + {removed} section/hadith files.")

    elif args.remerge:
        entries = merge_all()
        _print_summary(entries)
        raise SystemExit(0)

    elif args.force:
        n    = args.force
        lang = args.lang
        if not 1 <= n <= 114:
            print(f"  [error] Surah number must be 1-114 (got {n})")
            raise SystemExit(1)
        cs, ch, st = _load_checkpoint()
        cs.setdefault(lang, set()).discard(n)
        st.pop(f"{lang}_surah_{n}", None)
        fp = _section_path(n, lang)
        if fp.exists():
            fp.unlink()
        _save_checkpoint(cs, ch, st)
        print(f"  [force] Queued surah {n} ({SURAH_INFO[n]['name']}) [{lang}] for re-download.")

    elif args.patch:
        lang     = args.lang
        cs, ch, st = _load_checkpoint()
        patched  = [
            int(k.replace(f"{lang}_surah_", ""))
            for k, v in st.items()
            if k.startswith(f"{lang}_surah_") and isinstance(v, dict) and v.get("passages", 0) == 0
        ]
        for n in patched:
            cs.setdefault(lang, set()).discard(n)
            st.pop(f"{lang}_surah_{n}", None)
            fp = _section_path(n, lang)
            if fp.exists():
                fp.unlink()
        _save_checkpoint(cs, ch, st)
        print(
            f"  [patch] [{lang}] Queued {len(patched)} empty surahs: {patched}"
            if patched else f"  [patch] [{lang}] Nothing to patch."
        )

    download_islam()