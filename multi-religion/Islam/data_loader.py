"""
Islam Quran Downloader  (data_loader_islam.py)
==============================================
Sources (tried in order until one works):
  1. raw.githubusercontent.com/semarketir/quranjson  — per-surah English translation
  2. api.alquran.cloud/v1/surah/{n}/en.sahih         — alquran.cloud REST API (Sahih International)
  3. quranapi.pages.dev/api/surah.php?surah={n}      — QuranAPI.pages.dev

No API key required. No license restrictions.

Output:
  data/quran_raw.json           — all verse entries ready for chunk_and_embed
  data/sections/islam_*.json    — one file per surah (incremental resume)
  data/checkpoint_islam.json    — checkpoint to resume interrupted downloads

Usage:
  python data_loader_islam.py              # download all (resumes from checkpoint)
  python data_loader_islam.py --reset      # wipe checkpoint + sections, re-download all
  python data_loader_islam.py --patch      # re-run surahs that returned 0 verses
  python data_loader_islam.py --force 2    # force re-download one surah by number
  python data_loader_islam.py --remerge    # re-merge all section files → quran_raw.json
"""

import argparse
import json
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
CHECKPOINT_PATH = DATA_DIR / "checkpoint_islam.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_DELAY = 1.0
TIMEOUT       = 30
MAX_RETRIES   = 3
GROUP_SIZE    = 1     # 1 verse per passage — chunk_and_embed enriches each verse's
                      # embedding with neighbouring verse context at embed time

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
# Source definitions — tried in order, first success wins
# ─────────────────────────────────────────────────────────────────────────────
# Source 1: semarketir/quranjson (raw.githubusercontent.com)
#   URL: .../source/translation/en/en_translation_{n}.json
#   Format: {"1": {"aya": "1", "surah": "1", "text": "..."}, ...}
#
# Source 2: alquran.cloud REST API (Sahih International translation)
#   URL: https://api.alquran.cloud/v1/surah/{n}/en.sahih
#   Format: {"data": {"ayahs": [{"numberInSurah": 1, "text": "..."}, ...]}}
#
# Source 3: quranapi.pages.dev
#   URL: https://quranapi.pages.dev/api/{n}.json
#   Format: {"surahName": "...", "english": ["verse1", "verse2", ...]}

_SOURCES = [
    "semarketir",
    "alquran_cloud",
    "quranapi_pages",
]

_SEMARKETIR_URL    = "https://raw.githubusercontent.com/semarketir/quranjson/master/source/translation/en/en_translation_{n}.json"
_ALQURAN_CLOUD_URL = "https://api.alquran.cloud/v1/surah/{n}/en.sahih"
_QURANAPI_URL      = "https://quranapi.pages.dev/api/{n}.json"

# Tafsir (commentary) sources — fetched per-verse to enrich text content.
# This is what brings Islam data up to parity with other religions file sizes.
# Primary  : spa5k/tafsir_api CDN — Tafsir Ibn Kathir English, per-verse JSON
# Fallback : api.quran.com v4    — tafsir id 169 = Ibn Kathir (English)
_TAFSIR_CDN_URL       = "https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/en-tafisr-ibn-kathir/{surah}/{ayah}.json"
_QURAN_COM_TAFSIR_URL = "https://api.quran.com/api/v4/tafsirs/169/by_ayah/{surah}:{ayah}"

# ─────────────────────────────────────────────────────────────────────────────
# Surah metadata (all 114 hardcoded — no API call needed)
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

def _get_json(url: str) -> dict | list | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                time.sleep(REQUEST_DELAY)
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 0)) or min(20 * attempt, 60)
                print(f"    [429] waiting {wait}s ...")
                time.sleep(wait)
                continue
            # Don't retry 403/404 — source doesn't have this file
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
# Per-source parsers — return list[{ayah, text}] or []
# ─────────────────────────────────────────────────────────────────────────────

def _parse_semarketir(data: dict) -> list[dict]:
    """
    Format: {"1": {"aya": "1", "surah": "1", "text": "..."}, "2": {...}, ...}
    """
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
    """
    Format: {"data": {"ayahs": [{"numberInSurah": 1, "text": "..."}, ...]}}
    """
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
    """
    Format: {"surahName": "...", "english": ["verse1", "verse2", ...]}
    """
    english = data.get("english") or data.get("translation") or []
    verses  = []
    for i, text in enumerate(english, start=1):
        t = str(text).strip()
        if t:
            verses.append({"ayah": i, "text": t})
    return verses


def _fetch_tafsir(surah_num: int, ayah_num: int) -> str:
    """
    Fetch Ibn Kathir tafsir (English commentary) for a single verse.
    Returns the tafsir text, or empty string if both sources fail.
    Strips basic HTML tags that the CDN source includes.
    """
    import re

    # Source 1: spa5k CDN (fast, no rate limit)
    url = _TAFSIR_CDN_URL.format(surah=surah_num, ayah=ayah_num)
    data = _get_json(url)
    if data:
        text = (data.get("text") or data.get("tafsir") or "").strip()
        if text:
            text = re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
            text = re.sub(r"\s+", " ", text).strip()
            return text

    # Source 2: quran.com v4 API
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
# Fetch one surah — tries all three sources
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_surah(surah_num: int) -> tuple[list[dict], str]:
    """
    Returns (verses, source_name) where verses is a sorted list of
    {ayah, text, tafsir} dicts. Returns ([], '') if all sources fail.
    Tafsir (Ibn Kathir commentary) is fetched per-verse to enrich content.
    """
    sources = [
        (
            "semarketir",
            _SEMARKETIR_URL.format(n=surah_num),
            _parse_semarketir,
        ),
        (
            "alquran.cloud",
            _ALQURAN_CLOUD_URL.format(n=surah_num),
            _parse_alquran_cloud,
        ),
        (
            "quranapi.pages.dev",
            _QURANAPI_URL.format(n=surah_num),
            _parse_quranapi_pages,
        ),
    ]

    for source_name, url, parser in sources:
        data = _get_json(url)
        if data is None:
            continue
        verses = parser(data)
        if verses:
            # Enrich each verse with Ibn Kathir tafsir commentary
            print(f"    Fetching tafsir for {len(verses)} verses ...", flush=True)
            for v in verses:
                tafsir = _fetch_tafsir(surah_num, v["ayah"])
                v["tafsir"] = tafsir
            return verses, source_name
        # data came back but parsed empty — try next source

    return [], ""

def _group_verses(verses: list[dict], group_size: int = GROUP_SIZE) -> list[dict]:
    """
    Group verses into passages. With GROUP_SIZE=1 each passage is one verse.
    The stored text combines the translation + tafsir so the JSON is content-rich,
    bringing Islam data to parity in file size with other religions.
    Format:  "[ayah] <translation>\nTafsir: <commentary>"
    """
    passages = []
    for i in range(0, len(verses), group_size):
        group      = verses[i : i + group_size]
        start_ayah = group[0]["ayah"]
        parts = []
        for v in group:
            verse_line = f"[{v['ayah']}] {v['text']}"
            tafsir     = v.get("tafsir", "").strip()
            if tafsir:
                verse_line += f"\nTafsir: {tafsir}"
            parts.append(verse_line)
        joined = "  ".join(parts)
        passages.append({"ayah": start_ayah, "text": joined})
    return passages

# ─────────────────────────────────────────────────────────────────────────────
# Section / checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def _section_path(surah_num: int) -> Path:
    return SECTIONS_DIR / f"islam_surah_{surah_num:03d}.json"


def _save_section(surah_num: int, passages: list[dict]):
    p   = _section_path(surah_num)
    tmp = str(p) + ".tmp"
    Path(tmp).write_text(
        json.dumps(passages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        Path(tmp).replace(p)
    except Exception:
        p.write_bytes(Path(tmp).read_bytes())
        Path(tmp).unlink(missing_ok=True)


def _load_checkpoint() -> tuple[set[int], dict]:
    if not CHECKPOINT_PATH.exists():
        return set(), {}
    try:
        cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(int(x) for x in cp.get("completed", [])), cp.get("stats", {})
    except Exception:
        return set(), {}


def _save_checkpoint(completed: set[int], stats: dict):
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps({"completed": sorted(completed), "stats": stats},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        Path(tmp).replace(CHECKPOINT_PATH)
    except Exception:
        CHECKPOINT_PATH.write_bytes(Path(tmp).read_bytes())
        Path(tmp).unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Merge section files → quran_raw.json
# ─────────────────────────────────────────────────────────────────────────────

def merge_sections() -> list[dict]:
    all_entries: list[dict] = []
    for surah_num in range(1, 115):
        p = _section_path(surah_num)
        if not p.exists():
            continue
        try:
            passages = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] Could not read {p.name}: {exc}")
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
            })
    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps(all_entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        Path(tmp).replace(DATA_PATH)
    except Exception:
        DATA_PATH.write_bytes(Path(tmp).read_bytes())
        Path(tmp).unlink(missing_ok=True)
    return all_entries

# ─────────────────────────────────────────────────────────────────────────────
# Source probe — check which source is reachable before starting
# ─────────────────────────────────────────────────────────────────────────────

def _probe_sources() -> str:
    """Test surah 1 against each source. Return name of first working source."""
    probe_urls = [
        ("semarketir",        _SEMARKETIR_URL.format(n=1)),
        ("alquran.cloud",     _ALQURAN_CLOUD_URL.format(n=1)),
        ("quranapi.pages.dev",_QURANAPI_URL.format(n=1)),
    ]
    for name, url in probe_urls:
        print(f"  Probing {name} ... ", end="", flush=True)
        try:
            resp = _session.get(url, timeout=10)
            if resp.status_code == 200:
                print("OK ✓")
                return name
            print(f"HTTP {resp.status_code}")
        except Exception as exc:
            print(f"failed ({exc})")
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# Main download pipeline
# ─────────────────────────────────────────────────────────────────────────────

def download_islam() -> list[dict]:
    print(f"\n{'=' * 60}")
    print("  Quran Download Pipeline")
    print(f"  Sources  : semarketir → alquran.cloud → quranapi.pages.dev")
    print(f"  Grouping : {GROUP_SIZE} verses per passage")
    print(f"{'=' * 60}\n")

    # Probe which source is reachable
    working = _probe_sources()
    if not working:
        print("\n  [fatal] All sources unreachable. Check your internet connection.")
        return []
    print(f"\n  Using primary source: {working}\n")

    completed, stats = _load_checkpoint()
    if completed:
        print(f"  Resuming — {len(completed)} surahs already done\n")

    remaining = sorted(set(range(1, 115)) - completed)
    if not remaining:
        print("  All 114 surahs already downloaded. Merging sections...")
        all_entries = merge_sections()
        _print_summary(all_entries)
        return all_entries

    source_counts: dict[str, int] = {}

    for surah_num in remaining:
        meta = SURAH_INFO[surah_num]
        name = meta["name"]
        rev  = meta["revelation"]

        verses, source_used = _fetch_surah(surah_num)
        passages = _group_verses(verses) if verses else []

        if passages:
            _save_section(surah_num, passages)
            source_counts[source_used] = source_counts.get(source_used, 0) + 1
        else:
            print(f"  [warn] Surah {surah_num} ({name}): all sources failed — use --patch to retry")

        completed.add(surah_num)
        stats[str(surah_num)] = {
            "name": name, "passages": len(passages),
            "verses": len(verses), "source": source_used,
        }
        _save_checkpoint(completed, stats)

        status = f"{len(verses):>3}v → {len(passages)}p  [{source_used}]" if verses else "EMPTY"
        print(f"  [{surah_num:3d}/114] {name:<28} {rev:<8}  {status}")

    print(f"\n  Merging all sections → {DATA_PATH} ...")
    all_entries = merge_sections()
    _print_summary(all_entries, source_counts)
    return all_entries


def _print_summary(all_entries: list[dict], source_counts: dict | None = None):
    print(f"\n{'=' * 60}")
    print("  DOWNLOAD COMPLETE")
    print(f"  Total passage entries : {len(all_entries):,}")
    meccan  = sum(1 for e in all_entries if e["category"] == "Meccan")
    medinan = sum(1 for e in all_entries if e["category"] == "Medinan")
    print(f"  Meccan  passages      : {meccan:,}")
    print(f"  Medinan passages      : {medinan:,}")
    if source_counts:
        print(f"  Sources used          : {dict(sorted(source_counts.items(), key=lambda x: -x[1]))}")
    print(f"  Output                : {DATA_PATH}")
    print(f"{'=' * 60}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Quran data for Islam chatbot")
    parser.add_argument("--reset",   action="store_true",
                        help="Clear checkpoint + all section files, re-download everything")
    parser.add_argument("--patch",   action="store_true",
                        help="Re-run only surahs that previously returned 0 passages")
    parser.add_argument("--force",   metavar="SURAH", type=int,
                        help="Force re-download one surah by number (1–114)")
    parser.add_argument("--remerge", action="store_true",
                        help="Re-merge all section files into quran_raw.json (no download)")
    args = parser.parse_args()

    if args.reset:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        removed = sum(1 for p in SECTIONS_DIR.glob("islam_surah_*.json")
                      if p.unlink() or True)
        print(f"  [reset] Cleared checkpoint + {removed} section files.")

    elif args.force:
        n = args.force
        if not 1 <= n <= 114:
            print(f"  [error] Surah number must be 1–114 (got {n})")
            raise SystemExit(1)
        completed, stats = _load_checkpoint()
        completed.discard(n)
        stats.pop(str(n), None)
        fp = _section_path(n)
        if fp.exists():
            fp.unlink()
        _save_checkpoint(completed, stats)
        print(f"  [force] Queued surah {n} ({SURAH_INFO[n]['name']}) for re-download.")

    elif args.patch:
        completed, stats = _load_checkpoint()
        patched = [int(k) for k, v in stats.items()
                   if isinstance(v, dict) and v.get("passages", 0) == 0]
        for n in patched:
            completed.discard(n)
            stats.pop(str(n), None)
            fp = _section_path(n)
            if fp.exists():
                fp.unlink()
        _save_checkpoint(completed, stats)
        if patched:
            print(f"  [patch] Queued {len(patched)} empty surah(s): {patched}")
        else:
            print("  [patch] Nothing to patch.")

    elif args.remerge:
        entries = merge_sections()
        print(f"  Done. {len(entries):,} entries → {DATA_PATH}")
        raise SystemExit(0)

    download_islam()