"""
Islam Quran Downloader  (data_loader_islam.py)
==============================================
Source:
  Quran text  -> fawazahmed0/quran-api on GitHub (via jsDelivr CDN)
                 Translation: Muhammad Abdel Haleem (eng-abdulhaleem)
                 Fetched per-surah to avoid CDN 403 on large files.
  Surah info  -> Built-in fallback (all 114 surahs hardcoded).

Output:
  data/quran_raw.json           — all verse entries ready for chunk_and_embed
  data/sections/islam_*.json    — one file per surah (incremental resume)
  data/checkpoint_islam.json    — checkpoint to resume interrupted downloads

Usage:
  python data_loader.py              # download all (resumes from checkpoint)
  python data_loader.py --reset      # wipe checkpoint + sections, re-download all
  python data_loader.py --patch      # re-run surahs that returned 0 verses
  python data_loader.py --force 2    # force re-download one surah by number
  python data_loader.py --remerge    # re-merge all section files → quran_raw.json
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
CHECKPOINT_PATH = DATA_DIR / "checkpoint.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_DELAY = 1.5      # seconds between CDN requests
TIMEOUT       = 30
MAX_RETRIES   = 5
GROUP_SIZE    = 5        # verses per passage entry

# Per-surah URL pattern — avoids CDN 403 that hits full-edition JSON
# e.g. surah 2 → .../editions/eng-abdulhaleem/2.json
_CDN_BASE      = "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1"
_TRANSLATION   = "eng-abdulhaleem"
_SURAH_URL     = _CDN_BASE + "/editions/" + _TRANSLATION + "/{surah}.json"

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
# Surah metadata (all 114 surahs hardcoded — no API call needed)
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
                wait = int(resp.headers.get("Retry-After", 0)) or min(20 * attempt, 120)
                print(f"  [429] rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (403, 404):
                print(f"  [HTTP {resp.status_code}] {url}")
                return None
            print(f"  [HTTP {resp.status_code}] attempt {attempt}/{MAX_RETRIES}: {url}")
            time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError as exc:
            print(f"  [conn-error] attempt {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            print(f"  [timeout] attempt {attempt}/{MAX_RETRIES}")
            time.sleep(2 ** attempt)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fetch one surah
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_surah(surah_num: int) -> list[dict]:
    """
    Fetch verses for one surah from the per-chapter endpoint.
    Returns list of {ayah, text} dicts sorted by ayah number.

    Per-surah JSON structure:
    {
      "chapter": {
        "1": "In the name of God...",
        "2": "Praise be to God...",
        ...
      }
    }
    or sometimes:
    {
      "chapter": [
        {"verse": 1, "text": "..."},
        ...
      ]
    }
    """
    url  = _SURAH_URL.format(surah=surah_num)
    data = _get_json(url)
    if not data:
        return []

    chapter = data.get("chapter") or data.get("verses") or data.get("surah") or {}

    verses: list[dict] = []

    if isinstance(chapter, dict):
        # Format: {"1": "text", "2": "text", ...}
        for k, v in chapter.items():
            try:
                ayah = int(k)
                text = str(v).strip()
                if text:
                    verses.append({"ayah": ayah, "text": text})
            except (ValueError, TypeError):
                continue

    elif isinstance(chapter, list):
        # Format: [{"verse": 1, "text": "..."}, ...]
        for item in chapter:
            if not isinstance(item, dict):
                continue
            ayah = item.get("verse") or item.get("ayah") or item.get("id")
            text = item.get("text") or item.get("translation") or ""
            if ayah and text:
                try:
                    verses.append({"ayah": int(ayah), "text": str(text).strip()})
                except (ValueError, TypeError):
                    continue

    verses.sort(key=lambda x: x["ayah"])
    return verses


# ─────────────────────────────────────────────────────────────────────────────
# Group verses into passages
# ─────────────────────────────────────────────────────────────────────────────

def _group_verses(verses: list[dict], group_size: int = GROUP_SIZE) -> list[dict]:
    """Group consecutive verses into passages of group_size each."""
    passages = []
    for i in range(0, len(verses), group_size):
        group      = verses[i : i + group_size]
        start_ayah = group[0]["ayah"]
        joined     = "  ".join(f"[{v['ayah']}] {v['text']}" for v in group)
        passages.append({"ayah": start_ayah, "text": joined})
    return passages


# ─────────────────────────────────────────────────────────────────────────────
# Section file I/O
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


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

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
            passages: list[dict] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] Could not read {p.name}: {exc}")
            continue

        meta = SURAH_INFO.get(surah_num, {})
        name = meta.get("name", f"Surah {surah_num}")
        rev  = meta.get("revelation", "Meccan")

        for passage in passages:
            all_entries.append({
                "text":     passage["text"],
                "section":  name,
                "category": rev,
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
# Main download pipeline
# ─────────────────────────────────────────────────────────────────────────────

def download_islam() -> list[dict]:
    print(f"\n{'=' * 60}")
    print("  Quran Download Pipeline")
    print(f"  Source   : fawazahmed0/quran-api (jsDelivr CDN)")
    print(f"  Edition  : {_TRANSLATION} (Muhammad Abdel Haleem)")
    print(f"  Mode     : per-surah fetch (114 requests)")
    print(f"  Grouping : {GROUP_SIZE} verses per passage")
    print(f"{'=' * 60}\n")

    completed, stats = _load_checkpoint()
    if completed:
        print(f"  Resuming from checkpoint ({len(completed)} surahs already done)\n")

    remaining = sorted(set(range(1, 115)) - completed)
    if not remaining:
        print("  All 114 surahs already downloaded. Merging sections...")
        all_entries = merge_sections()
        _print_summary(all_entries)
        return all_entries

    for surah_num in remaining:
        meta = SURAH_INFO[surah_num]
        name = meta["name"]
        rev  = meta["revelation"]

        verses   = _fetch_surah(surah_num)
        passages = _group_verses(verses, GROUP_SIZE) if verses else []

        if passages:
            _save_section(surah_num, passages)
        else:
            print(f"  [warn] Surah {surah_num} ({name}): 0 verses — will retry with --patch")

        completed.add(surah_num)
        stats[str(surah_num)] = {
            "name": name, "passages": len(passages), "verses": len(verses)
        }
        _save_checkpoint(completed, stats)

        status = f"{len(verses):>3} verses → {len(passages)} passages" if verses else "EMPTY"
        print(f"  [{surah_num:3d}/114] {name:<28} {rev:<8}  {status}")

    print(f"\n  Merging all sections → {DATA_PATH} ...")
    all_entries = merge_sections()
    _print_summary(all_entries)
    return all_entries


def _print_summary(all_entries: list[dict]):
    print(f"\n{'=' * 60}")
    print("  DOWNLOAD COMPLETE")
    print(f"  Total passage entries : {len(all_entries):,}")
    meccan  = sum(1 for e in all_entries if e["category"] == "Meccan")
    medinan = sum(1 for e in all_entries if e["category"] == "Medinan")
    print(f"  Meccan  passages      : {meccan:,}")
    print(f"  Medinan passages      : {medinan:,}")
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
        removed = 0
        for p in SECTIONS_DIR.glob("islam_surah_*.json"):
            p.unlink(); removed += 1
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
            print(f"  [patch] Queued {len(patched)} empty surah(s) for retry: {patched}")
        else:
            print("  [patch] Nothing to patch — all surahs have data.")

    elif args.remerge:
        print("  Re-merging all surah section files...")
        entries = merge_sections()
        print(f"  Done. {len(entries):,} total passage entries → {DATA_PATH}")
        raise SystemExit(0)

    download_islam()