"""
patch_gita.py

Fetches Bhagavad Gita English translations from:
  praneshp1org/Bhagavad-Gita-JSON-data  (verse.json + translation.json)

Prints field names of both files so any mismatch is immediately visible,
then joins them flexibly using the first matching ID/language field found.

Run:  python patch_gita.py
"""

import json
import re
import time
from pathlib import Path

import requests

# -- Paths (must match download_hindu.py) ------------------------------------
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_PATH       = DATA_DIR / "hindu_raw.json"
SECTIONS_DIR    = DATA_DIR / "sections"
CHECKPOINT_PATH = DATA_DIR / "checkpoint_hindu.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

SECTION_NAME    = "Bhagavad Gita"
TIMEOUT         = 30
VERSE_URL       = "https://raw.githubusercontent.com/praneshp1org/Bhagavad-Gita-JSON-data/main/verse.json"
TRANSLATION_URL = "https://raw.githubusercontent.com/praneshp1org/Bhagavad-Gita-JSON-data/main/translation.json"

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
})

# ----------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------

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


def _fetch_json(url: str, label: str) -> list | None:
    print(f"  Fetching {label} ...")
    for attempt in range(1, 4):
        try:
            resp = _session.get(url, timeout=TIMEOUT)
            print(f"    HTTP {resp.status_code}  ({len(resp.content):,} bytes)")
            if resp.status_code != 200:
                print(f"    Non-200, attempt {attempt}/3")
                time.sleep(2 ** attempt)
                continue
            # utf-8-sig handles BOM; fall back to utf-8
            try:
                text = resp.content.decode("utf-8-sig")
            except Exception:
                text = resp.content.decode("utf-8", errors="replace")
            data = json.loads(text)
            print(f"    {len(data)} records loaded")
            return data
        except Exception as exc:
            print(f"    Error attempt {attempt}/3: {exc}")
            time.sleep(2 ** attempt)
    return None

# ----------------------------------------------------------------------------
# Inspect + join
# ----------------------------------------------------------------------------

def _find_key(record: dict, candidates: list[str]) -> str | None:
    """Return the first key from candidates that exists in record."""
    for c in candidates:
        if c in record:
            return c
    return None


def build_verses(verse_data: list, translation_data: list) -> list[str]:
    # ---- Print actual field names so any mismatch is visible ----------------
    print()
    print("  verse.json sample record:")
    for k, v in verse_data[0].items():
        print(f"    {k!r}: {str(v)[:70]!r}")

    print()
    print("  translation.json sample record:")
    for k, v in translation_data[0].items():
        print(f"    {k!r}: {str(v)[:70]!r}")
    print()

    # ---- Detect key names dynamically ---------------------------------------
    vr = verse_data[0]
    tr = translation_data[0]

    # ID key that links the two files
    v_id_key = _find_key(vr, ["verse_id", "id", "slok_id", "shloka_id"])
    t_id_key = _find_key(tr, ["verse_id", "id", "slok_id", "shloka_id",
                               "verseId", "verse_id"])

    # Chapter / verse number keys in verse.json
    ch_key = _find_key(vr, ["chapter_number", "chapter_id", "chapter", "ch"])
    vn_key = _find_key(vr, ["verse_number",  "verse_id",   "verse",  "sl"])
    # (verse_number is distinct from verse_id — prefer verse_number)
    if vn_key == v_id_key and "verse_number" not in vr:
        vn_key = _find_key(vr, ["verse_number", "verse", "sl", "shloka"])

    # Language key in translation.json
    lang_key = _find_key(tr, ["language", "lang", "language_id"])

    # Translation text key
    desc_key = _find_key(tr, ["description", "translation", "meaning",
                               "text", "verse_meaning", "purport"])

    print(f"  Detected keys:")
    print(f"    verse.json    id={v_id_key!r}  chapter={ch_key!r}  verse_num={vn_key!r}")
    print(f"    transl.json   id={t_id_key!r}  lang={lang_key!r}  text={desc_key!r}")
    print()

    if not all([v_id_key, t_id_key, ch_key, vn_key, desc_key]):
        print("  ERROR: Could not detect all required keys. See field names above.")
        return []

    # ---- Build verse_id -> (chapter, verse_number) --------------------------
    meta: dict[str, tuple] = {}
    for v in verse_data:
        vid = str(v.get(v_id_key, "")).strip()
        ch  = v.get(ch_key)
        vn  = v.get(vn_key)
        if vid and ch is not None and vn is not None:
            meta[vid] = (ch, vn)

    print(f"  verse_meta entries: {len(meta)}")

    # ---- Collect English translations per verse_id --------------------------
    PREFERRED = {"swami sivananda", "dr. s. sankaranarayan", "shri purohit swami"}
    best: dict[str, str] = {}

    for t in translation_data:
        vid  = str(t.get(t_id_key, "")).strip()
        desc = str(t.get(desc_key, "")).strip()

        if not vid or not desc:
            continue

        # Language filter: keep English, or keep if no language field at all
        if lang_key:
            lang = str(t.get(lang_key, "")).lower()
            if lang and "english" not in lang and lang not in ("en", "1", ""):
                continue

        if vid not in best:
            best[vid] = desc
        else:
            author = str(t.get("author_name", "")).lower()
            if author in PREFERRED:
                best[vid] = desc

    print(f"  English translations collected: {len(best)}")

    # Show a sample of IDs from each side to spot mismatches
    meta_sample = list(meta.keys())[:5]
    best_sample = list(best.keys())[:5]
    print(f"  verse_meta ID samples : {meta_sample}")
    print(f"  translation ID samples: {best_sample}")

    # ---- Join ---------------------------------------------------------------
    rows: list[tuple] = []
    for vid, text in best.items():
        if vid in meta:
            ch, vn = meta[vid]
            rows.append((int(ch), int(vn), text))

    rows.sort(key=lambda x: (x[0], x[1]))
    result = [f"Chapter {ch}, Verse {vn}: {text}" for ch, vn, text in rows]
    print(f"  Joined verses: {len(result)}")
    return result

# ----------------------------------------------------------------------------
# Save / checkpoint / merge
# ----------------------------------------------------------------------------

def save_section(verses: list[str]):
    safe = re.sub(r"[^\w]", "_", SECTION_NAME)
    p    = SECTIONS_DIR / f"hindu_{safe}.json"
    tmp  = str(p) + ".tmp"
    Path(tmp).write_text(json.dumps(verses, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    _safe_replace(tmp, str(p))
    print(f"  Saved section -> {p}")


def update_checkpoint(n: int):
    cp: dict = {}
    if CHECKPOINT_PATH.exists():
        try:
            cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    completed = set(cp.get("completed", []))
    stats     = cp.get("stats", {})
    completed.add(SECTION_NAME)
    stats[SECTION_NAME] = {"paragraphs": n}
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps({"completed": sorted(completed), "stats": stats},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(CHECKPOINT_PATH))
    print("  Checkpoint updated")


def merge_into_raw(verses: list[str]):
    existing: list[dict] = []
    if DATA_PATH.exists():
        try:
            existing = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            existing = [e for e in existing if e.get("section") != SECTION_NAME]
        except Exception:
            pass
    for v in verses:
        existing.append({
            "text":     v,
            "section":  SECTION_NAME,
            "category": "Epics",
            "religion": "Hinduism",
            "language": "en",
        })
    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    _safe_replace(tmp, str(DATA_PATH))
    print(f"  Merged -> {DATA_PATH}  ({len(existing):,} total entries)")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Bhagavad Gita Patcher")
    print("  Source: praneshp1org/Bhagavad-Gita-JSON-data")
    print("=" * 60)

    verse_data = _fetch_json(VERSE_URL, "verse.json")
    if not verse_data:
        print("  Could not fetch verse.json"); raise SystemExit(1)

    translation_data = _fetch_json(TRANSLATION_URL, "translation.json")
    if not translation_data:
        print("  Could not fetch translation.json"); raise SystemExit(1)

    verses = build_verses(verse_data, translation_data)

    if not verses:
        print()
        print("  0 verses produced. Check the field name output above.")
        print("  The script prints every key/value in the first record of each file.")
        print("  Update the 'candidates' lists in _find_key() calls to match.")
        raise SystemExit(1)

    print()
    print(f"  {len(verses)} English verses assembled")
    print(f"  Sample [  0]: {verses[0][:100]}")
    print(f"  Sample [ 99]: {verses[min(99,len(verses)-1)][:100]}")
    print(f"  Sample [699]: {verses[min(699,len(verses)-1)][:100]}")

    save_section(verses)
    update_checkpoint(len(verses))
    merge_into_raw(verses)

    print()
    print("=" * 60)
    print(f"  DONE -- {len(verses)} Bhagavad Gita verses in English")
    print("=" * 60)