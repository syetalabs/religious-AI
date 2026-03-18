"""
patch_rigveda.py
================
Fetches the complete Rig Veda (Griffith 1896 translation) from
Internet Archive as a plain text file, parses it into per-hymn
verse entries, and saves to data/sections/hindu_Rig_Veda.json.

Sources tried in order:
  1. https://archive.org/download/rigvedacomplete/rigvedacomplete_djvu.txt
  2. https://archive.org/download/HymnsOfTheRigvedaVol-i/HymnsOfTheRigvedaVol-i.txt

Run:
    python patch_rigveda.py
"""

import json
import re
import time
from pathlib import Path

import requests

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_PATH       = DATA_DIR / "hindu_raw.json"
SECTIONS_DIR    = DATA_DIR / "sections"
CHECKPOINT_PATH = DATA_DIR / "checkpoint_hindu.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

SECTION_NAME = "Rig Veda"
TIMEOUT      = 60

SOURCES = [
    ("archive.org rigvedacomplete djvu txt",
     "https://archive.org/download/rigvedacomplete/rigvedacomplete_djvu.txt"),
    ("archive.org HymnsOfRigveda vol1 txt",
     "https://archive.org/download/HymnsOfTheRigvedaVol-i/HymnsOfTheRigvedaVol-i_djvu.txt"),
    ("archive.org Griffith 1896",
     "https://archive.org/download/in.ernet.dli.2015.237767/2015.237767.The-Hymns_djvu.txt"),
    ("sanskritweb PDF text fallback",
     "http://www.sanskritweb.net/rigveda/griffith-p.pdf"),
]

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
})


def _safe_replace(src, dst):
    src_p, dst_p = Path(src), Path(dst)
    try: src_p.replace(dst_p); return
    except: pass
    try:
        if dst_p.exists(): dst_p.unlink()
        src_p.rename(dst_p); return
    except: pass
    dst_p.write_bytes(src_p.read_bytes())
    src_p.unlink(missing_ok=True)


def fetch_text(url, label):
    print(f"  Trying: {label}")
    print(f"    URL: {url}")
    try:
        resp = _session.get(url, timeout=TIMEOUT, stream=True)
        print(f"    HTTP {resp.status_code}  Content-Type: {resp.headers.get('content-type','?')}")
        if resp.status_code == 200:
            content = resp.content
            print(f"    Size: {len(content):,} bytes")
            # Try to decode as text
            for enc in ("utf-8", "latin-1", "utf-8-sig"):
                try:
                    return content.decode(enc)
                except Exception:
                    continue
        return None
    except Exception as exc:
        print(f"    Error: {exc}")
        return None


def parse_griffith_text(text):
    """
    Parse the Griffith plain text into verse entries.
    The text has patterns like:
      HYMN I.  Agni.
      1. I Laud Agni, the chosen Priest...
      2. Worthy is Agni to be praised...
      
      HYMN II.  Vayu.
      1. Beautiful Vayu...
    """
    verses = []
    
    # Detect book headers: BOOK I, Book 1, etc.
    current_book   = 1
    current_hymn   = None
    current_title  = ""
    current_verses = {}  # verse_num -> text lines
    current_v_num  = None

    book_re  = re.compile(r"^\s*BOOK\s+(I{1,3}V?|V?I{0,3}|\d+)\b", re.IGNORECASE)
    hymn_re  = re.compile(r"^\s*HYMN\s+(I{1,4}V?|VI{0,4}|V?I{0,4}X?|\d+)\.?\s*(.*)", re.IGNORECASE)
    verse_re = re.compile(r"^\s*(\d+)\.\s+(.+)")

    roman = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,
             "IX":9,"X":10,"XI":11,"XII":12,"XIII":13,"XIV":14,"XV":15,
             "XVI":16,"XVII":17,"XVIII":18,"XIX":19,"XX":20}

    def roman_to_int(s):
        s = s.strip().upper()
        if s.isdigit():
            return int(s)
        return roman.get(s, None)

    def flush_verse():
        nonlocal current_v_num
        if current_v_num is not None and current_v_num in current_verses:
            lines = current_verses[current_v_num]
            text = " ".join(" ".join(lines).split()).strip()
            if len(text) > 20:
                title_part = f" ({current_title.strip()})" if current_title.strip() else ""
                ref = f"Rig Veda {current_book}.{current_hymn}.{current_v_num}"
                verses.append(f"{ref}{title_part}: {text}")
        current_v_num = None

    def flush_hymn():
        flush_verse()
        current_verses.clear()

    for line in text.split("\n"):
        # Check for book header
        bm = book_re.match(line)
        if bm:
            flush_hymn()
            n = roman_to_int(bm.group(1))
            if n:
                current_book = n
            continue

        # Check for hymn header
        hm = hymn_re.match(line)
        if hm:
            flush_hymn()
            n = roman_to_int(hm.group(1))
            if n:
                current_hymn  = n
                current_title = hm.group(2).strip().rstrip(".")
            continue

        if current_hymn is None:
            continue

        # Check for verse start
        vm = verse_re.match(line)
        if vm:
            flush_verse()
            current_v_num = int(vm.group(1))
            current_verses[current_v_num] = [vm.group(2).strip()]
            continue

        # Continuation of current verse
        stripped = line.strip()
        if current_v_num is not None and stripped and len(stripped) > 3:
            # Skip obvious navigation/footnote lines
            if not re.match(r"^\[|\bPage\b|\bNote\b", stripped):
                if current_v_num not in current_verses:
                    current_verses[current_v_num] = []
                current_verses[current_v_num].append(stripped)

    flush_hymn()
    return verses


def save_section(verses):
    safe = re.sub(r"[^\w]", "_", SECTION_NAME)
    p    = SECTIONS_DIR / f"hindu_{safe}.json"
    tmp  = str(p) + ".tmp"
    Path(tmp).write_text(json.dumps(verses, indent=2, ensure_ascii=False), encoding="utf-8")
    _safe_replace(tmp, str(p))
    print(f"  Saved -> {p}")


def update_checkpoint(n):
    cp = {}
    if CHECKPOINT_PATH.exists():
        try: cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except: pass
    completed = set(cp.get("completed", []))
    stats     = cp.get("stats", {})
    completed.add(SECTION_NAME)
    stats[SECTION_NAME] = {"paragraphs": n}
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps({"completed": sorted(completed), "stats": stats},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    _safe_replace(tmp, str(CHECKPOINT_PATH))
    print("  Checkpoint updated")


def merge_into_raw(verses):
    existing = []
    if DATA_PATH.exists():
        try:
            existing = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            existing = [e for e in existing if e.get("section") != SECTION_NAME]
        except: pass
    for v in verses:
        existing.append({"text": v, "section": SECTION_NAME,
                         "category": "Vedas", "religion": "Hinduism", "language": "en"})
    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    _safe_replace(tmp, str(DATA_PATH))
    print(f"  Merged -> {DATA_PATH}  ({len(existing):,} total entries)")


if __name__ == "__main__":
    print("=" * 60)
    print("  Rig Veda Patcher — Internet Archive plain text")
    print("=" * 60)

    raw_text = None
    for label, url in SOURCES:
        raw_text = fetch_text(url, label)
        if raw_text and len(raw_text) > 10000:
            print(f"  Got {len(raw_text):,} chars from: {label}")
            print(f"  Preview: {raw_text[200:400].replace(chr(10),' ')}")
            break
        time.sleep(1)

    if not raw_text or len(raw_text) < 10000:
        print()
        print("  All sources failed or returned too little text.")
        print()
        print("  MANUAL FALLBACK:")
        print("  1. Open https://archive.org/details/rigvedacomplete in your browser")
        print("  2. Click 'TEXT FILE' or download the .txt version")
        print("  3. Save it as:  data/sections/rig.txt  in this folder")
        print("  4. Re-run:  python patch_rigveda.py --local")
        raise SystemExit(1)

    print("\n  Parsing verses...")
    verses = parse_griffith_text(raw_text)
    print(f"  Parsed: {len(verses)} verses")

    if not verses:
        print("  0 verses parsed. The text format may differ.")
        print(f"  First 500 chars: {raw_text[:500]}")
        raise SystemExit(1)

    print(f"  Sample [0]:   {verses[0][:100]}")
    print(f"  Sample [100]: {verses[min(100,len(verses)-1)][:100]}")

    save_section(verses)
    update_checkpoint(len(verses))
    merge_into_raw(verses)

    print()
    print("=" * 60)
    print(f"  DONE -- {len(verses)} Rig Veda verses added")
    print("=" * 60)



# ── Local file fallback (python patch_rigveda.py --local) ────────────────────
import sys
if "--local" in sys.argv:
    local_path = DATA_DIR / "sections" /"rig.txt"
    if not local_path.exists():
        # Also check current directory
        local_path = Path("rig.txt")
    if local_path.exists():
        print(f"  Reading local file: {local_path}")
        raw_text = local_path.read_text(encoding="utf-8", errors="replace")
        verses   = parse_griffith_text(raw_text)
        print(f"  Parsed: {len(verses)} verses")
        if verses:
            save_section(verses)
            update_checkpoint(len(verses))
            merge_into_raw(verses)
        else:
            print("  0 verses. Check the file format.")
    else:
        print(f"  File not found: {local_path}")
        print("  Download the text from archive.org and save as rig.txt")