import json
import os
import re
import time
from pathlib import Path
import requests
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR    = DATA_DIR / "sections"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.json"
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GitHub settings
# ═══════════════════════════════════════════════════════════════════════════════
GITHUB_API       = "https://api.github.com"
TIPITAKA_LK_REPO   = "pathnirvana/tipitaka.lk"
TIPITAKA_LK_BRANCH = "master"
TIPITAKA_LK_PATH   = "public/static/text"

REQUEST_DELAY = 0.4
TIMEOUT       = 20
MAX_RETRIES   = 3

_gh_session = requests.Session()
_gh_session.headers.update({
    "User-Agent": "TipitakaSinhalaLoader/1.0",
    "Accept":     "application/vnd.github+json",
})

_gh_token = os.environ.get("GITHUB_TOKEN", "")
if _gh_token:
    _gh_session.headers["Authorization"] = f"Bearer {_gh_token}"

_raw_session = requests.Session()
_raw_session.headers.update({"User-Agent": "TipitakaSinhalaLoader/1.0"})

# ═══════════════════════════════════════════════════════════════════════════════
# Sinhala volume mapping — adjust according to actual repo contents
# Most common: an1.json, mn1.json, kn-dhp.json etc. (no dash in nikaya numbers)
# ═══════════════════════════════════════════════════════════════════════════════
SI_FILE_MAP: dict[str, tuple[str, list[str]]] = {
    "dn": ("Digha Nikaya", [
        "dn1", "dn2", "dn3", "dn4", "dn5", "dn6", "dn7", "dn8", "dn9", "dn10",
        "dn11", "dn12", "dn13", "dn14", "dn15", "dn16", "dn17", "dn18", "dn19", "dn20",
        "dn21", "dn22", "dn23", "dn24", "dn25", "dn26", "dn27", "dn28", "dn29", "dn30",
        "dn31", "dn32", "dn33", "dn34",
    ]),
    "mn": ("Majjhima Nikaya", [
        "mn1", "mn2", "mn3", "mn4", "mn5", "mn6", "mn7", "mn8", "mn9", "mn10",
        "mn11", "mn12", "mn13", "mn14", "mn15", "mn16", "mn17", "mn18", "mn19", "mn20",
        # ... extend to mn152 as needed
    ]),
    "an": ("Anguttara Nikaya", [
        "an1", "an2", "an3", "an4", "an5", "an6", "an7", "an8", "an9", "an10", "an11",
    ]),
    "sn": ("Samyutta Nikaya", [
        "sn1", "sn2", "sn3", "sn4", "sn5", "sn6", "sn7", "sn8", "sn9", "sn10",
        # ... extend to sn56
    ]),
    "dhp": ("Dhammapada", ["kn-dhp"]),
    "snp": ("Sutta Nipata", ["kn-snp"]),
    "ud":  ("Udana", ["kn-ud"]),
    "iti": ("Itivuttaka", ["kn-iti"]),
    "thag": ("Theragatha", ["kn-thag"]),
    "thig": ("Therigatha", ["kn-thig"]),
    "kp":  ("Khuddakapatha", ["kn-kp"]),
    "cp":  ("Cariyapitaka", ["kn-cp"]),
    "ja":  ("Jataka", [
        "kn-ja-1", "kn-ja-2", "kn-ja-3", "kn-ja-4", "kn-ja-5", "kn-ja-6", "kn-ja-7",
    ]),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _get_raw(url: str) -> bytes | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _raw_session.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 90))
                tqdm.write(f"  [429] sleeping {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.content
        except Exception as exc:
            tqdm.write(f"  [raw error attempt {attempt}] {exc}")
            time.sleep(2 ** attempt)
    return None

def _gh_tree_paths(repo: str, branch: str, root_path: str) -> list[str]:
    ref_url = f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}"
    ref_data = _get_json(ref_url, _gh_session)
    if not ref_data:
        ref_url = f"{GITHUB_API}/repos/{repo}/branches/{branch}"
        ref_data = _get_json(ref_url, _gh_session)
        branch_sha = ref_data.get("commit", {}).get("sha") if ref_data else None
    else:
        branch_sha = ref_data.get("object", {}).get("sha")

    if not branch_sha:
        tqdm.write(f"  [error] Could not get branch SHA for {repo}/{branch}")
        return []

    tree_url = f"{GITHUB_API}/repos/{repo}/git/trees/{branch_sha}?recursive=1"
    tree_data = _get_json(tree_url, _gh_session)
    if not tree_data or not isinstance(tree_data, dict):
        return []

    prefix = root_path.rstrip("/") + "/"
    return [
        item["path"]
        for item in tree_data.get("tree", [])
        if item.get("type") == "blob" and item["path"].startswith(prefix)
    ]

def _get_json(url: str, session) -> dict | list | None:
    try:
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code in (403, 429):
            tqdm.write(f"  GitHub API limit hit ({resp.status_code})")
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        tqdm.write(f"  [json fetch error] {url} → {e}")
        return None

def _raw_url(repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"

def _list_tipitaka_lk_files() -> dict[str, str]:
    tqdm.write(f"  Listing files in {TIPITAKA_LK_REPO}/{TIPITAKA_LK_PATH}")
    paths = _gh_tree_paths(TIPITAKA_LK_REPO, TIPITAKA_LK_BRANCH, TIPITAKA_LK_PATH)
    result = {}
    for path in paths:
        if not path.endswith(".json"):
            continue
        filename = path.split("/")[-1]
        uid = filename[:-5]  # remove .json
        result[uid] = _raw_url(TIPITAKA_LK_REPO, TIPITAKA_LK_BRANCH, path)
    tqdm.write(f"  → Found {len(result)} JSON files")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Updated parser – handles tipitaka.lk page-based structure
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_tipitaka_lk_json(raw: bytes) -> list[str]:
    try:
        text = raw.decode("utf-8", errors="replace")
        data = json.loads(text)
    except Exception as e:
        tqdm.write(f"  [json decode/load error] {e}")
        return []

    segs = []

    def _collect_texts(obj):
        if isinstance(obj, dict):
            txt = obj.get("text", "").strip()
            if txt and len(txt) >= 15:
                segs.append(txt)
            for v in obj.values():
                _collect_texts(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect_texts(item)

    _collect_texts(data)

    # Remove exact duplicates while preserving first occurrence
    seen = set()
    unique = []
    for s in segs:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique

# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint / Save
# ═══════════════════════════════════════════════════════════════════════════════
def load_checkpoint() -> tuple[set[str], dict]:
    if not CHECKPOINT_PATH.exists():
        return set(), {}
    try:
        cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(cp.get("completed", [])), cp.get("stats", {})
    except:
        return set(), {}

def save_checkpoint(completed: set[str], stats: dict):
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({
            "completed": sorted(completed),
            "stats": stats,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(CHECKPOINT_PATH)

def load_section(key: str) -> list[str]:
    p = SECTIONS_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except:
            pass
    return []

def save_section(key: str, segments: list[str]):
    p = SECTIONS_DIR / f"{key}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(p)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 – Sinhala download
# ═══════════════════════════════════════════════════════════════════════════════
def phase4_sinhala():
    print("\n" + "═"*70)
    print(" PHASE 4 – Sinhala (Buddha Jayanthi Tipitaka) from tipitaka.lk")
    print("═"*70)

    completed, stats = load_checkpoint()
    file_map = _list_tipitaka_lk_files()

    if not file_map:
        print("  [ERROR] No files found. Probably GitHub rate limit.")
        print("  → Set GITHUB_TOKEN env var or wait 1 hour.")
        return

    for section, (label, expected) in SI_FILE_MAP.items():
        key = f"{section}-si"
        print(f"\n── {label}  [{key}]")

        pending = []
        for fname in expected:
            ck = f"si:{fname}"
            if ck in completed:
                continue
            if fname in file_map:
                pending.append(fname)
            else:
                # Try alternate naming style
                alt = fname.replace("-", "") if "-" in fname else f"{fname[:2]}-{fname[2:]}"
                if alt in file_map:
                    pending.append(alt)

        if not pending:
            print("  Already complete")
            continue

        segs = load_section(key)
        prev = stats.get(key, {"ok":0, "warn":0, "count":0})
        ok, warn = prev["ok"], prev["warn"]

        for fname in tqdm(pending, desc=label[:30], unit="file"):
            url = file_map.get(fname)
            if not url:
                tqdm.write(f"  [missing url] {fname}")
                continue

            raw = _get_raw(url)
            if not raw:
                tqdm.write(f"  [download failed] {fname}")
                warn += 1
                continue

            # Show sample of decoded content
            try:
                sample = raw.decode("utf-8", errors="replace")[:220].replace("\n", " ").strip()
                tqdm.write(f"  {fname}: {len(raw):,} bytes  |  {sample}…")
            except:
                tqdm.write(f"  {fname}: {len(raw):,} bytes (decode failed)")

            parsed = _parse_tipitaka_lk_json(raw)
            if parsed:
                segs.extend(parsed)
                ok += 1
                tqdm.write(f"    → extracted {len(parsed):,} segments  (total now {len(segs):,})")
            else:
                tqdm.write(f"    [warn] 0 segments from {fname}")
                warn += 1

            completed.add(f"si:{fname}")
            stats[key] = {"ok": ok, "warn": warn, "count": len(segs)}
            save_checkpoint(completed, stats)
            time.sleep(REQUEST_DELAY)

        save_section(key, segs)
        print(f"  Saved {len(segs):,} Sinhala segments to {key}.json")

    print("\nPhase 4 finished.")
def merge_all_sections_into_raw():
    """
    Combine ALL section files (both English and Sinhala) into one tipitaka_raw.json
    Each record will have a "language" field: "en" or "si"
    """
    print("\nMerging all sections into tipitaka_raw.json ...")

    all_records = []

    # 1. English sections (no -si suffix)
    english_sections = [
        "dn", "mn", "sn", "an", "kp", "dhp", "ud", "iti", "snp",
        "thag", "thig", "ja", "cp",
        # add vinaya/abhidhamma sections if you processed them earlier
    ]

    for sec in english_sections:
        p = SECTIONS_DIR / f"{sec}.json"
        if p.exists():
            try:
                chunks = json.loads(p.read_text(encoding="utf-8"))
                for chunk in chunks:
                    all_records.append({
                        "text": chunk,
                        "section": sec,
                        "language": "en"
                    })
                print(f"  Added {len(chunks):,} English segments from {sec}")
            except Exception as e:
                print(f"  [error] Failed to load {sec}.json → {e}")

    # 2. Sinhala sections (-si suffix)
    for sec in SI_FILE_MAP.keys():
        p = SECTIONS_DIR / f"{sec}-si.json"
        if p.exists():
            try:
                chunks = json.loads(p.read_text(encoding="utf-8"))
                for chunk in chunks:
                    all_records.append({
                        "text": chunk,
                        "section": sec,
                        "language": "si"
                    })
                print(f"  Added {len(chunks):,} Sinhala segments from {sec}-si")
            except Exception as e:
                print(f"  [error] Failed to load {sec}-si.json → {e}")

    # Save final combined file
    DATA_PATH = DATA_DIR / "tipitaka_raw.json"
    DATA_PATH.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    en_count = sum(1 for r in all_records if r["language"] == "en")
    si_count = len(all_records) - en_count

    print(f"\nMerge complete:")
    print(f"  Total records  : {len(all_records):,}")
    print(f"  English        : {en_count:,}")
    print(f"  Sinhala        : {si_count:,}")
    print(f"  Saved → {DATA_PATH}")
# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Starting Sinhala download phase...")
    phase4_sinhala()

    print("\n" + "═"*60)
    print("Merging English + Sinhala into tipitaka_raw.json")
    print("═"*60)
    merge_all_sections_into_raw()