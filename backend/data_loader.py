import json
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_PATH       = DATA_DIR / "tipitaka_raw.json"
SECTIONS_DIR    = DATA_DIR / "sections"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

SC_SUTTAPLEX_API = "https://suttacentral.net/api/suttaplex"
SC_BILARA_API    = "https://suttacentral.net/api/bilarasuttas"
SC_LEGACY_API    = "https://suttacentral.net/api/suttas"

GITHUB_API       = "https://api.github.com"
BILARA_REPO      = "suttacentral/bilara-data"
BILARA_BRANCH    = "published"
SCDATA_REPO      = "suttacentral/sc-data"
SCDATA_BRANCH    = "master"

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP settings
# ═══════════════════════════════════════════════════════════════════════════════

REQUEST_DELAY = 0.35
TIMEOUT       = 15
MAX_RETRIES   = 3
SUJATO_AUTHOR = "sujato"

_sc_session = requests.Session()
_sc_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TipitakaLoader/2.0)"})

# GitHub REST API — sends Accept header so responses are JSON; counts against rate-limit
_gh_session = requests.Session()
_gh_session.headers.update({
    "User-Agent": "TipitakaLoader/2.0",
    "Accept":     "application/vnd.github+json",
})
_gh_token = os.environ.get("GITHUB_TOKEN", "")
if _gh_token:
    _gh_session.headers["Authorization"] = f"Bearer {_gh_token}"

_raw_session = requests.Session()
_raw_session.headers.update({"User-Agent": "TipitakaLoader/2.0"})

# ═══════════════════════════════════════════════════════════════════════════════
# Section configuration
# ═══════════════════════════════════════════════════════════════════════════════

# ── Phase 1: SuttaCentral API ────────────────────────────────────────────────
SC_API_SECTIONS: dict[str, int | None] = {
    "dn":   None, "mn":   None, "sn":  None, "an":  None,
    "kp":   None, "dhp":  None, "ud":  None, "iti": None,
    "snp":  None, "thag": None, "thig": None,
    "ja":   547,  "cp":  None,
}
SC_API_AUTHORS: dict[str, str] = {
    "dn":  "brahmali", "mn":  "bodhi",  "sn":  "bodhi",   "an":  "bodhi",
    "kp":  "sujato",   "dhp": "sujato", "ud":  "sujato",  "iti": "sujato",
    "snp": "sujato",   "thag":"sujato", "thig":"sujato",
    "ja":  "cowell",   "ap":  "walters","cp":  "horner",
}

# ── Phase 2: Vinaya via bilara-data GitHub ───────────────────────────────────
VINAYA_SECTIONS: dict[str, str] = {
    "pli-tv-bu-vb": "Bhikkhu Vibhanga",
    "pli-tv-bi-vb": "Bhikkhuni Vibhanga",
    "pli-tv-kd":    "Khandhaka",
    "pli-tv-pvr":   "Parivara",
    "pli-tv-bu-pm": "Bhikkhu Patimokkha",
    "pli-tv-bi-pm": "Bhikkhuni Patimokkha",
}
BILARA_VINAYA_PATHS: list[str] = [
    "translation/en/brahmali/vinaya",
    "translation/en/sujato/vinaya",
]

# ── Phase 3: Internet Archive sources ────────────────────────────────────────
#
# Each entry: section -> (label, [(identifier, djvu_filename), ...])
# Filenames are verified from IA metadata — exact names including spaces.
# Download URL: https://archive.org/download/<id>/<filename>_djvu.txt
# This returns raw plain text directly (unlike /stream/ which wraps in HTML).
#
IA_DL_BASE = "https://archive.org/download"

IA_SECTIONS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    # All identifiers and filenames confirmed from archive.org stream URLs.
    # Downloads use /download/<id>/<filename>_djvu.txt for raw plain text.

    "ds": ("Dhammasangani", [
        ("buddhistmanualof00davirich",
         "buddhistmanualof00davirich_djvu.txt"),
    ]),

    "vb": ("Vibhanga", [
        ("Vibhanga", "Vibhanga_a_djvu.txt"),
        ("Vibhanga", "Vibhanga_b_djvu.txt"),
    ]),

    "kv": ("Kathavatthu", [
        ("PointsOfControversyKathavatthu",
         "Points of Controversy (Kathavatthu)_djvu.txt"),
    ]),

    "mil": ("Milindapanha", [
        ("MilindaPanha-TheQuestionsOfKingMilinda-Part-1",
         "MilindaPanha-TheQuestionsOfKingMilindaByT.W.RhysDavids-Part-I_djvu.txt"),
        ("questionsofkingm02davi",
         "questionsofkingm02davi_djvu.txt"),
    ]),

    "dt": ("Dhatu-Katha", [
        ("DiscourseOnElementsDhatukatha",
         "Discourse on Elements (Dhatukatha)_djvu.txt"),
    ]),

    "pp": ("Patisambhidamagga", [
        ("dhatukatha-pts",
         "Patisambhidamagga_djvu.txt"),
    ]),

    "pv": ("Petavatthu", [
        ("in.ernet.dli.2015.282259",
         "2015.282259.The-Minor_djvu.txt"),
    ]),
    "vv": ("Vimanavatthu", [
        ("in.ernet.dli.2015.282259",
         "2015.282259.The-Minor_djvu.txt"),
    ]),

    "ps": ("Patthana", [
        ("myanmartipitakatEnglishtranslations",
         "06. Abhidhamma Pitaka 7. Conditional Relations (Patthana) Vol.1 (Tr. by U Narada, Mula Patthana Sayadaw, London-1969, PTS (322p) OCRed_djvu.txt"),
        ("myanmartipitakatEnglishtranslations",
         "06. Abhidhamma Pitaka 7. Conditional Relations (Patthana) Vol.2 (Tr. by U Narada, Mula Patthana Sayadaw, PTS 1981 (364p)) OCRed_djvu.txt"),
    ]),

    "ya": ("Ya (Puggalapannatti)", [
        ("puggalapannatti_202003",
         "Puggalapannatti_djvu.txt"),
    ]),

    # ja-ia: Jataka full 6-volume Cowell edition from IA
    # Used as fallback for JA UIDs that return {} HTML from SC API.
    # The SC API is tried first (per-sutta); this provides the bulk prose text.
    "ja-ia": ("Jataka (Cowell, IA supplement)", [
        ("complete-cowell-jataka-six-volumes-in-one",
         "complete-cowell-jataka-six-volumes-in-one_djvu.txt"),
    ]),
}

UNAVAILABLE_SECTIONS: dict[str, str] = {}
ATI_SECTIONS:     dict[str, tuple[str, list[str]]] = {}
LEGACY_SECTIONS:  dict[str, str]                   = {}
LEGACY_AUTHORS_FALLBACK: dict[str, list[str]]      = {}

# ── Human-readable labels ────────────────────────────────────────────────────
SECTION_LABELS: dict[str, str] = {
    "dn": "Digha Nikaya",      "mn":  "Majjhima Nikaya",
    "sn": "Samyutta Nikaya",   "an":  "Anguttara Nikaya",
    "kp": "Khuddakapatha",     "dhp": "Dhammapada",
    "ud": "Udana",             "iti": "Itivuttaka",
    "snp":"Sutta Nipata",      "thag":"Theragatha",
    "thig":"Therigatha",       "ja":  "Jataka",
    "ap": "Apadana",           "cp":  "Cariyapitaka",
    "ja-ia": "Jataka (IA supplement)",
    **VINAYA_SECTIONS,
    **UNAVAILABLE_SECTIONS,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Generic HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_json(url: str, session: requests.Session | None = None) -> dict | list | None:
    """GET a URL and return parsed JSON, with retries and rate-limit handling."""
    s = session or _sc_session
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = s.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            # GitHub rate-limit
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait  = max(reset - time.time() + 2, 5)
                tqdm.write(f"  [rate-limit] sleeping {wait:.0f}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            # Warn when GitHub quota is getting low (API calls only)
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) < 10:
                tqdm.write(f"  [warn] GitHub API quota low: {remaining} requests remaining")
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES:
                tqdm.write(f"  [timeout] {url}")
            else:
                time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code
            if 500 <= code < 600 and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  [http {code}] {url}")
                return None
        except Exception as exc:
            tqdm.write(f"  [error] {exc}")
            return None
    return None


def _get_raw(url: str) -> bytes | None:
    """
    Download a raw file (raw.githubusercontent.com or similar).
    Uses a plain session with no GitHub API headers so the request is NOT
    counted against the 60 req/hr unauthenticated API rate-limit.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _raw_session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                # raw.githubusercontent.com can occasionally throttle
                wait = int(resp.headers.get("Retry-After", 60))
                tqdm.write(f"  [raw throttle] sleeping {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.content
        except Exception as exc:
            tqdm.write(f"  [raw error attempt {attempt}] {exc}")
            time.sleep(2 ** attempt)
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# SuttaCentral API helpers  (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_uids(section: str, limit: int | None = None) -> list[str]:
    """Walk the suttaplex tree for a section and return all leaf UIDs."""
    data = _get_json(f"{SC_SUTTAPLEX_API}/{section}?lang=en")
    if not data:
        return []
    uids: list[str] = []
    def _walk(nodes):
        for item in nodes:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "leaf" and item.get("uid"):
                uids.append(item["uid"])
            elif item.get("children"):
                _walk(item["children"])
    _walk(data if isinstance(data, list) else [data])
    return uids[:limit] if limit else uids


def _fetch_bilara(uid: str, author: str = SUJATO_AUTHOR) -> dict | None:
    return _get_json(f"{SC_BILARA_API}/{uid}/{author}?lang=en")


def _fetch_legacy(uid: str, author: str) -> dict | None:
    return _get_json(f"{SC_LEGACY_API}/{uid}/{author}")


def _is_placeholder_segment(text: str) -> bool:
    """Return True if a segment is just an HTML template with {} placeholders."""
    stripped = text.strip()
    # Pure HTML tag lines with placeholders
    if stripped.startswith("<") and "{}" in stripped:
        return True
    # Segments where the majority of content is {} tokens
    placeholder_count = stripped.count("{}")
    if placeholder_count == 0:
        return False
    word_count = len(stripped.split())
    return placeholder_count >= max(1, word_count // 2)


def _extract_bilara(data: dict) -> list[str]:
    """Extract plain-text segments from a bilara API response."""
    if not isinstance(data, dict):
        return []
    translation = data.get("translation_text") or data.get("translationText") or {}
    if not translation:
        # Some endpoints nest the translation dict without a predictable key
        for v in data.values():
            if isinstance(v, dict) and len(v) > 5:
                translation = v
                break
    if not isinstance(translation, dict):
        return []
    return [
        s.strip() for s in translation.values()
        if isinstance(s, str) and s.strip()
        and not _is_placeholder_segment(s)
    ]


def _extract_legacy_html(data: dict) -> list[str]:
    """Strip HTML from a legacy API response and return text paragraphs."""
    if not isinstance(data, dict):
        return []
    trans = data.get("translation") or {}
    html  = trans.get("text", "") if isinstance(trans, dict) else ""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.name not in ("p", "li", "blockquote"):
            tag.unwrap()
    results = []
    for el in soup.find_all(["p", "li", "blockquote"]):
        text = el.get_text(" ").strip()
        if len(text) <= 20:
            continue
        if _is_placeholder_segment(text):
            continue
        results.append(text)
    return results


def get_segments_sc(uid: str, legacy_author: str | None = None) -> list[str]:
    """
    Try (in order):
      1. bilara API with sujato
      2. bilara API with legacy_author (if different)
      3. legacy HTML API with legacy_author
    """
    try:
        data = _fetch_bilara(uid)
        if data:
            segs = _extract_bilara(data)
            if segs:
                return segs
        if legacy_author and legacy_author != SUJATO_AUTHOR:
            data = _fetch_bilara(uid, legacy_author)
            if data:
                segs = _extract_bilara(data)
                if segs:
                    return segs
        if legacy_author:
            data = _fetch_legacy(uid, legacy_author)
            if data:
                segs = _extract_legacy_html(data)
                if segs:
                    return segs
    except Exception as exc:
        tqdm.write(f"  [error] {uid}: {exc}")
    return []

# ═══════════════════════════════════════════════════════════════════════════════
# GitHub API helpers  (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def _gh_tree_paths(repo: str, branch: str, root_path: str) -> list[str]:
    """
    Return all blob paths under root_path using a SINGLE git/trees?recursive=1
    call.  This costs exactly 2 GitHub API requests (one for the branch SHA,
    one for the tree) regardless of how many files exist — safe without a token.
    """
    # 1. Resolve the branch tip SHA
    ref_data = _get_json(
        f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}",
        session=_gh_session,
    )
    if not ref_data:
        # fallback: /branches endpoint
        ref_data = _get_json(
            f"{GITHUB_API}/repos/{repo}/branches/{branch}",
            session=_gh_session,
        )
        branch_sha = (ref_data or {}).get("commit", {}).get("sha") if ref_data else None
    else:
        branch_sha = ref_data.get("object", {}).get("sha")

    if not branch_sha:
        tqdm.write(f"  [warn] could not resolve SHA for {repo}/{branch}")
        return []

    # 2. Fetch the recursive tree in one request
    tree_data = _get_json(
        f"{GITHUB_API}/repos/{repo}/git/trees/{branch_sha}?recursive=1",
        session=_gh_session,
    )
    if not tree_data or not isinstance(tree_data, dict):
        return []

    if tree_data.get("truncated"):
        tqdm.write(f"  [warn] git tree truncated for {repo} — some files may be missed")

    prefix = root_path.rstrip("/") + "/"
    return [
        item["path"]
        for item in tree_data.get("tree", [])
        if item.get("type") == "blob" and item.get("path", "").startswith(prefix)
    ]


def _raw_url(repo: str, branch: str, path: str) -> str:
    """Build a raw.githubusercontent.com download URL from a repo path."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def list_bilara_translation_files(repo: str, branch: str, root_path: str) -> list[str]:
    """
    Return raw download URLs for every *_translation-en-*.json file under
    root_path.  Uses a single git/trees API call — no per-directory requests.
    """
    paths = _gh_tree_paths(repo, branch, root_path)
    return [
        _raw_url(repo, branch, p)
        for p in paths
        if "_translation-en-" in p and p.endswith(".json")
    ]


# Cache for sc-data legacy directory listings (avoids repeated API calls)
_scdata_dir_cache: dict[str, list[dict]] = {}

def _gh_contents(repo: str, path: str, branch: str) -> list | dict | None:
    """Fetch a GitHub Contents API listing, with a simple in-process cache."""
    cache_key = f"{repo}/{branch}/{path}"
    if cache_key in _scdata_dir_cache:
        return _scdata_dir_cache[cache_key]
    url    = f"{GITHUB_API}/repos/{repo}/contents/{path}?ref={branch}"
    result = _get_json(url, session=_gh_session)
    if isinstance(result, list):
        _scdata_dir_cache[cache_key] = result
    return result


def _parse_bilara_json(raw: bytes) -> list[str]:
    """Extract non-empty, non-numeric segment strings from a bilara JSON blob."""
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    segs = []
    for val in data.values():
        if not isinstance(val, str):
            continue
        text = val.strip()
        if text and not re.fullmatch(r"[\d\.\-]+", text) and not _is_placeholder_segment(text):
            segs.append(text)
    return segs


def _parse_legacy_html_bytes(raw: bytes) -> list[str]:
    """Strip HTML bytes into plain-text paragraphs."""
    try:
        html = raw.decode("utf-8", errors="replace")
    except Exception:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return [
        el.get_text(" ", strip=True)
        for el in soup.find_all(["p", "li", "blockquote", "dd", "h2", "h3", "h4"])
        if len(el.get_text(" ", strip=True)) > 20
        and not _is_placeholder_segment(el.get_text(" ", strip=True))
    ]


def _scdata_html_fallback(uid: str, section: str) -> list[str]:
    """
    Last-resort: look for <uid>.html in sc-data's legacy/en/<section>/
    or legacy/en/ directory and parse it.
    Directory listings are cached so repeated calls don't consume API quota.
    """
    for search_path in (f"legacy/en/{section}", "legacy/en"):
        listing = _gh_contents(SCDATA_REPO, search_path, SCDATA_BRANCH)
        if not listing or not isinstance(listing, list):
            continue
        for item in listing:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            if ((name.startswith(uid) or uid.replace("-", "_") in name)
                    and name.endswith(".html")):
                # Prefer the download_url from the API; fall back to constructing it
                raw_url = (
                    item.get("download_url")
                    or _raw_url(SCDATA_REPO, SCDATA_BRANCH, f"{search_path}/{name}")
                )
                raw = _get_raw(raw_url)
                if raw:
                    return _parse_legacy_html_bytes(raw)
    return []

# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint & section I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_replace(src: str, dst: str):
    """
    Atomically replace `dst` with `src`, working around Windows quirks:
      - PermissionError (WinError 5) from antivirus / UAC locking the target
      - os.replace() failing on some Windows FS configurations
    Falls back through three strategies before giving up.
    """
    src_p = Path(src)
    dst_p = Path(dst)
    # Strategy 1: standard atomic replace (works on Linux/macOS always)
    try:
        src_p.replace(dst_p)
        return
    except PermissionError:
        pass
    # Strategy 2: delete target first, then rename
    try:
        if dst_p.exists():
            dst_p.unlink()
        src_p.rename(dst_p)
        return
    except OSError:
        pass
    # Strategy 3: plain copy + remove (non-atomic but always works)
    try:
        dst_p.write_bytes(src_p.read_bytes())
        src_p.unlink(missing_ok=True)
    except Exception as exc:
        tqdm.write(f"  [warn] _safe_replace failed ({src} -> {dst}): {exc}")


def load_checkpoint() -> tuple[set[str], dict]:
    if not CHECKPOINT_PATH.exists():
        return set(), {}
    try:
        cp = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(cp.get("completed_uids", [])), cp.get("section_stats", {})
    except Exception:
        return set(), {}


def save_checkpoint(completed_uids: set[str], section_stats: dict):
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps({
            "completed_uids": sorted(completed_uids),
            "section_stats":  section_stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(CHECKPOINT_PATH))


def load_section(section: str) -> list[str]:
    p = SECTIONS_DIR / f"{section}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_section(section: str, segs: list[str]):
    p   = SECTIONS_DIR / f"{section}.json"
    tmp = str(p) + ".tmp"
    Path(tmp).write_text(
        json.dumps(segs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(p))


def merge_sections() -> list[dict]:
    all_sections = (
        list(SC_API_SECTIONS)
        + list(VINAYA_SECTIONS)
        + [s for s in IA_SECTIONS if s != "ja-ia"]  # ja-ia is merged into "ja"
    )
    all_texts: list[dict] = []
    for section in all_sections:
        segs = load_section(section)
        for seg in segs:
            all_texts.append({"text": seg, "section": section})
    tmp = str(DATA_PATH) + ".tmp"
    Path(tmp).write_text(
        json.dumps(all_texts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_replace(tmp, str(DATA_PATH))
    return all_texts


def load_all_texts() -> list[str]:
    """Load every segment from all section files already on disk."""
    all_texts: list[str] = []
    for p in SECTIONS_DIR.iterdir():
        if p.suffix == ".json":
            all_texts.extend(load_section(p.stem))
    return all_texts

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — SuttaCentral bilara / legacy API
# ═══════════════════════════════════════════════════════════════════════════════

def _phase1_sc_api(completed: set[str], stats: dict):
    print(f"\n{'═'*60}")
    print("  PHASE 1 — Nikayas, Khuddaka, Jataka  (SC API)")
    print(f"{'═'*60}")

    for section, limit in SC_API_SECTIONS.items():
        author = SC_API_AUTHORS.get(section)
        label  = SECTION_LABELS.get(section, section.upper())

        print(f"\n── {label} [{section}]")
        uids = discover_uids(section, limit=limit)
        if not uids:
            stats.setdefault(section, {"ok": 0, "warn": 0, "segments": 0})
            continue

        pending = [u for u in uids if u not in completed]
        if not pending:
            tqdm.write(f"  ✓ already complete ({len(uids)} UIDs)")
            continue

        section_segs = load_section(section)
        prev = stats.get(section, {"ok": 0, "warn": 0, "segments": 0})
        ok, warn = prev.get("ok", 0), prev.get("warn", 0)

        # Track how many JA UIDs got no text from SC API (need IA supplement)
        ja_empty_count = 0

        for uid in tqdm(pending, desc=label[:28], unit="sutta"):
            segs = get_segments_sc(uid, author)
            if segs:
                section_segs.extend(segs)
                ok += 1
            else:
                tqdm.write(f"  [warn] no text: {uid}")
                warn += 1
                if section == "ja":
                    ja_empty_count += 1
            completed.add(uid)
            stats[section] = {"ok": ok, "warn": warn, "segments": len(section_segs)}
            save_checkpoint(completed, stats)
            time.sleep(REQUEST_DELAY)

        save_section(section, section_segs)
        tqdm.write(f"  → {len(section_segs)} segments saved to sections/{section}.json")

        if section == "ja" and ja_empty_count > 0:
            tqdm.write(f"  [info] {ja_empty_count} JA UIDs returned no text from SC API")
            tqdm.write(f"  [info] Will supplement with full Cowell IA text in Phase 3")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1b — Jataka verses JA 80-547 via bilara-data GitHub
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sujato translated all 547 Jataka verse headers in bilara JSON format.
# The SC API returns these fine for JA 1-79 but the legacy Cowell prose
# bleeds through as {} HTML for JA 80+.  Going directly to the GitHub
# bilara-data repo guarantees clean verse text for every story.
#
# Path: translation/en/sujato/sutta/kn/ja/**/*_translation-en-sujato.json

BILARA_JA_PATH = "translation/en/sujato/sutta/kn/ja"

def _phase1b_jataka_github(completed: set[str], stats: dict):
    print(f"\n{'═'*60}")
    print("  PHASE 1b — Jataka verses  (bilara-data GitHub, all 547)")
    print(f"{'═'*60}")

    checkpoint_key = "github:ja-bilara"
    if checkpoint_key in completed:
        tqdm.write("  ✓ Jataka bilara GitHub already complete")
        return

    # Discover all JA bilara translation JSON URLs in one API call
    tqdm.write(f"  Listing {BILARA_REPO}/{BILARA_JA_PATH} …")
    urls = list_bilara_translation_files(BILARA_REPO, BILARA_BRANCH, BILARA_JA_PATH)
    tqdm.write(f"  → {len(urls)} JA translation files found")

    if not urls:
        tqdm.write("  [warn] No JA files found — check BILARA_JA_PATH or GitHub rate limit")
        return

    # Load existing ja segments (may already have JA 1-79 from Phase 1)
    section_segs = load_section("ja")
    # Track which UIDs we already have to avoid duplicates
    existing_uids: set[str] = set(
        stats.get("ja", {}).get("fetched_uids", [])
    )

    prev = stats.get("ja", {"ok": 0, "warn": 0, "segments": 0})
    ok   = prev.get("ok", 0)
    warn = prev.get("warn", 0)
    new_count = 0

    for url in tqdm(urls, desc="Jataka (GitHub)", unit="file"):
        # Derive UID from filename: ja1_translation-en-sujato.json → ja1
        fname = url.split("/")[-1]
        uid   = fname.split("_translation-")[0]

        # Skip if we already fetched this UID via Phase 1 SC API
        if uid in existing_uids or uid in completed:
            continue

        raw = _get_raw(url)
        if raw:
            segs = _parse_bilara_json(raw)
            if segs:
                section_segs.extend(segs)
                ok += 1
                new_count += 1
            else:
                tqdm.write(f"  [warn] empty: {uid}")
                warn += 1
        else:
            tqdm.write(f"  [warn] download failed: {uid}")
            warn += 1

        existing_uids.add(uid)
        stats["ja"] = {"ok": ok, "warn": warn, "segments": len(section_segs),
                       "fetched_uids": sorted(existing_uids)}
        save_checkpoint(completed, stats)

    save_section("ja", section_segs)
    completed.add(checkpoint_key)
    save_checkpoint(completed, stats)
    tqdm.write(f"  → {new_count} new files fetched, {len(section_segs)} total segments in ja")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Vinaya via bilara-data GitHub tree walk
# ═══════════════════════════════════════════════════════════════════════════════

def _phase2_vinaya_github(completed: set[str], stats: dict):
    print(f"\n{'═'*60}")
    print("  PHASE 2 — Vinaya  (bilara-data GitHub tree walk)")
    print(f"{'═'*60}")

    # Discover all translation file URLs across both translator paths
    all_urls: list[str] = []
    for path in BILARA_VINAYA_PATHS:
        tqdm.write(f"  Listing {BILARA_REPO}/{path} …")
        urls = list_bilara_translation_files(BILARA_REPO, BILARA_BRANCH, path)
        tqdm.write(f"    → {len(urls)} files found")
        all_urls.extend(urls)

    # Deduplicate by UID, preferring brahmali over sujato
    seen: dict[str, str] = {}
    for url in all_urls:
        fname = url.split("/")[-1]
        uid   = fname.split("_translation-")[0]
        if uid not in seen or "brahmali" in url:
            seen[uid] = url

    # Bucket URLs by Vinaya section prefix
    buckets: dict[str, list[str]] = {k: [] for k in VINAYA_SECTIONS}
    unmatched = 0
    for uid, url in seen.items():
        matched = False
        for prefix in VINAYA_SECTIONS:
            if uid.startswith(prefix):
                buckets[prefix].append(url)
                matched = True
                break
        if not matched:
            unmatched += 1
    if unmatched:
        tqdm.write(f"  [info] {unmatched} Vinaya UIDs did not match any section prefix")

    # Download and extract each section
    for prefix, label in VINAYA_SECTIONS.items():
        urls         = buckets[prefix]
        section_segs = load_section(prefix)
        prev         = stats.get(prefix, {"ok": 0, "warn": 0, "segments": 0})
        ok, warn     = prev.get("ok", 0), prev.get("warn", 0)
        pending      = [
            u for u in urls
            if u.split("/")[-1].split("_translation-")[0] not in completed
        ]

        print(f"\n── {label} [{prefix}] — {len(urls)} files, {len(pending)} pending")

        for url in tqdm(pending, desc=label[:28], unit="file"):
            uid = url.split("/")[-1].split("_translation-")[0]
            raw = _get_raw(url)
            if raw:
                segs = _parse_bilara_json(raw)
                if segs:
                    section_segs.extend(segs)
                    ok += 1
                else:
                    tqdm.write(f"  [warn] empty JSON: {uid}")
                    warn += 1
            else:
                tqdm.write(f"  [warn] download failed: {uid}")
                warn += 1
            completed.add(uid)
            stats[prefix] = {"ok": ok, "warn": warn, "segments": len(section_segs)}
            save_checkpoint(completed, stats)

        save_section(prefix, section_segs)
        tqdm.write(f"  → {len(section_segs)} segments saved to sections/{prefix}.json")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Internet Archive sources
# ═══════════════════════════════════════════════════════════════════════════════

def _ia_discover_djvu_filename(identifier: str) -> str | None:
    url  = f"https://archive.org/metadata/{identifier}/files"
    raw  = _get_raw(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    for f in (data.get("result") or []):
        name = f.get("name", "")
        if name.endswith("_djvu.txt"):
            return name
    return None


def _ia_download_djvu(identifier: str, filename: str) -> str | None:
    """
    Download a _djvu.txt from IA /download/ endpoint (returns raw plain text).
    If the hardcoded filename returns HTML/404, auto-discovers the real name
    via the IA metadata API and retries once.
    """
    from urllib.parse import quote

    def _try(fname: str) -> bytes | None:
        url = f"{IA_DL_BASE}/{identifier}/{quote(fname, safe='')}"
        tqdm.write(f"  Fetching: archive.org/download/{identifier}/{fname}")
        raw = _get_raw(url)
        if raw and not (raw[:200].lstrip().startswith(b'<!DOCTYPE')
                        or raw[:200].lstrip().startswith(b'<html')):
            return raw
        return None

    raw = _try(filename)
    if raw is None:
        tqdm.write(f"  [warn] hardcoded filename failed, querying IA metadata…")
        real = _ia_discover_djvu_filename(identifier)
        if real and real != filename:
            tqdm.write(f"  [info] found: {real}")
            raw = _try(real)
        elif real is None:
            # Try the uoft mirror as fallback for Milinda Part 2
            if "questionsofkingm02davi" in identifier:
                tqdm.write(f"  [info] trying uoft mirror…")
                alt_id = "questionsofkingm02daviuoft"
                real2 = _ia_discover_djvu_filename(alt_id)
                if real2:
                    url2 = f"{IA_DL_BASE}/{alt_id}/{quote(real2, safe='')}"
                    tqdm.write(f"  Fetching (uoft mirror): {url2}")
                    raw = _get_raw(url2)
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace")

def _clean_djvu_text(text: str) -> list[str]:
    """
    Extract readable paragraphs from IA plain OCR text.
    Filters page numbers, running headers, very short lines, ALL-CAPS titles.
    """
    segs = []
    for block in re.split(r'\n{2,}|\f', text):
        block = block.strip()
        if not block:
            continue
        if re.fullmatch(r'[\d\s\-\u2013\u2014\.\|]+', block):
            continue
        if len(block) < 40:
            continue
        if block == block.upper() and len(block) < 120:
            continue
        block = re.sub(r'[ \t]+', ' ', block)
        block = re.sub(r'\n', ' ', block)
        segs.append(block)
    return segs


def _phase3_legacy_extra(completed: set[str], stats: dict):
    print(f"\n{'═'*60}")
    print("  PHASE 3 — Abhidhamma + Misc  (Internet Archive)")
    print(f"{'═'*60}")

    for section, (label, identifiers) in IA_SECTIONS.items():
        checkpoint_key = f"ia:{section}"
        if checkpoint_key in completed:
            tqdm.write(f"  ✓  {label} [{section}] already complete")
            continue

        # ja-ia is a supplement: its text belongs in the "ja" section,
        # not in a separate file. This merges Cowell IA text directly into
        # sections/ja.json so all Jataka content is in one place.
        save_section_key = "ja" if section == "ja-ia" else section

        section_segs = load_section(save_section_key)
        prev         = stats.get(save_section_key, {"ok": 0, "warn": 0, "segments": 0})
        ok, warn     = prev.get("ok", 0), prev.get("warn", 0)

        print(f"\n── {label} [{section}]  ({len(identifiers)} volume(s))")
        tqdm.write(f"  Source: archive.org/download/{identifiers[0][0]}/...")
        if section == "ja-ia":
            tqdm.write(f"  [info] Cowell IA text will be merged into sections/ja.json")

        all_segs: list[str] = list(section_segs)
        all_ok = True

        for identifier, filename in identifiers:
            text = _ia_download_djvu(identifier, filename)
            if text:
                segs = _clean_djvu_text(text)
                tqdm.write(f"    → {len(segs)} segments from {identifier}")
                all_segs.extend(segs)
                ok += 1
            else:
                tqdm.write(f"  [warn] failed: {identifier}")
                warn += 1
                all_ok = False
            time.sleep(REQUEST_DELAY)

        save_section(save_section_key, all_segs)
        stats[save_section_key] = {"ok": ok, "warn": warn, "segments": len(all_segs)}

        if all_ok:
            completed.add(checkpoint_key)
        save_checkpoint(completed, stats)
        tqdm.write(f"  → {len(all_segs)} segments saved to sections/{section}.json")

# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def download_tipitaka() -> list[dict]:
    completed, stats = load_checkpoint()

    _phase1_sc_api(completed, stats)
    _phase1b_jataka_github(completed, stats)   # fills JA 80-547 from bilara-data
    _phase2_vinaya_github(completed, stats)
    _phase3_legacy_extra(completed, stats)

    print(f"\n  Merging all sections → {DATA_PATH} …")
    all_texts = merge_sections()

    total_ok   = sum(s.get("ok",   0) for s in stats.values())
    total_warn = sum(s.get("warn", 0) for s in stats.values())
    print(f"\n{'═'*60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  {len(all_texts):,} segments  |  {total_ok} ok  |  {total_warn} warnings")
    print(f"  Output: {DATA_PATH}")
    print(f"{'═'*60}")
    return all_texts


if __name__ == "__main__":
    download_tipitaka()