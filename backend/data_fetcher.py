"""
data_fetcher.py  —  Buddhism + Christianity + Hinduism
Downloads FAISS index and chunks DB from HuggingFace for each religion.
Deletes any stale old-named files so they are never loaded by mistake.
"""
import shutil
from pathlib import Path

HF_REPO_ID   = "sdevr/religious-ai-data"
HF_REPO_TYPE = "dataset"
DATA_ROOT    = Path("/tmp/religious-ai-data")

_FILES = {
    "buddhism": [
        ("buddhism/faiss_index-en-si.bin",      "faiss_index-en-si.bin", False),
        ("buddhism/chunks-en-si.db",            "chunks-en-si.db",       False),
    ],
    "christianity": [
        ("christianity/faiss_index-en-si-ta.bin", "faiss_index-en-si-ta.bin", False),
        ("christianity/chunks-en-si-ta.db",       "chunks-en-si-ta.db",       False),
    ],
    "hinduism": [
        ("hinduism/faiss_index.bin", "faiss_index.bin", False),
        ("hinduism/chunks.db",       "chunks.db",       False),
    ],
    "islam": [
        ("islam/faiss_index-en-si-ta.bin", "faiss_index-en-si-ta.bin", False),
        ("islam/chunks-en-si-ta.db",       "chunks-en-si-ta.db",       False),
    ],
}

# Old filenames that must be removed so retrieve.py never opens them
_STALE_FILES = {
    "buddhism":     ["faiss_index.bin", "chunks.db"],
    "christianity": ["faiss_index.bin", "chunks.db"],   # replaced by en-si-ta files
    "hinduism":     [],
    "islam":        ["faiss_index.bin", "chunks.db"],
}


def _purge_stale(religion: str) -> None:
    dest_dir = DATA_ROOT / religion
    for name in _STALE_FILES.get(religion, []):
        stale = dest_dir / name
        if stale.exists():
            stale.unlink()
            print(f"  [data_fetcher] Removed stale file: {stale}")


def _download_file(repo_path: str, dest: Path, optional: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [data_fetcher] Already exists: {dest.name} ({dest.stat().st_size // 1024:,} KB)")
        return
    print(f"  [data_fetcher] Downloading {repo_path} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    try:
        tmp = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=repo_path,
            cache_dir=str(DATA_ROOT / ".hf_cache"),
        )
    except Exception as exc:
        if optional:
            print(f"  [data_fetcher] Optional file not available, skipping: {repo_path} ({exc})")
            return
        raise
    shutil.copy2(tmp, dest)
    print(f"  [data_fetcher] Saved {dest.name} ({dest.stat().st_size // 1024:,} KB)")


def ensure_data_files(religions=None) -> None:
    if religions is None:
        religions = list(_FILES.keys())
    for religion in religions:
        key = religion.lower()   # normalise "Buddhism" → "buddhism"
        _purge_stale(key)
        dest_dir = DATA_ROOT / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        for repo_path, local_name, optional in _FILES[key]:
            _download_file(repo_path, dest_dir / local_name, optional=optional)
    print("  [data_fetcher] All data files ready.")


if __name__ == "__main__":
    ensure_data_files()