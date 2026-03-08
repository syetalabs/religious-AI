"""
data_fetcher.py  —  Buddhism + Christianity
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
        ("buddhism/faiss_index-en-si.bin", "faiss_index-en-si.bin"),
        ("buddhism/chunks-en-si.db",       "chunks-en-si.db"),
    ],
    "christianity": [
        ("christianity/faiss_index.bin", "faiss_index.bin"),
        ("christianity/chunks.db",       "chunks.db"),
    ],
}

# Old filenames that must be removed so retrieve.py never opens them
_STALE_FILES = {
    "buddhism": ["faiss_index.bin", "chunks.db"],
}


def _purge_stale(religion: str) -> None:
    dest_dir = DATA_ROOT / religion
    for name in _STALE_FILES.get(religion, []):
        stale = dest_dir / name
        if stale.exists():
            stale.unlink()
            print(f"  [data_fetcher] Removed stale file: {stale}")


def _download_file(repo_path: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [data_fetcher] Already exists: {dest.name} ({dest.stat().st_size // 1024:,} KB)")
        return
    print(f"  [data_fetcher] Downloading {repo_path} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    tmp = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=repo_path,
        cache_dir=str(DATA_ROOT / ".hf_cache"),
    )
    shutil.copy2(tmp, dest)
    print(f"  [data_fetcher] Saved {dest.name} ({dest.stat().st_size // 1024:,} KB)")


def ensure_data_files(religions=None) -> None:
    if religions is None:
        religions = list(_FILES.keys())
    for religion in religions:
        _purge_stale(religion)
        dest_dir = DATA_ROOT / religion
        dest_dir.mkdir(parents=True, exist_ok=True)
        for repo_path, local_name in _FILES[religion]:
            _download_file(repo_path, dest_dir / local_name)
    print("  [data_fetcher] All data files ready.")


if __name__ == "__main__":
    ensure_data_files()