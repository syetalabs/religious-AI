import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO_ID = "sdevr/religious-ai-data/buddhism/"  # update this
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
CACHE_DIR  = Path("/tmp/religious-ai-data")
FILES      = ["faiss_index.bin", "chunks.db"]

def ensure_data_files() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        dest = CACHE_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [cache] {filename} already in /tmp")
            continue
        print(f"  [download] Fetching {filename} from Hugging Face...")
        tmp = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN or None,
            cache_dir=str(CACHE_DIR / ".hf_cache"),
        )
        shutil.copy2(tmp, dest)
        mb = dest.stat().st_size / 1_048_576
        print(f"  [download] {filename} ready ({mb:.1f} MB)")
    return CACHE_DIR