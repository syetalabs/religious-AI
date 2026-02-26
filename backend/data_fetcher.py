import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

HF_REPO_ID   = "sdevr/religious-ai-data"
HF_SUBFOLDER = "buddhism"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
CACHE_DIR    = Path("/tmp/religious-ai-data/buddhism")
FILES        = ["faiss_index.bin", "chunks.db"]

def ensure_data_files() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Debug: list all files visible in the repo ──
    print("  [debug] Files found in HF repo:")
    try:
        for f in list_repo_files(HF_REPO_ID, repo_type="dataset", token=HF_TOKEN or None):
            print(f"    {f}")
    except Exception as e:
        print(f"    ERROR listing repo: {e}")

    for filename in FILES:
        dest = CACHE_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [cache] {filename} already in {CACHE_DIR}")
            continue

        hf_path = f"{HF_SUBFOLDER}/{filename}"
        print(f"  [download] Fetching {hf_path} ...")
        try:
            tmp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                repo_type="dataset",
                token=HF_TOKEN or None,
                cache_dir=str(CACHE_DIR / ".hf_cache"),
            )
            print(f"  [download] Downloaded to tmp path: {tmp}")
            print(f"  [download] tmp exists: {Path(tmp).exists()}, size: {Path(tmp).stat().st_size / 1_048_576:.1f} MB")

            shutil.copy2(tmp, dest)
            print(f"  [download] Copied to: {dest}, exists: {dest.exists()}")

        except Exception as e:
            print(f"  [ERROR] Failed to download {hf_path}: {e}")

    return CACHE_DIR