import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CORPUS_PATH      = DATA_DIR / "tipitaka_raw.json"
CHUNKS_PATH      = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH  = DATA_DIR / "embeddings.npy"

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────

CHUNK_SIZE   = 500
CHUNK_OVERLAP = 100

MODEL_NAME = "all-MiniLM-L6-v2"

# ────────────────────────────────────────────────────────────────
# Load Corpus
# ────────────────────────────────────────────────────────────────

print("Loading corpus...")
corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

# ────────────────────────────────────────────────────────────────
# Chunking
# ────────────────────────────────────────────────────────────────

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

    return chunks

print("Chunking corpus...")
all_chunks = []

for segment in tqdm(corpus):
    all_chunks.extend(chunk_text(segment))

print(f"Total chunks created: {len(all_chunks):,}")

CHUNKS_PATH.write_text(
    json.dumps(all_chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# ────────────────────────────────────────────────────────────────
# Embeddings
# ────────────────────────────────────────────────────────────────

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")
embeddings = model.encode(
    all_chunks,
    batch_size=64,
    show_progress_bar=True
)

np.save(EMBEDDINGS_PATH, embeddings)

