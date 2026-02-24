import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CORPUS_PATH = DATA_DIR / "bible_raw.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_PATH = DATA_DIR / "faiss_index.bin"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500


# ─────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────

print("Loading corpus...")
corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# Simple chunking
# ─────────────────────────────────────────────

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
        if len(words[i:i + size]) > 20
    ]


print("Chunking...")
chunks = []

for item in corpus:
    text = item["text"] if isinstance(item, dict) else str(item)
    for chunk in chunk_text(text):
        chunks.append({"text": chunk})

print(f"Total chunks: {len(chunks)}")

CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
# Create embeddings
# ─────────────────────────────────────────────

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

texts = [c["text"] for c in chunks]

print("Generating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True)

np.save(EMBEDDINGS_PATH, embeddings)


# ─────────────────────────────────────────────
# Build FAISS index
# ─────────────────────────────────────────────

print("Building FAISS index...")

dimension = embeddings.shape[1]
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(FAISS_PATH))

print("Done ✅")