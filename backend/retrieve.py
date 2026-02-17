import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "backend" / "data"

CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBED_PATH = DATA_DIR / "embeddings.npy"

TOP_K = 5

# ────────────────────────────────────────────────────────────────
# Loading chunks and embeddings
# ────────────────────────────────────────────────────────────────
print("Loading data...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(EMBED_PATH)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ────────────────────────────────────────────────────────────────
# Search function
# ────────────────────────────────────────────────────────────────
def search(query, top_k=TOP_K):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["book"],
            "score": float(scores[idx])
        })

    return results

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        results = search(q)

        for r in results:
            print("\n---")
            print("Source:", r["source"])
            print("Score:", r["score"])
            print(r["text"])