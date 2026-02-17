import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"

CORPUS_PATH     = DATA_DIR / "tipitaka_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_PATH      = DATA_DIR / "faiss_index.bin"

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
MODEL_NAME    = "all-MiniLM-L6-v2"

# ────────────────────────────────────────────────────────────────
# Metadata maps
# ────────────────────────────────────────────────────────────────

SECTION_LABELS = {
    "dn":           "Digha Nikaya",
    "mn":           "Majjhima Nikaya",
    "sn":           "Samyutta Nikaya",
    "an":           "Anguttara Nikaya",
    "kp":           "Khuddakapatha",
    "dhp":          "Dhammapada",
    "ud":           "Udana",
    "iti":          "Itivuttaka",
    "snp":          "Sutta Nipata",
    "thag":         "Theragatha",
    "thig":         "Therigatha",
    "ja":           "Jataka",
    "cp":           "Cariyapitaka",
    "pli-tv-bu-vb": "Vinaya: Bhikkhu Vibhanga",
    "pli-tv-bi-vb": "Vinaya: Bhikkhuni Vibhanga",
    "pli-tv-kd":    "Vinaya: Khandhaka",
    "pli-tv-pvr":   "Vinaya: Parivara",
    "pli-tv-bu-pm": "Vinaya: Bhikkhu Patimokkha",
    "pli-tv-bi-pm": "Vinaya: Bhikkhuni Patimokkha",
    "ds":           "Abhidhamma: Dhammasangani",
    "vb":           "Abhidhamma: Vibhanga",
    "dt":           "Abhidhamma: Dhatu-Katha",
    "kv":           "Abhidhamma: Kathavatthu",
    "pp":           "Abhidhamma: Patisambhidamagga",
    "ps":           "Abhidhamma: Patthana",
    "ya":           "Abhidhamma: Puggalapannatti",
    "mil":          "Milindapanha",
    "pv":           "Petavatthu",
    "vv":           "Vimanavatthu",
}

PITAKA_MAP = {
    "dn": "Sutta Pitaka",  "mn": "Sutta Pitaka",  "sn": "Sutta Pitaka",
    "an": "Sutta Pitaka",  "kp": "Sutta Pitaka",  "dhp": "Sutta Pitaka",
    "ud": "Sutta Pitaka",  "iti": "Sutta Pitaka", "snp": "Sutta Pitaka",
    "thag": "Sutta Pitaka","thig": "Sutta Pitaka","ja": "Sutta Pitaka",
    "cp": "Sutta Pitaka",  "mil": "Sutta Pitaka", "pv": "Sutta Pitaka",
    "vv": "Sutta Pitaka",
    "pli-tv-bu-vb": "Vinaya Pitaka", "pli-tv-bi-vb": "Vinaya Pitaka",
    "pli-tv-kd":    "Vinaya Pitaka", "pli-tv-pvr":   "Vinaya Pitaka",
    "pli-tv-bu-pm": "Vinaya Pitaka", "pli-tv-bi-pm": "Vinaya Pitaka",
    "ds": "Abhidhamma Pitaka", "vb": "Abhidhamma Pitaka",
    "dt": "Abhidhamma Pitaka", "kv": "Abhidhamma Pitaka",
    "pp": "Abhidhamma Pitaka", "ps": "Abhidhamma Pitaka",
    "ya": "Abhidhamma Pitaka",
}

# ────────────────────────────────────────────────────────────────
# Load corpus
# ────────────────────────────────────────────────────────────────

print("Loading corpus...")
corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(corpus):,} entries")

# ────────────────────────────────────────────────────────────────
# Chunking
# ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

print("Chunking corpus with metadata...")
all_chunks = []

for item in tqdm(corpus, desc="Chunking"):
    if isinstance(item, dict):
        text    = item.get("text", "")
        section = item.get("section") or item.get("id") or "unknown"
    else:
        text    = str(item)
        section = "unknown"

    if not text.strip():
        continue

    book    = SECTION_LABELS.get(section, section)
    pitaka  = PITAKA_MAP.get(section, "Unknown")

    for chunk in chunk_text(text):
        all_chunks.append({
            "text":     chunk,
            "source":   section,        # e.g. "mn"
            "book":     book,           # e.g. "Majjhima Nikaya"
            "pitaka":   pitaka,         # e.g. "Sutta Pitaka"
            "religion": "Buddhism",
            "language": "en",           # English translations throughout
            "source_url": f"https://suttacentral.net/{section}",
        })

print(f"Total chunks created: {len(all_chunks):,}")
CHUNKS_PATH.write_text(
    json.dumps(all_chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print(f"Saved → {CHUNKS_PATH}")

# ────────────────────────────────────────────────────────────────
# Embeddings
# ────────────────────────────────────────────────────────────────

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")
texts_only = [c["text"] for c in all_chunks]

embeddings = model.encode(
    texts_only,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

# Save raw numpy (useful for inspection / reloading)
np.save(EMBEDDINGS_PATH, embeddings)
print(f"Embeddings saved → {EMBEDDINGS_PATH}")
print(f"Embedding shape : {embeddings.shape}")

# ────────────────────────────────────────────────────────────────
# Build FAISS index
# ────────────────────────────────────────────────────────────────

print("Building FAISS index...")

dimension = embeddings.shape[1]

# IndexFlatIP = exact inner-product search on L2-normalised vectors
# which is equivalent to cosine similarity — no approximation errors
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(FAISS_PATH))
print(f"FAISS index saved → {FAISS_PATH}  ({index.ntotal:,} vectors)")
print("Done.")