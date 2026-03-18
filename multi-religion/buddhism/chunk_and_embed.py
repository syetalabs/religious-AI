import json
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from huggingface_hub import HfApi, login, create_repo, upload_file

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CORPUS_PATH     = DATA_DIR / "tipitaka_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks-en-si.json"
CHUNKS_DB_PATH  = DATA_DIR / "chunks-en-si.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings-en-si.npy"
FAISS_PATH      = DATA_DIR / "faiss_index-en-si.bin"

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
MODEL_NAME    = "all-MiniLM-L6-v2"          # good multilingual baseline

# You can experiment with better multilingual models later, e.g.:
# "intfloat/multilingual-e5-large-instruct"
# "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# "intfloat/multilingual-e5-small"

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
    "cp": "Sutta Pitaka",
    "pli-tv-bu-vb": "Vinaya Pitaka", "pli-tv-bi-vb": "Vinaya Pitaka",
    "pli-tv-kd":    "Vinaya Pitaka", "pli-tv-pvr":   "Vinaya Pitaka",
    "pli-tv-bu-pm": "Vinaya Pitaka", "pli-tv-bi-pm": "Vinaya Pitaka",
    "ds": "Abhidhamma Pitaka", "vb": "Abhidhamma Pitaka",
    "dt": "Abhidhamma Pitaka", "kv": "Abhidhamma Pitaka",
    "pp": "Abhidhamma Pitaka", "ps": "Abhidhamma Pitaka",
    "ya": "Abhidhamma Pitaka",
}

# ────────────────────────────────────────────────────────────────
# Load corpus (contains both en and si)
# ────────────────────────────────────────────────────────────────
print("Loading corpus...")
corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(corpus):,} entries")

# ────────────────────────────────────────────────────────────────
# Chunking function
# ────────────────────────────────────────────────────────────────
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(i + size, len(words))
        chunk = " ".join(words[i:end])
        if len(chunk) > 60:   # avoid very short junk
            chunks.append(chunk)
        i += size - overlap
    return chunks

# ────────────────────────────────────────────────────────────────
# Create chunks with language-aware metadata
# ────────────────────────────────────────────────────────────────
print("Chunking corpus (English + Sinhala)...")
all_chunks = []

for item in tqdm(corpus, desc="Chunking"):
    if not isinstance(item, dict):
        continue

    text = item.get("text", "").strip()
    if not text:
        continue

    lang     = item.get("language", "en")           # "en" or "si"
    section  = item.get("section", "unknown")
    book     = SECTION_LABELS.get(section, section)
    pitaka   = PITAKA_MAP.get(section, "Unknown")

    for chunk in chunk_text(text):
        all_chunks.append({
            "text":     chunk,
            "source":   section,        # e.g. "mn", "an-si"
            "book":     book,
            "pitaka":   pitaka,
            "religion": "Buddhism",
            "language": lang,           # ← crucial field
            "source_url": (
                f"https://suttacentral.net/{section}"
                if lang == "en" else
                "https://tipitaka.lk"   # or more precise if you have it
            ),
        })

print(f"Total chunks created: {len(all_chunks):,}")

# Optional: save human-readable backup
CHUNKS_PATH.write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"Saved chunks.json → {CHUNKS_PATH}")

# ────────────────────────────────────────────────────────────────
# Build SQLite database (fast metadata lookup)
# ────────────────────────────────────────────────────────────────
print("\nBuilding chunks.db ...")
if CHUNKS_DB_PATH.exists():
    CHUNKS_DB_PATH.unlink()  # fresh start

con = sqlite3.connect(CHUNKS_DB_PATH)
con.executescript("""
    CREATE TABLE chunks (
        id       INTEGER PRIMARY KEY,
        text     TEXT    NOT NULL,
        book     TEXT    NOT NULL DEFAULT '',
        pitaka   TEXT    NOT NULL DEFAULT '',
        source   TEXT    NOT NULL DEFAULT '',
        religion TEXT    NOT NULL DEFAULT 'Buddhism',
        language TEXT    NOT NULL DEFAULT 'en'
    );
    CREATE INDEX idx_lang_religion ON chunks (language, religion);
    CREATE INDEX idx_book            ON chunks (book);
    CREATE INDEX idx_pitaka          ON chunks (pitaka);
""")

con.executemany(
    """
    INSERT INTO chunks (id, text, book, pitaka, source, religion, language)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    [
        (
            i,
            c["text"],
            c["book"],
            c["pitaka"],
            c["source"],
            c["religion"],
            c["language"],
        )
        for i, c in enumerate(all_chunks)
    ]
)
con.commit()
con.close()

db_mb = CHUNKS_DB_PATH.stat().st_size / (1024*1024)
print(f"chunks.db saved → {CHUNKS_DB_PATH}  ({db_mb:.1f} MiB)")

# ────────────────────────────────────────────────────────────────
# Embed with multilingual-aware model
# ────────────────────────────────────────────────────────────────
print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding chunks...")
texts = [c["text"] for c in all_chunks]

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # important for cosine / IP
    convert_to_numpy=True,
)

np.save(EMBEDDINGS_PATH, embeddings)
print(f"Embeddings saved → {EMBEDDINGS_PATH}")
print(f"Shape: {embeddings.shape}")

# ────────────────────────────────────────────────────────────────
# FAISS Index (Flat Inner Product = cosine after L2 norm)
# ────────────────────────────────────────────────────────────────
print("\nBuilding FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(FAISS_PATH))
print(f"FAISS index saved → {FAISS_PATH}  ({index.ntotal:,} vectors)")

# ────────────────────────────────────────────────────────────────
# Optional: Upload to Hugging Face
# ────────────────────────────────────────────────────────────────
def upload_to_hf():
    print("\nUploading to Hugging Face...")

    # 1. Login (run once or use token)
    # You can also set HF_TOKEN env var instead
    # login()   # interactive login — comment out if using token

    repo_id = "sathmi/tipitaka-english-sinhala-chunks-faiss"   # ← CHANGE THIS

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id)
        print(f"Repo already exists: {repo_id}")
    except:
        create_repo(repo_id, private=False, exist_ok=True)
        print(f"Created repo: {repo_id}")

    # Upload the two most important runtime files
    files_to_upload = [
        (CHUNKS_DB_PATH,  "chunks.db"),
        (FAISS_PATH,      "faiss_index.bin"),
        # (CHUNKS_PATH,   "chunks.json"),      # optional – large
        # (EMBEDDINGS_PATH, "embeddings.npy"), # optional – very large
    ]

    for local_path, repo_path in files_to_upload:
        print(f"  Uploading {local_path.name} → {repo_path}")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add chunks.db + FAISS index (en+si)"
        )

    print(f"\nUpload finished.")
    print(f"Repo: https://huggingface.co/{repo_id}")
    print("Don't forget to write a good README.md !")

# Uncomment to upload automatically (or run manually)
# upload_to_hf()

# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("Processing completed. Files created:")
for p, note in [
    (CHUNKS_DB_PATH,  "← upload this + faiss_index.bin to HF"),
    (FAISS_PATH,      "← upload this"),
    (EMBEDDINGS_PATH, "backup – optional"),
    (CHUNKS_PATH,     "human readable backup – optional"),
]:
    mb = p.stat().st_size / (1024*1024)
    print(f"  {p.name:<20} {mb:>6.1f} MiB   {note}")

en_count = sum(1 for c in all_chunks if c["language"] == "en")
si_count = len(all_chunks) - en_count

print(f"\nLanguage split:")
print(f"  English chunks : {en_count:,}")
print(f"  Sinhala  chunks : {si_count:,}")
print("─"*60)