import json
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"

CORPUS_PATH     = DATA_DIR / "bible_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks.json"
CHUNKS_DB_PATH  = DATA_DIR / "chunks.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_PATH      = DATA_DIR / "faiss_index.bin"

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 600    # slightly smaller than Tipitaka — verses are shorter units
CHUNK_OVERLAP = 100
MODEL_NAME    = "all-MiniLM-L6-v2"

# ────────────────────────────────────────────────────────────────
# Metadata maps
# ────────────────────────────────────────────────────────────────

OLD_TESTAMENT_BOOKS = {
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi",
}

NEW_TESTAMENT_BOOKS = {
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation",
}

# Genre groupings for richer metadata
GENRE_MAP: dict[str, str] = {
    # Pentateuch
    "Genesis": "Law", "Exodus": "Law", "Leviticus": "Law",
    "Numbers": "Law", "Deuteronomy": "Law",
    # History
    "Joshua": "History", "Judges": "History", "Ruth": "History",
    "1 Samuel": "History", "2 Samuel": "History",
    "1 Kings": "History", "2 Kings": "History",
    "1 Chronicles": "History", "2 Chronicles": "History",
    "Ezra": "History", "Nehemiah": "History", "Esther": "History",
    # Wisdom
    "Job": "Wisdom", "Psalms": "Wisdom", "Proverbs": "Wisdom",
    "Ecclesiastes": "Wisdom", "Song of Solomon": "Wisdom",
    # Major Prophets
    "Isaiah": "Major Prophets", "Jeremiah": "Major Prophets",
    "Lamentations": "Major Prophets", "Ezekiel": "Major Prophets",
    "Daniel": "Major Prophets",
    # Minor Prophets
    "Hosea": "Minor Prophets", "Joel": "Minor Prophets",
    "Amos": "Minor Prophets", "Obadiah": "Minor Prophets",
    "Jonah": "Minor Prophets", "Micah": "Minor Prophets",
    "Nahum": "Minor Prophets", "Habakkuk": "Minor Prophets",
    "Zephaniah": "Minor Prophets", "Haggai": "Minor Prophets",
    "Zechariah": "Minor Prophets", "Malachi": "Minor Prophets",
    # Gospels & Acts
    "Matthew": "Gospels", "Mark": "Gospels",
    "Luke": "Gospels",   "John": "Gospels",
    "Acts": "Acts",
    # Pauline Epistles
    "Romans": "Pauline Epistles", "1 Corinthians": "Pauline Epistles",
    "2 Corinthians": "Pauline Epistles", "Galatians": "Pauline Epistles",
    "Ephesians": "Pauline Epistles", "Philippians": "Pauline Epistles",
    "Colossians": "Pauline Epistles", "1 Thessalonians": "Pauline Epistles",
    "2 Thessalonians": "Pauline Epistles", "1 Timothy": "Pauline Epistles",
    "2 Timothy": "Pauline Epistles", "Titus": "Pauline Epistles",
    "Philemon": "Pauline Epistles",
    # General Epistles
    "Hebrews": "General Epistles", "James": "General Epistles",
    "1 Peter": "General Epistles", "2 Peter": "General Epistles",
    "1 John": "General Epistles", "2 John": "General Epistles",
    "3 John": "General Epistles", "Jude": "General Epistles",
    # Apocalyptic
    "Revelation": "Apocalyptic",
}

# ─────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────

print("Loading corpus...")
corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(corpus):,} entries")

# ─────────────────────────────────────────────
# chunking
# ─────────────────────────────────────────────

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > 40:
            chunks.append(chunk)
    return chunks


print("Chunking corpus with metadata...")
all_chunks = []

for item in tqdm(corpus, desc="Chunking"):
    if isinstance(item, dict):
        text      = item.get("text", "")
        section   = item.get("section") or item.get("id") or "unknown"  # book name e.g. "John"
        testament = item.get("testament", "Unknown")
    else:
        text      = str(item)
        section   = "unknown"
        testament = "Unknown"

    if not text.strip():
        continue

    # Derive testament from book name if not supplied in corpus
    if testament == "Unknown":
        if section in OLD_TESTAMENT_BOOKS:
            testament = "Old Testament"
        elif section in NEW_TESTAMENT_BOOKS:
            testament = "New Testament"

    genre = GENRE_MAP.get(section, "General")

    for chunk in chunk_text(text):
        all_chunks.append({
            "text":      chunk,
            "source":    section,       # e.g. "John"
            "book":      section,       # kept same for API compatibility with retrieve.py
            "testament": testament,     # "Old Testament" | "New Testament"
            "genre":     genre,         # "Gospels" | "Pauline Epistles" | "Wisdom" …
            "pitaka":    testament,     # alias — retrieve.py uses "pitaka" field for display
            "religion":  "Christianity",
            "language":  "en",
            "source_url": f"https://www.biblegateway.com/passage/?search={section.replace(' ', '+')}&version=KJV",
        })

print(f"Total chunks created: {len(all_chunks):,}")

# Save chunks.json (kept as a backup — not used at runtime)
CHUNKS_PATH.write_text(
    json.dumps(all_chunks, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"Saved → {CHUNKS_PATH}")

# ────────────────────────────────────────────────────────────────
# SQLite — runtime chunk store
# ────────────────────────────────────────────────────────────────
# chunks.db replaces chunks.json at runtime.
# Row id matches the FAISS vector index exactly (0, 1, 2, …)
# so retrieve.py can do a direct lookup with no scanning.

print("\nBuilding chunks.db ...")

if CHUNKS_DB_PATH.exists():
    CHUNKS_DB_PATH.unlink()   # always rebuild fresh alongside new embeddings

con = sqlite3.connect(str(CHUNKS_DB_PATH))
con.executescript("""
    CREATE TABLE chunks (
        id        INTEGER PRIMARY KEY,
        text      TEXT    NOT NULL,
        book      TEXT    NOT NULL DEFAULT '',
        testament TEXT    NOT NULL DEFAULT '',
        genre     TEXT    NOT NULL DEFAULT '',
        source    TEXT    NOT NULL DEFAULT '',
        religion  TEXT    NOT NULL DEFAULT 'Christianity',
        language  TEXT    NOT NULL DEFAULT 'en'
    );

    -- Covers the religion + language filter used in every search query
    CREATE INDEX idx_religion_language ON chunks (religion, language);
""")

con.executemany(
    "INSERT INTO chunks (id, text, book, testament, genre, source, religion, language) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    [
        (
            i,
            c["text"],
            c.get("book",      ""),
            c.get("testament", ""),
            c.get("genre",     ""),
            c.get("source",    ""),
            c.get("religion",  "Christianity"),
            c.get("language",  "en"),
        )
        for i, c in enumerate(all_chunks)
    ],
)

con.commit()
con.close()

db_mb = CHUNKS_DB_PATH.stat().st_size / 1_048_576
print(f"chunks.db saved → {CHUNKS_DB_PATH}  ({db_mb:.1f} MB)")

# ────────────────────────────────────────────────────────────────
# Embeddings
# ────────────────────────────────────────────────────────────────

print("\nLoading embedding model...")
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
print(f"Embedding shape  : {embeddings.shape}")


# ─────────────────────────────────────────────
# Build FAISS index
# ─────────────────────────────────────────────

print("\nBuilding FAISS index...")

dimension = embeddings.shape[1]

# IndexFlatIP = exact inner-product search on L2-normalised vectors
# which is equivalent to cosine similarity — no approximation errors
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(FAISS_PATH))
print(f"FAISS index saved → {FAISS_PATH}  ({index.ntotal:,} vectors)")

# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
print("Done. Files produced:")
for path, note in [
    (CHUNKS_DB_PATH,  "← upload to Hugging Face  (runtime)"),
    (FAISS_PATH,      "← upload to Hugging Face  (runtime)"),
    (EMBEDDINGS_PATH, "← keep locally as backup  (not needed at runtime)"),
    (CHUNKS_PATH,     "← keep locally as backup  (not needed at runtime)"),
]:
    mb = path.stat().st_size / 1_048_576
    print(f"  {path.name:<22} {mb:>7.1f} MB   {note}")
print("─" * 50)