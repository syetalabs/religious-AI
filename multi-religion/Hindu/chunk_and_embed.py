"""
chunk_and_embed.py  -  Hinduism scripture data

Reads:  data/hindu_raw.json
Writes:
  data/chunks.json              (backup — all chunk metadata without embed_text)
  data/chunks-hindu.db          (SQLite — runtime retrieval store)
  data/embeddings-hindu.npy     (numpy backup)
  data/faiss_index-hindu.bin    (FAISS index — runtime similarity search)

Each entry in hindu_raw.json must have:
  {
    "text":     "<scripture text>",
    "section":  "<scripture name>",   e.g. "Bhagavad Gita"
    "category": "<category>",         e.g. "Epics", "Upanishads", "Vedas"
    "religion": "Hinduism",
    "language": "en"
  }
"""

import json
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"

CORPUS_PATH     = DATA_DIR / "hindu_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks.json"
CHUNKS_DB_PATH  = DATA_DIR / "chunks.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_PATH      = DATA_DIR / "faiss_index.bin"

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 400     # words per chunk
CHUNK_OVERLAP = 80      # word overlap between consecutive chunks
MODEL_NAME    = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Category → scripture type label
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_LABELS: dict[str, str] = {
    "Epics":            "Hindu Epic",
    "Upanishads":       "Upanishad",
    "Vedas":            "Veda",
    "Yoga / Philosophy":"Yoga Philosophy",
    "Dharmashastra":    "Dharmashastra",
}

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC PREAMBLE MAP
#
# Format: (section_name_substring, preamble_text)
# Matched with  section.lower().startswith(key.lower())  so partial prefixes work.
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_PREAMBLES: list[tuple[str, str]] = [

    # ── Bhagavad Gita ─────────────────────────────────────────────────────────
    ("Bhagavad Gita", (
        "Bhagavad Gita. Song of God. Krishna and Arjuna. Kurukshetra battlefield. "
        "Dharma duty righteousness. Karma yoga path of action. "
        "Jnana yoga path of knowledge. Bhakti yoga path of devotion. "
        "Self and soul Atman. Non-attachment to results of actions. "
        "Do your duty without attachment to fruit. Nishkama karma. "
        "God incarnate. Avatar of Vishnu. Supreme Being Brahman. "
        "Reincarnation rebirth cycle of samsara. Liberation moksha. "
        "Three gunas: sattva rajas tamas. Yoga of meditation dhyana. "
        "Mind and senses control. Eternal soul never dies. "
        "Surrender to God. Sarva dharman parityajya. Devotion to Krishna."
    )),

    # ── Upanishads (generic preamble for all) ─────────────────────────────────
    ("Upanishad", (
        "Upanishad. Vedanta philosophy. End of the Vedas. "
        "Brahman ultimate reality. Atman individual soul. "
        "Brahman and Atman are one. Tat tvam asi That thou art. "
        "Aham Brahmasmi I am Brahman. Unity of self and universe. "
        "Maya illusion. Knowledge and ignorance avidya. "
        "Meditation contemplation. Liberation moksha. "
        "Consciousness and being. Eternal self. "
        "What is the nature of the self. Who am I. What is reality."
    )),

    # ── Isha Upanishad ────────────────────────────────────────────────────────
    ("Isha Upanishad", (
        "Isha Upanishad. Ishopanishad. All this is pervaded by the Lord. "
        "Non-possessive living. Renunciation and enjoyment. "
        "Deeds not binding the knower of Brahman. "
        "God dwells in all things. Oneness of existence."
    )),

    # ── Kena Upanishad ────────────────────────────────────────────────────────
    ("Kena Upanishad", (
        "Kena Upanishad. By whose will does the mind go forth. "
        "That which eye cannot see ear cannot hear. "
        "Brahman is not the known nor the unknown. "
        "Ultimate reality beyond senses and intellect."
    )),

    # ── Katha Upanishad ───────────────────────────────────────────────────────
    ("Katha Upanishad", (
        "Katha Upanishad. Nachiketa and Yama the god of death. "
        "Secret of death and immortality. "
        "The self is not born nor does it die. "
        "Smaller than the small greater than the great. "
        "Discrimination between the pleasant and the good. "
        "Chariot metaphor: body is chariot senses are horses mind is reins."
    )),

    # ── Mundaka Upanishad ─────────────────────────────────────────────────────
    ("Mundaka Upanishad", (
        "Mundaka Upanishad. Two kinds of knowledge: lower and higher. "
        "Higher knowledge is knowledge of Brahman. "
        "The self is not attained by the weak or the careless. "
        "Brahman as a bird on the tree of life. "
        "Satyameva Jayate Truth alone triumphs."
    )),

    # ── Chandogya Upanishad ───────────────────────────────────────────────────
    ("Chandogya Upanishad", (
        "Chandogya Upanishad. Tat tvam asi That art thou. "
        "Prana life force. Udgitha chanting of Om. "
        "Brahman as food and breath. "
        "Svetaketu and Uddalaka: subtle essence salt in water. "
        "Underlying unity of all things."
    )),

    # ── Brihadaranyaka Upanishad ──────────────────────────────────────────────
    ("Brihadaranyaka Upanishad", (
        "Brihadaranyaka Upanishad. Largest Upanishad. "
        "Yajnavalkya and Maitreyi: the self as the innermost principle. "
        "Neti neti not this not this. "
        "Honey doctrine: all beings are honey for each other. "
        "Transmigration of souls. Karma and rebirth. "
        "Self is the seer of seeing hearer of hearing. "
        "Atman is Brahman. Fear arises from duality."
    )),

    # ── Mandukya Upanishad ────────────────────────────────────────────────────
    ("Mandukya Upanishad", (
        "Mandukya Upanishad. Om and its four quarters. "
        "Four states: waking dreaming deep sleep turiya. "
        "Turiya fourth state is pure consciousness. "
        "Om is all this. The syllable Om as Brahman. "
        "Gaudapada karika on non-duality advaita."
    )),

    # ── Taittiriya Upanishad ──────────────────────────────────────────────────
    ("Taittiriya Upanishad", (
        "Taittiriya Upanishad. Five sheaths koshas of the self. "
        "Annamaya physical body. Pranamaya vital body. "
        "Manomaya mental body. Vijnanamaya intellect body. Anandamaya bliss body. "
        "Brahman is truth knowledge infinite. Satyam jnanam anantam Brahman. "
        "Bliss of Brahman is the source of all beings."
    )),

    # ── Rig Veda ──────────────────────────────────────────────────────────────
    ("Rig Veda", (
        "Rig Veda. Rigveda. Oldest Hindu scripture. Ancient hymns mantras. "
        "Hymns to Agni fire god. Hymns to Indra king of gods. "
        "Hymns to Varuna cosmic order. Hymns to Soma. "
        "Nasadiya Sukta: creation hymn. In the beginning there was neither being nor non-being. "
        "Purusha Sukta: cosmic person hymn. Origin of the universe. "
        "Prayers to dawn Ushas. Sacred fire ritual yajna. "
        "Vedic deities gods. Rta cosmic order. Dharma right action. "
        "Ancient Indian prayer and worship. Vedic religion."
    )),

    # ── Yoga Sutras ───────────────────────────────────────────────────────────
    ("Yoga Sutras", (
        "Yoga Sutras of Patanjali. Classical yoga system. "
        "Yoga is the cessation of fluctuations of the mind. Chitta vritti nirodha. "
        "Eight limbs of yoga: yama niyama asana pranayama pratyahara dharana dhyana samadhi. "
        "Yama ethical restraints: ahimsa non-violence satya truthfulness asteya non-stealing "
        "brahmacharya celibacy aparigraha non-greed. "
        "Niyama observances: saucha purity santosha contentment tapas austerity "
        "svadhyaya self-study Ishvarapranidhana surrender to God. "
        "Asana posture. Pranayama breath control. Pratyahara withdrawal of senses. "
        "Dharana concentration. Dhyana meditation. Samadhi absorption enlightenment. "
        "Samskara mental impressions. Vritti mental modifications. Klesha afflictions. "
        "Avidya ignorance. Asmita ego. Raga attraction. Dvesha aversion. Abhinivesha fear of death. "
        "How to meditate. How to attain liberation moksha. Union with the divine."
    )),

    # ── Laws of Manu ──────────────────────────────────────────────────────────
    ("Laws of Manu", (
        "Laws of Manu. Manusmriti. Hindu law code. "
        "Four stages of life ashrama: brahmacharya student grihastha householder "
        "vanaprastha forest dweller sannyasa renunciation. "
        "Four varnas castes: Brahmin Kshatriya Vaishya Shudra. "
        "Dharma duty of each stage and class. "
        "Rules of conduct righteousness ethics. "
        "Hindu social order. Sacred law. Samskaras rites of passage. "
        "Duties of kings warriors priests. Hindu ethics morality."
    )),

    # ── Ramayana ──────────────────────────────────────────────────────────────
    ("Ramayana", (
        "Ramayana. Epic of Rama and Sita. Valmiki Ramayana. "
        "Rama avatar of Vishnu. Ideal king and husband. "
        "Sita abducted by Ravana demon king of Lanka. "
        "Hanuman devotee of Rama crosses the ocean. "
        "Battle of Lanka. Ravana defeated. Dharma over adharma. "
        "Exile of Rama for fourteen years. Ayodhya. "
        "Ideal family duty loyalty devotion. Bhakti devotion. "
        "Victory of good over evil. Righteousness dharma. "
        "Lakshmana brother of Rama. Bharata loyalty."
    )),

    # ── Mahabharata ───────────────────────────────────────────────────────────
    ("Mahabharata", (
        "Mahabharata. Great epic of India. Pandavas and Kauravas. "
        "Kurukshetra war. Arjuna and Krishna. Dharma duty righteousness. "
        "Yudhishthira justice. Bhima strength. Arjuna warrior devotion. "
        "Draupadi. Duryodhana and adharma. "
        "Thirteen years of exile. Dice game. "
        "Vishnu Bhishma Drona Karna. Epic family conflict. "
        "What is righteousness. How to live a good life. "
        "Dharma is subtle. Ahimsa non-violence. "
        "Vyasa the author. Longest epic poem in the world."
    )),
]


def _get_topic_preamble(section: str) -> str:
    """Return the best matching topic preamble for a given scripture section."""
    section_lower = section.lower()
    # Try longest matching prefix first for specificity
    best_match = ""
    best_len   = 0
    for key, preamble in TOPIC_PREAMBLES:
        if key.lower() in section_lower and len(key) > best_len:
            best_match = preamble.strip()
            best_len   = len(key)
    return best_match


# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM EXPANSION MAP
# ─────────────────────────────────────────────────────────────────────────────

SYNONYM_EXPANSIONS: list[tuple[list[str], str]] = [

    (["karma", "action", "duty", "deed", "fruit of action"],
     "What is karma. Law of karma. Karma and dharma. "
     "Actions and their consequences. Karma yoga. "
     "Do your duty without attachment. Nishkama karma."),

    (["dharma", "duty", "righteousness", "right action", "moral law"],
     "What is dharma. Hindu concept of duty. "
     "Dharma and adharma. Righteous living. Moral obligation. "
     "Sanatana dharma. Eternal law."),

    (["moksha", "liberation", "freedom", "enlightenment", "nirvana", "mukti"],
     "What is moksha. How to attain liberation. "
     "Freedom from cycle of rebirth samsara. "
     "Spiritual liberation. Self-realization. Union with Brahman."),

    (["atman", "self", "soul", "true self", "inner self", "consciousness"],
     "What is atman. Individual soul. "
     "The self is not the body or mind. "
     "Eternal self. Pure consciousness. Who am I."),

    (["brahman", "ultimate reality", "absolute", "god", "divine", "supreme"],
     "What is Brahman. Ultimate reality in Hinduism. "
     "Nirguna Brahman without qualities. Saguna Brahman with qualities. "
     "God in Hindu philosophy. Supreme being."),

    (["maya", "illusion", "ignorance", "avidya", "veil"],
     "What is maya. Illusion in Hindu philosophy. "
     "World as illusion. Cosmic illusion. "
     "Avidya spiritual ignorance. Veil of maya. "
     "How maya creates suffering."),

    (["meditation", "dhyana", "samadhi", "contemplation", "mindfulness"],
     "How to meditate. What is meditation in Hinduism. "
     "Dhyana meditation practice. Samadhi deep meditation. "
     "Yoga and meditation. Quieting the mind. "
     "Om meditation. Transcendental meditation."),

    (["reincarnation", "rebirth", "samsara", "cycle", "birth and death"],
     "What is reincarnation. Hindu belief in rebirth. "
     "Cycle of samsara. Soul transmigration. "
     "Birth death and rebirth. Karma and reincarnation. "
     "How to escape the cycle of rebirth."),

    (["yoga", "path", "union", "spiritual practice"],
     "What is yoga. Types of yoga. "
     "Karma yoga. Jnana yoga. Bhakti yoga. Raja yoga. "
     "Yoga as spiritual path. Union with the divine. "
     "Eight limbs of yoga."),

    (["devotion", "bhakti", "worship", "surrender", "prayer", "puja"],
     "What is bhakti. Devotion to God in Hinduism. "
     "Bhakti yoga path of devotion. Hindu worship puja. "
     "Surrender to God. Loving devotion. "
     "Devotion to Krishna Rama Shiva Devi."),

    (["god", "krishna", "vishnu", "shiva", "devi", "divine mother", "brahma"],
     "Hindu concept of God. Trimurti: Brahma Vishnu Shiva. "
     "Krishna as supreme deity. Vishnu preserver. Shiva destroyer transformer. "
     "Devi goddess. Many gods one truth. Ekam sat. "
     "Avatar incarnation of God."),

    (["suffering", "pain", "sorrow", "grief", "despair", "fear"],
     "Hindu view of suffering. Cause of suffering. "
     "Attachment as source of suffering. Detachment as remedy. "
     "God's presence in suffering. Equanimity in difficulty. "
     "Arjuna's grief on the battlefield."),

    (["non-violence", "ahimsa", "compassion", "kindness"],
     "Ahimsa non-violence in Hinduism. "
     "Compassion for all beings. "
     "Non-harm as spiritual practice. Vegetarianism. "
     "Reverence for life. All beings are sacred."),

    (["creation", "universe", "cosmos", "origin", "beginning"],
     "Hindu creation stories. Creation of the universe. "
     "Nasadiya Sukta Rig Veda creation hymn. "
     "Brahma creator god. Cosmic cycle of creation and dissolution. "
     "Kalpa cosmic age. Time cycles. Vishnu's cosmic sleep."),

    (["om", "aum", "mantra", "sacred sound", "chant"],
     "What is Om. Significance of Om in Hinduism. "
     "Om as the sound of Brahman. Primordial sound. "
     "Om Shanti peace mantra. Sacred syllable. "
     "Chanting mantras as spiritual practice."),

    (["samskar", "ritual", "ceremony", "rite of passage", "tradition"],
     "Hindu samskaras rites of passage. "
     "Birth naming thread ceremony marriage death rituals. "
     "Hindu religious ceremonies. Yajna fire ritual. "
     "Religious observances festivals."),

    (["equanimity", "balance", "peace", "contentment", "stillness", "samata"],
     "Equanimity in Hindu philosophy. "
     "Sama equal vision. Steady mind. "
     "Peace beyond understanding. Contentment santosha. "
     "Stillness of mind in yoga."),
]


def _get_synonym_expansion(text: str) -> str:
    """Return synonym expansions for any trigger words found in text."""
    text_lower = text.lower()
    additions  = []
    for triggers, expansion in SYNONYM_EXPANSIONS:
        if any(t in text_lower for t in triggers):
            additions.append(expansion)
    return " ".join(additions)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking helper
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str,
               size: int    = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    words  = text.split()
    chunks = []
    step   = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + size])
        if len(chunk.strip()) > 40:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Data quality checks
# ─────────────────────────────────────────────────────────────────────────────

# Known boilerplate strings that should have been cleaned but may still appear
_BOILERPLATE_CHECKS = [
    "isbn-10:",
    "isbn-13:",
    "humbly request your help",
    "words | isbn",
    "let's make the world a better place",
]

# Unicode ranges for non-English scripts (Devanagari, Tamil, etc.)
_NON_ENGLISH_RANGES = [
    (0x0900, 0x097F),   # Devanagari
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0C80, 0x0CFF),   # Kannada
    (0x0D00, 0x0D7F),   # Malayalam
]

def _has_non_english(text: str, threshold: float = 0.05) -> bool:
    count = sum(
        1 for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in _NON_ENGLISH_RANGES)
    )
    return (count / max(len(text), 1)) >= threshold

def _is_boilerplate(text: str) -> bool:
    tl = text.lower()
    return any(frag in tl for frag in _BOILERPLATE_CHECKS)


# ─────────────────────────────────────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Hindu Scripture Chunk & Embed Pipeline")
print("=" * 60)

print(f"\nLoading corpus from {CORPUS_PATH.name} ...")
corpus: list[dict] = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(corpus):,} raw entries")

# ── Data quality report before chunking ──────────────────────────────────────
print("\n--- Data Quality Check ---")
by_section:   dict[str, int] = {}
by_category:  dict[str, int] = {}
skipped_non_english  = 0
skipped_boilerplate  = 0
skipped_too_short    = 0
skipped_duplicate    = 0
seen_texts: set[str] = set()

clean_corpus: list[dict] = []
for item in corpus:
    text     = item.get("text", "").strip()
    section  = item.get("section", "unknown")
    category = item.get("category", "Unknown")

    if not text or len(text) < 30:
        skipped_too_short += 1
        continue
    if _is_boilerplate(text):
        skipped_boilerplate += 1
        continue
    if _has_non_english(text):
        skipped_non_english += 1
        continue
    if text in seen_texts:
        skipped_duplicate += 1
        continue

    seen_texts.add(text)
    clean_corpus.append(item)
    by_section[section]   = by_section.get(section, 0) + 1
    by_category[category] = by_category.get(category, 0) + 1

print(f"  Entries passed   : {len(clean_corpus):,}")
print(f"  Skipped (too short)   : {skipped_too_short:,}")
print(f"  Skipped (boilerplate) : {skipped_boilerplate:,}")
print(f"  Skipped (non-English) : {skipped_non_english:,}")
print(f"  Skipped (duplicate)   : {skipped_duplicate:,}")
print(f"\n  By category:")
for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
    print(f"    {cat:<34} {count:>7,}")
print(f"\n  By scripture (top 20 by count):")
for sec, count in sorted(by_section.items(), key=lambda x: -x[1])[:20]:
    print(f"    {sec:<40} {count:>6,}")

# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Chunking ({CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap) ---")
all_chunks: list[dict] = []

for item in tqdm(clean_corpus, desc="Chunking"):
    text     = item.get("text", "").strip()
    section  = item.get("section", "unknown")
    category = item.get("category", "Unknown")
    language = item.get("language", "en")

    cat_label = CATEGORY_LABELS.get(category, category)
    preamble  = _get_topic_preamble(section)

    for raw_chunk in chunk_text(text):
        # Build the embed_text: preamble + metadata label + chunk + synonyms
        parts = []

        if preamble:
            parts.append(preamble)

        parts.append(
            f"Scripture: {section}. Category: {cat_label}. Religion: Hinduism."
        )
        parts.append(raw_chunk)

        synonyms = _get_synonym_expansion(raw_chunk)
        if synonyms:
            parts.append(synonyms)

        embed_text = " ".join(parts)

        all_chunks.append({
            "text":       raw_chunk,
            "embed_text": embed_text,
            "section":    section,
            "category":   category,
            "religion":   "Hinduism",
            "language":   language,
        })

print(f"Total chunks produced: {len(all_chunks):,}")

# Chunk count by scripture
chunk_by_section: dict[str, int] = {}
for c in all_chunks:
    chunk_by_section[c["section"]] = chunk_by_section.get(c["section"], 0) + 1
print("\n  Chunks by scripture:")
for sec, count in sorted(chunk_by_section.items(), key=lambda x: -x[1]):
    print(f"    {sec:<40} {count:>6,}")

# ─────────────────────────────────────────────────────────────────────────────
# SQLite
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Building SQLite DB: {CHUNKS_DB_PATH.name} ---")

if CHUNKS_DB_PATH.exists():
    CHUNKS_DB_PATH.unlink()

con = sqlite3.connect(str(CHUNKS_DB_PATH))
con.executescript("""
    CREATE TABLE chunks (
        id        INTEGER PRIMARY KEY,
        text      TEXT    NOT NULL,
        section   TEXT    NOT NULL DEFAULT '',
        category  TEXT    NOT NULL DEFAULT '',
        religion  TEXT    NOT NULL DEFAULT 'Hinduism',
        language  TEXT    NOT NULL DEFAULT 'en'
    );

    CREATE INDEX idx_religion   ON chunks (religion);
    CREATE INDEX idx_section    ON chunks (section);
    CREATE INDEX idx_category   ON chunks (category);
""")

con.executemany(
    "INSERT INTO chunks (id, text, section, category, religion, language) "
    "VALUES (?, ?, ?, ?, ?, ?)",
    [
        (
            i,
            c["text"],
            c.get("section",  ""),
            c.get("category", ""),
            c.get("religion", "Hinduism"),
            c.get("language", "en"),
        )
        for i, c in enumerate(all_chunks)
    ],
)
con.commit()
con.close()

db_mb = CHUNKS_DB_PATH.stat().st_size / 1_048_576
print(f"Saved {CHUNKS_DB_PATH.name}  ({db_mb:.1f} MB, {len(all_chunks):,} rows)")

# chunks.json backup (no embed_text to keep size manageable)
CHUNKS_PATH.write_text(
    json.dumps(
        [{k: v for k, v in c.items() if k != "embed_text"} for c in all_chunks],
        indent=2, ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"Saved {CHUNKS_PATH.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Generating embeddings with {MODEL_NAME} ---")
model = SentenceTransformer(MODEL_NAME)

embeddings: np.ndarray = model.encode(
    [c["embed_text"] for c in all_chunks],
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

np.save(EMBEDDINGS_PATH, embeddings)
print(f"Saved {EMBEDDINGS_PATH.name}  shape={embeddings.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# FAISS index
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Building FAISS index ---")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, str(FAISS_PATH))
print(f"Saved {FAISS_PATH.name}  ({index.ntotal:,} vectors)")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  Input entries (after cleaning) : {len(clean_corpus):,}")
print(f"  Total chunks embedded          : {len(all_chunks):,}")
print(f"\n  Output files:")
for path, note in [
    (CHUNKS_DB_PATH,  "<-- upload to HuggingFace / runtime"),
    (FAISS_PATH,      "<-- upload to HuggingFace / runtime"),
    (EMBEDDINGS_PATH, "<-- local backup"),
    (CHUNKS_PATH,     "<-- local backup"),
]:
    mb = path.stat().st_size / 1_048_576
    print(f"    {path.name:<32} {mb:>7.1f} MB   {note}")

print(f"\n  Chunks by category:")
chunk_by_cat: dict[str, int] = {}
for c in all_chunks:
    chunk_by_cat[c["category"]] = chunk_by_cat.get(c["category"], 0) + 1
for cat, count in sorted(chunk_by_cat.items(), key=lambda x: -x[1]):
    print(f"    {cat:<34} {count:>7,}")
print("=" * 60)