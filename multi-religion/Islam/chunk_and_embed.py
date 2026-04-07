"""
chunk_and_embed.py  —  Islam  (English + Sinhala + Tamil)

Follows the same chunking strategy as Christianity:
  - Word-based chunking (CHUNK_SIZE / CHUNK_OVERLAP) for all languages
  - Single model: all-MiniLM-L6-v2 for all languages → single FAISS index
  - English  : Surah preamble + synonym expansion injected into embed_text
  - Sinhala / Tamil: plain "Surah + Category" header + raw chunk only
                     (no English preambles — keeps native script chunks clean)
  - Hadith   : source header + raw text (+ synonym expansion for English)

Single model ensures query vectors and index vectors are always comparable.

Reads:  data/quran_raw.json
Writes:
  data/chunks-en-si-ta.json    (backup — all chunk metadata without embed_text)
  data/chunks-en-si-ta.db      (SQLite — runtime retrieval store, all languages)
  data/embeddings-en-si-ta.npy (numpy backup)
  data/faiss_index-en-si-ta.bin (FAISS index — all languages, single index)

quran_raw.json entry schema:
  {
    "text":     "<passage text>",
    "section":  "<surah name | hadith collection name>",
    "category": "<Meccan | Medinan | Hadith>",
    "surah":    <int>,
    "ayah":     <int>,
    "religion": "Islam",
    "language": "en" | "sin" | "tam",
    "source":   "quran" | "bukhari" | "muslim" | ...
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

CORPUS_PATH     = DATA_DIR / "quran_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks-en-si-ta.json"
CHUNKS_DB_PATH  = DATA_DIR / "chunks-en-si-ta.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings-en-si-ta.npy"
FAISS_PATH      = DATA_DIR / "faiss_index-en-si-ta.bin"

# ─────────────────────────────────────────────────────────────────────────────
# Settings  — mirrors Christianity exactly
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 400    # words per chunk  (Christianity uses 400)
CHUNK_OVERLAP = 80     # word overlap     (Christianity uses 80)

# Single model for all languages — same model used at query time in retrieve.py
# This ensures query vectors and index vectors are always from the same space.
MODEL_NAME = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC PREAMBLE MAP  (English only — injected for EN entries)
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_PREAMBLES: list[tuple[str, str]] = [
    ("Al-Fatihah", (
        "Al-Fatihah. Opening chapter. Bismillah. In the name of Allah. "
        "Praise to Allah Lord of the worlds. Most Gracious Most Merciful. "
        "Master of the Day of Judgment. You alone we worship. "
        "Guide us to the straight path. Surah Fatiha. Daily prayer. Salah. "
        "Opening of the Quran. First surah."
    )),
    ("Al-Baqarah", (
        "Al-Baqarah. Longest surah. Quran guidance for believers. "
        "Taqwa. Fear of Allah. Five pillars of Islam. "
        "Prayer salah five times a day. Fajr Dhuhr Asr Maghrib Isha. "
        "Establish prayer. And establish regular prayer salah. "
        "Fasting sawm. Zakat. Hajj. "
        "Belief in Allah angels books prophets hereafter. "
        "Ayatul Kursi. Throne verse. Allah living sustaining. "
        "No compulsion in religion. 2:256. "
        "Riba interest prohibited. Marriage divorce. Inheritance. Qibla. "
        "Children of Israel. Moses Pharaoh. Story of Adam. "
        "Abraham builds Kaaba. Fasting Ramadan. Laylatul Qadr."
    )),
    ("Ali 'Imran", (
        "Ali Imran. Family of Imran. Jesus birth of Jesus. Mary Maryam. "
        "Isa prophet. Trinity refuted. Allah one God. "
        "Battle of Uhud. Unity of believers. Obedience to Allah. "
        "Hold fast to the rope of Allah. Best nation ummah. "
        "Commanding good forbidding evil. Patience sabr."
    )),
    ("An-Nisa", (
        "An-Nisa. Women. Rights of women in Islam. Marriage. "
        "Inheritance law. Orphans protection. Justice. "
        "Prohibition of usury riba. Guardianship. Divorce. "
        "Hypocrites. Jihad. Prayer shortened in travel. "
        "Verily prayer has been decreed upon the believers at fixed times. "
        "Prayer is obligatory at fixed times. Five obligatory prayers."
    )),
    ("Al-Ma'idah", (
        "Al-Maidah. Table spread. Halal food permitted. Forbidden foods. "
        "Pork blood prohibited. Wudu ablution. "
        "Jesus disciples. Punishment for theft hudud. "
        "Jews Christians People of the Book."
    )),
    ("Al-An'am", (
        "Al-Anam. Cattle. Tawhid oneness of Allah. Polytheism refuted. "
        "Abraham argues against idol worship. Signs of Allah in creation. "
        "Day of Judgment. Accountability. Prohibition of shirk."
    )),
    ("Al-A'raf", (
        "Al-Araf. Heights. Story of Adam and Iblis. Fall of man. "
        "Stories of prophets Hud Salih Shuayb Lot Moses. "
        "Hell heaven described. Arrogance of Satan."
    )),
    ("At-Tawbah", (
        "At-Tawbah. Repentance. Only surah without Bismillah. "
        "Hypocrites exposed. Jihad. "
        "Repentance accepted by Allah. Allah turns to those who repent."
    )),
    ("Yunus", (
        "Yunus. Jonah. Story of Prophet Yunus Jonah whale. "
        "Patience of Yunus. Signs of Allah. Day of Judgment."
    )),
    ("Yusuf", (
        "Yusuf. Joseph. Best of stories ahsan al-qasas. Brothers of Yusuf. "
        "Egypt. Zulaikha temptation. Prison. Dream interpretation. "
        "Patience and trust in Allah. Forgiveness."
    )),
    ("Ibrahim", (
        "Ibrahim. Abraham. Prayer of Ibrahim for Mecca. "
        "Gratitude shukr. Parable of good word tree. Day of Judgment."
    )),
    ("Al-Kahf", (
        "Al-Kahf. Cave. Companions of the cave. "
        "Story of Khidr and Moses. Story of Dhul-Qarnayn. Gog Magog. "
        "Dajjal protection. Read Al-Kahf on Friday."
    )),
    ("Maryam", (
        "Maryam. Mary. Birth of Jesus Isa. Virgin birth. "
        "Zakariyya prayer for son. Birth of Yahya John the Baptist. "
        "Jesus speaks from cradle. Jesus prophet of Allah."
    )),
    ("Ta-Ha", (
        "Ta-Ha. Moses Musa and Pharaoh Firaun. Burning bush. "
        "Staff of Moses. Parting of sea. Ease after hardship."
    )),
    ("Al-Anbiya", (
        "Al-Anbiya. Prophets. Stories of many prophets. "
        "Ibrahim Lut Nuh Dawud Sulayman Ayyub Yunus Zakariyya Maryam Isa. "
        "Tawhid message of all prophets. Mercy to the worlds."
    )),
    ("Al-Mu'minun", (
        "Al-Muminun. Believers. Qualities of true believers. "
        "Humble in prayer. Paying zakat. Fulfilling trusts. "
        "Creation of man. Stages of human development."
    )),
    ("An-Nur", (
        "An-Nur. Light. Verse of Light. Allah is light of heavens and earth. "
        "Prohibition of zina fornication. Punishment for adultery. "
        "Hijab modesty for men and women. Lower your gaze. Privacy."
    )),
    ("Al-Furqan", (
        "Al-Furqan. Criterion. Quran as criterion truth from falsehood. "
        "Attributes of servants of the Most Merciful. Night prayer tahajjud."
    )),
    ("Ya-Sin", (
        "Ya-Sin. Heart of the Quran. Resurrection. Day of Judgment. "
        "Signs of Allah in creation. Read Yasin for the dying."
    )),
    ("Az-Zumar", (
        "Az-Zumar. Groups. Sincere devotion to Allah. "
        "Do not despair of Allah's mercy. Allah forgives all sins. Tawbah repentance."
    )),
    ("Ghafir", (
        "Ghafir. Forgiver. Call upon Allah He answers. Dua supplication."
    )),
    ("Fussilat", (
        "Fussilat. Explained in detail. Quran guidance and healing. "
        "We will show them Our signs. Limbs will testify on Day of Judgment."
    )),
    ("Muhammad", (
        "Muhammad. Surah Muhammad. Jihad. "
        "Paradise rivers of water milk honey wine. Obey Allah and messenger."
    )),
    ("Al-Hujurat", (
        "Al-Hujurat. Chambers. Islamic etiquette adab. "
        "Verify news. Do not mock others. Avoid suspicion. Do not spy. "
        "Backbiting gheebah. All mankind from Adam and Eve. "
        "Most noble is most pious. Believers are brothers."
    )),
    ("Ar-Rahman", (
        "Ar-Rahman. Most Merciful. Which favours of your Lord will you deny. "
        "Blessings of Allah. Two paradises jannah. Balance mizan. Justice."
    )),
    ("Al-Waqi'ah", (
        "Al-Waqiah. Inevitable event. Day of Judgment. "
        "Three groups forerunners companions of right companions of left. "
        "Paradise described. Hell described. Read surah Waqiah for rizq."
    )),
    ("Al-Mulk", (
        "Al-Mulk. Sovereignty. Dominion. Created death and life to test. "
        "Seven heavens. Read Al-Mulk every night. Protection from punishment of grave."
    )),
    ("Al-Insan", (
        "Al-Insan. Man. Creation of man tested. "
        "Reward of paradise for feeding poor orphan captive."
    )),
    ("An-Naba", (
        "An-Naba. Great news. Day of Judgment. "
        "Trumpet blown. Hell torment. Paradise reward."
    )),
    ("Al-Fajr", (
        "Al-Fajr. Dawn. Stories of Aad Thamud Pharaoh destroyed. "
        "Soul at rest nafs mutmainnah return to your Lord."
    )),
    ("Al-Ikhlas", (
        "Al-Ikhlas. Sincerity. Say He is Allah One. Ahad. "
        "Allah the Eternal Absolute Samad. He begets not nor was begotten. "
        "Tawhid. Oneness of God. Pure monotheism. Equal to one third of Quran."
    )),
    ("Al-Falaq", (
        "Al-Falaq. Daybreak. Seek refuge in Lord of daybreak. "
        "Evil of what He created. Protection from black magic evil eye. Ruqyah."
    )),
    ("An-Nas", (
        "An-Nas. Mankind. Seek refuge in Lord of mankind. "
        "Evil of whispering retreating Shaytan. Protection from Shaytan."
    )),
    ("Al-Qadr", (
        "Al-Qadr. Power decree. Night of Power Laylatul Qadr. "
        "Better than thousand months. Last ten nights of Ramadan. Odd nights."
    )),
    ("Al-'Asr", (
        "Al-Asr. Time. By time mankind is in loss. "
        "Except those who believe do good deeds enjoin truth and patience."
    )),
]


def _get_topic_preamble(section: str) -> str:
    section_lower = section.lower()
    best_match, best_len = "", 0
    for key, preamble in TOPIC_PREAMBLES:
        if key.lower() in section_lower and len(key) > best_len:
            best_match, best_len = preamble.strip(), len(key)
    return best_match


# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM EXPANSION MAP  (English only)
# ─────────────────────────────────────────────────────────────────────────────

SYNONYM_EXPANSIONS: list[tuple[list[str], str]] = [
    (["allah", "god", "lord", "rabb", "ilah", "creator"],
     "Who is Allah. Nature of God in Islam. Tawhid oneness of Allah. "
     "Allah's names and attributes. 99 names of Allah. Asmaul Husna. Omnipotent."),
    (["prayer", "salah", "salat", "pray", "worship", "prostrate",
      "ruku", "sujud", "wudu", "ablution", "qibla", "mosque", "masjid",
      "times", "how many", "five times", "fajr", "dhuhr", "asr",
      "maghrib", "isha", "obligatory", "establish prayer"],
     "Salah prayer in Islam. Five daily prayers five times a day. "
     "Fajr dawn prayer. Dhuhr midday prayer. Asr afternoon prayer. "
     "Maghrib sunset prayer. Isha night prayer. "
     "How many times do Muslims pray per day. Muslims pray five times daily. "
     "Establish regular prayer salah. Second pillar of Islam. "
     "Prayer pillar of Islam. Wudu before prayer. "
     "Friday prayer Jumuah. Night prayer tahajjud. "
     "And establish prayer and give zakat. Quran commands prayer. "
     "Verily prayer has been decreed upon the believers at fixed times."),
    (["fast", "fasting", "sawm", "ramadan", "iftar", "suhoor", "suhur"],
     "Fasting in Islam. Sawm Ramadan. Pillar of Islam. "
     "Why Muslims fast. Ramadan month of Quran. "
     "Laylatul Qadr night of power. Eid ul-Fitr after Ramadan."),
    (["zakat", "charity", "sadaqah", "giving", "poor", "needy", "orphan"],
     "Zakat in Islam. Obligatory charity. Pillar of Islam. "
     "Sadaqah voluntary charity. Giving to poor needy orphans."),
    (["hajj", "pilgrimage", "kaaba", "mecca", "ihram", "tawaf",
      "safa marwa", "arafat", "mina", "umrah"],
     "Hajj pilgrimage to Mecca. Fifth pillar of Islam. "
     "Kaaba house of Allah. Eid ul-Adha sacrifice after Hajj."),
    (["quran", "scripture", "revelation", "kitab", "word of allah", "recite"],
     "Quran as word of Allah. Reading reciting the Quran. "
     "Quran guidance for mankind. Quran revealed in Ramadan."),
    (["prophet", "muhammad", "messenger", "rasul", "nabi",
      "sunnah", "hadith", "pbuh", "peace be upon him"],
     "Prophet Muhammad peace be upon him. Last prophet. "
     "Following the sunnah. Hadith teachings of prophet."),
    (["paradise", "heaven", "jannah", "garden", "reward", "hereafter",
      "akhira", "eternal life", "bliss"],
     "Jannah paradise in Islam. Description of paradise. "
     "Reward for believers. Gardens rivers of milk honey. "
     "How to enter paradise. Akhira hereafter."),
    (["hell", "hellfire", "jahannam", "punishment", "torment", "fire", "azab"],
     "Jahannam hellfire in Islam. Punishment of hellfire. "
     "Who goes to hell. Sins leading to hellfire."),
    (["day of judgment", "qiyamah", "resurrection", "reckoning",
      "accountability", "scales mizan", "book of deeds"],
     "Yawm al-Qiyamah Day of Judgment. Resurrection of the dead. "
     "Accountability before Allah. Scales of deeds mizan."),
    (["sin", "repentance", "tawbah", "forgiveness", "istighfar",
      "mercy", "rahma"],
     "Tawbah repentance in Islam. Allah forgives all sins. "
     "Seeking forgiveness istighfar. Allah's mercy. "
     "Do not despair of Allah's mercy."),
    (["shirk", "polytheism", "idol", "idol worship", "associating partners",
      "tawhid", "oneness"],
     "Shirk worst sin in Islam. Associating partners with Allah. "
     "Tawhid monotheism. Idol worship forbidden. Pure monotheism Islam."),
    (["jesus", "isa", "messiah", "christ", "mary", "maryam"],
     "Isa Jesus in Islam. Prophet Isa ibn Maryam. "
     "Miraculous birth of Isa. Jesus is prophet not son of God."),
    (["moses", "musa", "pharaoh", "firaun", "exodus"],
     "Musa Moses prophet in Islam. Moses and Pharaoh Firaun. "
     "Exodus from Egypt. Children of Israel Bani Israel."),
    (["abraham", "ibrahim", "ismail", "ishmael", "kaaba", "sacrifice"],
     "Ibrahim Abraham prophet in Islam. Father of prophets. "
     "Building of Kaaba with Ismail. Ibrahim breaks idols."),
    (["patience", "sabr", "trial", "test", "hardship", "difficulty",
      "tribulation", "suffering"],
     "Sabr patience in Islam. Allah tests believers. "
     "Indeed with hardship comes ease. Trust in Allah tawakkul."),
    (["dua", "supplication", "ask allah", "call upon", "invocation"],
     "Dua supplication in Islam. Call upon Allah He answers. "
     "How to make dua. Dua is worship ibadah."),
    (["knowledge", "ilm", "learn", "education", "seek knowledge", "wisdom"],
     "Seeking knowledge in Islam. First revelation iqra read. "
     "Scholars inherit prophets. Knowledge as obligation."),
    (["marriage", "nikah", "husband", "wife", "family", "children",
      "divorce", "talaq", "spouse"],
     "Marriage nikah in Islam. Rights of husband and wife. "
     "Divorce talaq last resort. Family as foundation of society."),
    (["death", "dying", "soul", "ruh", "grave", "barzakh"],
     "Death in Islam. Soul ruh. What happens after death. "
     "Life in grave barzakh. Every soul shall taste death."),
    (["jihad", "struggle", "strive", "way of allah", "fight"],
     "Jihad in Islam. Striving in way of Allah. "
     "Greater jihad against one's own nafs. Islam religion of peace."),
    (["halal", "haram", "forbidden", "permitted", "pork", "alcohol", "riba"],
     "Halal and haram in Islam. Pork forbidden. Alcohol forbidden. "
     "Riba interest forbidden. Islamic law sharia."),
]


def _get_synonym_expansion(text: str) -> str:
    text_lower = text.lower()
    additions  = []
    for triggers, expansion in SYNONYM_EXPANSIONS:
        if any(t in text_lower for t in triggers):
            additions.append(expansion)
    return " ".join(additions)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking helper  — mirrors Christianity exactly
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str,
               size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > 40:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Islam Chunk & Embed — English + Sinhala + Tamil")
print("  Strategy: word-based chunking (mirrors Christianity)")
print("=" * 60)

print(f"\nLoading corpus from {CORPUS_PATH.name} ...")
corpus: list[dict] = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(corpus):,} passage entries")

by_lang_src: dict[str, int] = {}
for item in corpus:
    key = f"{item.get('language','?')}:{item.get('source','?')}"
    by_lang_src[key] = by_lang_src.get(key, 0) + 1
print(f"\n  By language:source:")
for k, cnt in sorted(by_lang_src.items(), key=lambda x: -x[1]):
    print(f"    {k:<28} {cnt:>7,}")

# ─────────────────────────────────────────────────────────────────────────────
# Build chunks
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Chunking corpus ({CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap) ---")
all_chunks: list[dict] = []

for item in tqdm(corpus, desc="Chunking"):
    raw_text = item.get("text", "").strip()
    section  = item.get("section",  "Unknown")
    category = item.get("category", "Unknown")
    surah    = item.get("surah",    0)
    ayah     = item.get("ayah",     0)
    language = item.get("language", "en")
    source   = item.get("source",   "quran")

    if not raw_text:
        continue

    for chunk in chunk_text(raw_text):
        parts = []

        if language == "en":
            # English: topic preamble + source header + chunk + synonym expansion
            preamble = _get_topic_preamble(section)
            if preamble:
                parts.append(preamble)
            parts.append(
                f"Surah: {section}. Category: {category}. Religion: Islam."
            )
            parts.append(chunk)
            synonyms = _get_synonym_expansion(chunk)
            if synonyms:
                parts.append(synonyms)

        else:
            # Sinhala / Tamil: plain header + chunk only.
            # No English preambles — keeps native chunks clean so they
            # match native-script queries in FAISS.
            lang_label = {"sin": "Sinhala", "tam": "Tamil"}.get(language, language)
            parts.append(
                f"Surah: {section}. Category: {category}. Language: {lang_label}."
            )
            parts.append(chunk)

        all_chunks.append({
            "text":       chunk,
            "embed_text": " ".join(parts),
            "book":       section,
            "section":    section,
            "category":   category,
            "surah":      surah,
            "ayah":       ayah,
            "religion":   "Islam",
            "language":   language,
            "source":     source,
        })

print(f"Total chunks produced: {len(all_chunks):,}")

by_lang: dict[str, int] = {}
for c in all_chunks:
    by_lang[c["language"]] = by_lang.get(c["language"], 0) + 1
print(f"\n  Chunks by language:")
for lang, cnt in sorted(by_lang.items()):
    label = {"en": "English", "sin": "Sinhala", "tam": "Tamil"}.get(lang, lang)
    print(f"    {label:<10} {cnt:>8,}")

# ─────────────────────────────────────────────────────────────────────────────
# SQLite  — chunks-en-si-ta.db
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Building SQLite DB: {CHUNKS_DB_PATH.name} ---")

if CHUNKS_DB_PATH.exists():
    CHUNKS_DB_PATH.unlink()

con = sqlite3.connect(str(CHUNKS_DB_PATH))
con.executescript("""
    CREATE TABLE chunks (
        id        INTEGER PRIMARY KEY,
        text      TEXT    NOT NULL,
        book      TEXT    NOT NULL DEFAULT '',
        section   TEXT    NOT NULL DEFAULT '',
        category  TEXT    NOT NULL DEFAULT '',
        surah     INTEGER NOT NULL DEFAULT 0,
        ayah      INTEGER NOT NULL DEFAULT 0,
        religion  TEXT    NOT NULL DEFAULT 'Islam',
        language  TEXT    NOT NULL DEFAULT 'en',
        source    TEXT    NOT NULL DEFAULT 'quran'
    );

    CREATE INDEX idx_religion  ON chunks (religion);
    CREATE INDEX idx_language  ON chunks (language);
    CREATE INDEX idx_book      ON chunks (book);
    CREATE INDEX idx_surah     ON chunks (surah);
    CREATE INDEX idx_category  ON chunks (category);
    CREATE INDEX idx_source    ON chunks (source);
    CREATE INDEX idx_lang_src  ON chunks (language, source);
    CREATE INDEX idx_lang_cat  ON chunks (language, category);
""")

con.executemany(
    "INSERT INTO chunks "
    "(id, text, book, section, category, surah, ayah, religion, language, source) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    [
        (
            i,
            c["text"],
            c.get("book",     ""),
            c.get("section",  ""),
            c.get("category", ""),
            c.get("surah",    0),
            c.get("ayah",     0),
            c.get("religion", "Islam"),
            c.get("language", "en"),
            c.get("source",   "quran"),
        )
        for i, c in enumerate(all_chunks)
    ],
)
con.commit()

print(f"  DB row counts per language:")
for lang, cnt in sorted(by_lang.items()):
    row = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE language=?", (lang,)
    ).fetchone()
    label = {"en": "English", "sin": "Sinhala", "tam": "Tamil"}.get(lang, lang)
    print(f"    {label:<10} {row[0]:>8,}")

con.close()

db_mb = CHUNKS_DB_PATH.stat().st_size / 1_048_576
print(f"Saved {CHUNKS_DB_PATH.name}  ({db_mb:.1f} MB, {len(all_chunks):,} rows)")

CHUNKS_PATH.write_text(
    json.dumps(
        [{k: v for k, v in c.items() if k != "embed_text"} for c in all_chunks],
        indent=2, ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"Saved {CHUNKS_PATH.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings + FAISS  — single model, single index (mirrors Christianity)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Generating embeddings with '{MODEL_NAME}' ---")
print(f"    Single model for all languages — matches query model in retrieve.py")
model = SentenceTransformer(MODEL_NAME)

embeddings: np.ndarray = model.encode(
    [c["embed_text"] for c in all_chunks],
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

np.save(EMBEDDINGS_PATH, embeddings)
print(f"Saved {EMBEDDINGS_PATH.name}  shape={embeddings.shape}")

print(f"\n--- Building FAISS index ---")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, str(FAISS_PATH))
print(f"Saved {FAISS_PATH.name}  ({index.ntotal:,} vectors, dim={embeddings.shape[1]})")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  Input passages  : {len(corpus):,}")
print(f"  Total chunks    : {len(all_chunks):,}")
print(f"\n  Chunks by language:")
for lang, cnt in sorted(by_lang.items()):
    label = {"en": "English", "sin": "Sinhala", "tam": "Tamil"}.get(lang, lang)
    print(f"    {label:<10} {cnt:>8,}")

print(f"\n  Output files:")
for path, note in [
    (CHUNKS_DB_PATH,  "<-- upload to HuggingFace / runtime"),
    (FAISS_PATH,      "<-- upload to HuggingFace / runtime"),
    (EMBEDDINGS_PATH, "<-- local backup"),
    (CHUNKS_PATH,     "<-- local backup"),
]:
    mb = path.stat().st_size / 1_048_576
    print(f"    {path.name:<42} {mb:>7.1f} MB   {note}")

print(f"\n  Chunks by category:")
chunk_by_cat: dict[str, int] = {}
for c in all_chunks:
    chunk_by_cat[c["category"]] = chunk_by_cat.get(c["category"], 0) + 1
for cat, cnt in sorted(chunk_by_cat.items(), key=lambda x: -x[1]):
    print(f"    {cat:<14} {cnt:>8,}")

print("=" * 60)