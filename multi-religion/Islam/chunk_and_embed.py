"""
chunk_and_embed.py  —  Islam / Quran (English)

Reads:  data/quran_raw.json
Writes:
  data/chunks.json          (backup — all chunk metadata without embed_text)
  data/chunks.db            (SQLite — runtime retrieval store)
  data/embeddings.npy       (numpy backup)
  data/faiss_index.bin      (FAISS index — runtime similarity search)

Each entry in quran_raw.json must have:
  {
    "text":     "<passage text — 5 grouped verses with [ayah] prefixes>",
    "section":  "<surah name>",     e.g. "Al-Baqarah"
    "category": "<Meccan|Medinan>",
    "surah":    2,
    "ayah":     1,
    "religion": "Islam",
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

CORPUS_PATH     = DATA_DIR / "quran_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks.json"
CHUNKS_DB_PATH  = DATA_DIR / "chunks.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_PATH      = DATA_DIR / "faiss_index.bin"

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

# Chunking is intentionally bypassed — each verse is stored as its own chunk.
# Neighbouring verse context is injected into embed_text only (not stored text),
# so FAISS finds verses semantically while retrieval returns the exact verse.
CONTEXT_WINDOW = 2      # number of surrounding verses to inject into embed_text
                        # (2 = 1 verse before + 1 verse after)
MODEL_NAME    = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC PREAMBLE MAP
# Keyed by surah name substring. Improves FAISS retrieval for common topics.
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
        "Taqwa. Fear of Allah. Five pillars. Prayer salah. Fasting sawm. "
        "Zakat. Hajj. Belief in Allah angels books prophets hereafter. "
        "Ayatul Kursi. Throne verse. Allah living sustaining. "
        "No compulsion in religion. 2:256. "
        "Riba interest prohibited. Usury forbidden. "
        "Marriage divorce. Inheritance. Qibla direction of prayer. "
        "Children of Israel. Moses Pharaoh. Story of Adam. Iblis Satan. "
        "Abraham builds Kaaba. Hypocrites. Believers disbelievers. "
        "Fasting Ramadan. Laylatul Qadr. Night of Power. "
        "Last two verses. Amana rasool."
    )),

    ("Ali 'Imran", (
        "Ali Imran. Family of Imran. Jesus birth of Jesus. Mary Maryam. "
        "Isa prophet. Trinity refuted. Allah one God. "
        "Battle of Uhud. Unity of believers. Obedience to Allah. "
        "Hold fast to the rope of Allah. 3:103. Do not divide. "
        "Best nation ummah. Commanding good forbidding evil. 3:110. "
        "Steadfastness in hardship. Patience sabr."
    )),

    ("An-Nisa", (
        "An-Nisa. Women. Rights of women in Islam. Marriage. "
        "Inheritance law. Orphans protection. Justice. "
        "Obedience to Allah and messenger. "
        "Prohibition of usury riba. Guardianship. Divorce. "
        "Hypocrites. Jihad. Prayer shortened in travel."
    )),

    ("Al-Ma'idah", (
        "Al-Maidah. Table spread. Halal food permitted. Forbidden foods. "
        "Pork blood prohibited. Fulfil contracts covenants. "
        "Wudu ablution. Tayammum dry ablution. "
        "Jesus disciples Hawariyyun. Last supper miracle. "
        "Punishment for theft hudud. Qisas retaliation. "
        "Jews Christians People of the Book. Friendship with disbelievers. "
        "Allah forgives all sins except shirk."
    )),

    ("Al-An'am", (
        "Al-Anam. Cattle. Tawhid oneness of Allah. Polytheism refuted. "
        "Abraham argues against idol worship. Signs of Allah in creation. "
        "Day of Judgment. Accountability. Straight path sirat. "
        "Prohibition of shirk. Associating partners with Allah."
    )),

    ("Al-A'raf", (
        "Al-Araf. Heights. Story of Adam and Iblis. Fall of man. "
        "Stories of prophets Hud Salih Shuayb Lot Moses. "
        "People of the heights. Hell heaven described. "
        "Arrogance of Satan. Obedience to Allah."
    )),

    ("At-Tawbah", (
        "At-Tawbah. Repentance. Bara'ah. Only surah without Bismillah. "
        "Hypocrites exposed. Jihad. Jizya tax non-Muslims. "
        "Repentance accepted by Allah. Tawbah. Allah turns to those who repent. "
        "Masjid Dirar. Cave of Hira. Cave of Thawr."
    )),

    ("Yunus", (
        "Yunus. Jonah. Story of Prophet Yunus Jonah whale. "
        "Patience of Yunus. Tasbih dhikr in darkness. "
        "Signs of Allah. Day of Judgment. "
        "Intercession on Day of Judgment. Faith of people of Nineveh."
    )),

    ("Yusuf", (
        "Yusuf. Joseph. Story of Prophet Yusuf Joseph. "
        "Best of stories ahsan al-qasas. Brothers of Yusuf. "
        "Egypt. Zulaikha temptation. Prison. Dream interpretation. "
        "Reunion with father Yaqub Jacob. Patience and trust in Allah. "
        "Jealousy. Forgiveness. Allah's plan."
    )),

    ("Ibrahim", (
        "Ibrahim. Abraham. Prayer of Ibrahim for Mecca. "
        "Gratitude shukr. Ingratitude kufr. "
        "Parable of good word tree. Parable of evil word. "
        "Day of Judgment. Intercession. Hellfire."
    )),

    ("Al-Kahf", (
        "Al-Kahf. Cave. Companions of the cave Ashaab al-Kahf. "
        "Story of Khidr and Moses. Story of Dhul-Qarnayn. Gog Magog Yajuj Majuj. "
        "Parable of rich man and poor man. Garden parable. "
        "Dajjal protection. Read Al-Kahf on Friday. "
        "Wealth and children are trial. Say InshAllah."
    )),

    ("Maryam", (
        "Maryam. Mary. Birth of Jesus Isa. Virgin birth. "
        "Zakariyya prayer for son. Birth of Yahya John the Baptist. "
        "Jesus speaks from cradle. Jesus prophet of Allah. "
        "Ibrahim father rejects idols. Ismail Idris prophets."
    )),

    ("Ta-Ha", (
        "Ta-Ha. Moses Musa and Pharaoh Firaun. Burning bush. "
        "Staff of Moses. Parting of sea. Samiri golden calf. "
        "Adam and Eve in paradise. Iblis. "
        "Expanded chest for messenger. Ease after hardship."
    )),

    ("Al-Anbiya", (
        "Al-Anbiya. Prophets. Stories of many prophets. "
        "Ibrahim Lut Nuh Dawud Sulayman Ayyub Yunus Zakariyya Maryam Isa. "
        "Tawhid message of all prophets. Day of Judgment. "
        "We created you in pairs. Mercy to the worlds rahmatan lil alameen."
    )),

    ("Al-Mu'minun", (
        "Al-Muminun. Believers. Qualities of true believers. "
        "Humble in prayer. Avoiding vain talk. Paying zakat. "
        "Guarding private parts. Fulfilling trusts and covenants. "
        "Creation of man from clay then sperm nutfah. Stages of human development."
    )),

    ("An-Nur", (
        "An-Nur. Light. Verse of Light ayat un-nur. Allah is light of heavens and earth. "
        "Prohibition of zina fornication. Punishment for adultery. "
        "Hijab modesty for men and women. Lower your gaze. "
        "False accusation qazf. Story of Aisha. "
        "Permission to enter homes. Privacy."
    )),

    ("Al-Furqan", (
        "Al-Furqan. Criterion. Quran as criterion distinguishing truth from falsehood. "
        "Attributes of servants of the Most Merciful Ibad ur-Rahman. "
        "Those who walk humbly. Say Salam to ignorant. "
        "Night prayer tahajjud. Balance between extravagance and miserliness."
    )),

    ("Ya-Sin", (
        "Ya-Sin. Heart of the Quran. Resurrection. Day of Judgment. "
        "Signs of Allah in creation. Dead earth revived by rain. "
        "Story of three messengers to a city. "
        "Read Yasin for the dying. Importance of surah Yasin."
    )),

    ("Az-Zumar", (
        "Az-Zumar. Groups. Sincere devotion to Allah. "
        "Worship Allah alone. Do not despair of Allah's mercy. 39:53. "
        "Allah forgives all sins. Tawbah repentance. "
        "Groups on Day of Judgment to hell and paradise."
    )),

    ("Ghafir", (
        "Ghafir. Forgiver. Believer of Pharaoh's household. "
        "Secret believer defends Musa. Call upon Allah He answers. 40:60. "
        "Dua supplication. Allah responds to dua. Ask Allah for your needs."
    )),

    ("Fussilat", (
        "Fussilat. Explained in detail. Quran guidance and healing. "
        "For believers guidance and healing. For disbelievers deafness. "
        "We will show them Our signs in horizons and themselves. 41:53. "
        "Limbs will testify on Day of Judgment."
    )),

    ("Muhammad", (
        "Muhammad. Surah Muhammad. Jihad fighting in way of Allah. "
        "Believers forgiven sins. Paradise rivers of water milk honey wine. "
        "Obey Allah and messenger. Do not nullify deeds. "
        "Be not faint-hearted call for peace."
    )),

    ("Al-Hujurat", (
        "Al-Hujurat. Chambers. Islamic etiquette adab. "
        "Do not raise voices above Prophet. Verify news tabayyun. "
        "Do not mock others. Avoid suspicion. Do not spy. "
        "Backbiting gheebah like eating dead brother's flesh. "
        "All mankind created from Adam and Eve. Most noble is most pious. "
        "Believers are brothers. Make peace between brothers."
    )),

    ("Ar-Rahman", (
        "Ar-Rahman. Most Merciful. Which favours of your Lord will you deny. "
        "Fa bi ayyi ala i rabbikuma tukadhdhibaan. "
        "Blessings of Allah. Creation of man from clay. "
        "Two paradises jannah described. Rivers springs fruits. "
        "Balance mizan. Justice. Jinn and mankind."
    )),

    ("Al-Waqi'ah", (
        "Al-Waqiah. Inevitable event. Day of Judgment. "
        "Three groups: forerunners sabiqoon companions of right companions of left. "
        "Paradise described. Hell described. "
        "Read surah Waqiah for rizq provision wealth. "
        "Creation of food water fire as blessings."
    )),

    ("Al-Mulk", (
        "Al-Mulk. Sovereignty. Dominion. Blessed is He in whose hand is dominion. "
        "Created death and life to test which is best in deed. "
        "Seven heavens layers of creation. "
        "Allah knows what is concealed. Read Al-Mulk every night. "
        "Intercedes for reciter in grave. Protection from punishment of grave."
    )),

    ("Al-Insan", (
        "Al-Insan. Man. Creation of man tested. "
        "Righteous drink from cup of kafoor and zanjabeel. "
        "Reward of paradise for feeding poor orphan captive. "
        "We fed you for sake of Allah we want no reward. "
        "Patience patience of believers."
    )),

    ("An-Naba", (
        "An-Naba. Great news. Day of Judgment described. "
        "Earth as cradle mountains as pegs. Sleep as rest. "
        "Trumpet blown. Hell torment. Paradise reward. "
        "Day of Sorting out yawm al-fasl."
    )),

    ("Al-Fajr", (
        "Al-Fajr. Dawn. Oath by dawn. "
        "Stories of Aad Thamud Pharaoh destroyed for corruption. "
        "Man ungrateful in hardship. Regret on Day of Judgment. "
        "Soul at rest nafs mutmainnah return to your Lord well-pleased."
    )),

    ("Al-Ikhlas", (
        "Al-Ikhlas. Sincerity. Say He is Allah One. Ahad. "
        "Allah the Eternal Absolute Samad. He begets not nor was begotten. "
        "None equal to Him. Tawhid. Oneness of God. Pure monotheism. "
        "Equal to one third of Quran. Most important surah for tawhid."
    )),

    ("Al-Falaq", (
        "Al-Falaq. Daybreak. Seek refuge in Lord of daybreak. "
        "Evil of what He created. Evil of darkness. "
        "Evil of those who blow on knots. Evil of envier. "
        "Protection from black magic evil eye hasad. Ruqyah."
    )),

    ("An-Nas", (
        "An-Nas. Mankind. Seek refuge in Lord of mankind. "
        "King of mankind God of mankind. "
        "Evil of whispering retreating Shaytan. "
        "Who whispers in hearts of men. Jinn and men. Protection from Shaytan."
    )),

    ("Al-Qadr", (
        "Al-Qadr. Power decree. Night of Power Laylatul Qadr. "
        "Better than thousand months. Angels and Spirit descend. "
        "Peace until dawn. Last ten nights of Ramadan. Odd nights. "
        "Seek Laylatul Qadr. Dua on night of power."
    )),

    ("Al-'Asr", (
        "Al-Asr. Time. By time mankind is in loss. "
        "Except those who believe do good deeds enjoin truth and patience. "
        "Four qualities of successful person. Time is running out. "
        "Importance of faith deeds truth patience."
    )),
]


def _get_topic_preamble(section: str) -> str:
    section_lower = section.lower()
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

    (["allah", "god", "lord", "rabb", "ilah", "creator"],
     "Who is Allah. Nature of God in Islam. Tawhid oneness of Allah. "
     "Allah's names and attributes. 99 names of Allah. Asmaul Husna. "
     "Allah is one Ahad. Allah is eternal Samad. "
     "Allah all-knowing all-seeing all-hearing. Omnipotent."),

    (["prayer", "salah", "salat", "pray", "worship", "prostrate",
      "ruku", "sujud", "wudu", "ablution", "qibla", "mosque", "masjid"],
     "Salah prayer in Islam. Five daily prayers. Fajr Dhuhr Asr Maghrib Isha. "
     "How to perform salah. Importance of prayer. "
     "Prayer pillar of Islam. Wudu before prayer. Direction of Mecca. "
     "Friday prayer Jumuah. Night prayer tahajjud qiyam al-layl."),

    (["fast", "fasting", "sawm", "ramadan", "iftar", "suhoor", "suhur"],
     "Fasting in Islam. Sawm Ramadan. Pillar of Islam. "
     "Why Muslims fast. Benefits of fasting. Ramadan month of Quran. "
     "Laylatul Qadr night of power. Eid ul-Fitr after Ramadan. "
     "What breaks the fast. Intention for fasting."),

    (["zakat", "charity", "sadaqah", "giving", "poor", "needy",
      "orphan", "spending in the way of allah"],
     "Zakat in Islam. Obligatory charity. Pillar of Islam. "
     "Sadaqah voluntary charity. Giving to poor needy orphans. "
     "Spending in way of Allah. Wealth purification. "
     "Who must pay zakat. Nisab threshold. Reward for charity."),

    (["hajj", "pilgrimage", "kaaba", "mecca", "ihram", "tawaf",
      "safa marwa", "arafat", "mina", "umrah"],
     "Hajj pilgrimage to Mecca. Fifth pillar of Islam. "
     "Kaaba house of Allah. Circumambulation tawaf. "
     "Stoning of devil Jamarat. Mount Arafat. "
     "Umrah lesser pilgrimage. Ihram state of sanctity. "
     "Eid ul-Adha sacrifice after Hajj."),

    (["quran", "quran recitation", "scripture", "revelation", "kitab",
      "book", "word of allah", "recite"],
     "Quran as word of Allah. Reading reciting the Quran. "
     "Importance of Quran. Quran guidance for mankind. "
     "Quran revealed in Ramadan. Quran preserved unchanged. "
     "Learn Quran memorize Quran. Tajweed recitation rules."),

    (["prophet", "muhammad", "messenger", "rasul", "nabi",
      "sunnah", "hadith", "pbuh", "peace be upon him"],
     "Prophet Muhammad peace be upon him. Last prophet. "
     "Following the sunnah. Hadith teachings of prophet. "
     "Love for the prophet. Obey Allah and His messenger. "
     "Life of prophet Muhammad. Sirah. Character of prophet."),

    (["paradise", "heaven", "jannah", "garden", "reward", "hereafter",
      "akhira", "eternal life", "bliss"],
     "Jannah paradise in Islam. Description of paradise. "
     "Reward for believers. Gardens of paradise rivers of milk honey. "
     "Seeing Allah in paradise. Highest level Firdaus. "
     "How to enter paradise. Good deeds for jannah. Akhira hereafter."),

    (["hell", "hellfire", "jahannam", "punishment", "torment", "fire",
      "wrath", "azab"],
     "Jahannam hellfire in Islam. Punishment of hellfire. "
     "Torment of hell. Boiling water. Zaqqum tree of hell. "
     "Who goes to hell. Sins leading to hellfire. "
     "Punishment of grave. Fear of Allah's wrath."),

    (["day of judgment", "qiyamah", "resurrection", "reckoning",
      "accountability", "scales mizan", "book of deeds", "sirat bridge"],
     "Yawm al-Qiyamah Day of Judgment. Resurrection of the dead. "
     "Accountability before Allah. Scales of deeds mizan. "
     "Book of deeds. Bridge over hellfire sirat. "
     "Intercession shafaah. What happens on Day of Judgment."),

    (["sin", "repentance", "tawbah", "forgiveness", "istighfar",
      "astaghfirullah", "mercy", "rahma"],
     "Tawbah repentance in Islam. Allah forgives all sins. "
     "Seeking forgiveness istighfar. Allah's mercy rahma. "
     "Conditions of repentance. Turn back to Allah. "
     "Do not despair of Allah's mercy. 39:53."),

    (["shirk", "polytheism", "idol", "idol worship", "associating partners",
      "tawhid", "oneness"],
     "Shirk worst sin in Islam. Associating partners with Allah. "
     "Tawhid monotheism. Idol worship forbidden. "
     "Allah does not forgive shirk. Polytheism refuted in Quran. "
     "Pure monotheism Islam."),

    (["jesus", "isa", "messiah", "christ", "mary", "maryam",
      "virgin birth", "crucifixion"],
     "Isa Jesus in Islam. Prophet Isa ibn Maryam. "
     "Miraculous birth of Isa. Jesus is prophet not son of God. "
     "Isa did not die on cross. Allah raised Isa. "
     "Second coming of Isa. Maryam Mary honoured in Islam."),

    (["moses", "musa", "pharaoh", "firaun", "exodus", "parting sea",
      "torah", "tawrat"],
     "Musa Moses prophet in Islam. Moses and Pharaoh Firaun. "
     "Exodus from Egypt. Miracles of Moses staff. Parting of Red Sea. "
     "Tawrat Torah revealed to Moses. Ten Commandments. "
     "Children of Israel Bani Israel."),

    (["abraham", "ibrahim", "ismail", "ishmael", "kaaba", "sacrifice",
      "monotheism", "hanif"],
     "Ibrahim Abraham prophet in Islam. Father of prophets. "
     "Building of Kaaba with Ismail. Sacrifice of Ismail. Eid ul-Adha. "
     "Ibrahim breaks idols. Hanif pure monotheist. "
     "Friend of Allah Khalilullah."),

    (["patience", "sabr", "trial", "test", "hardship", "difficulty",
      "tribulation", "suffering"],
     "Sabr patience in Islam. Allah tests believers. "
     "Indeed with hardship comes ease. 94:5-6. "
     "Patience in hardship illness loss. Reward for patient. "
     "Do not lose hope in Allah. Trust in Allah tawakkul."),

    (["dua", "supplication", "ask allah", "pray to allah",
      "call upon", "invocation"],
     "Dua supplication in Islam. Call upon Allah He answers. 40:60. "
     "How to make dua. Conditions for dua acceptance. "
     "Times when dua is answered. Last third of night. "
     "Dua is worship ibadah. Ask only from Allah."),

    (["knowledge", "ilm", "learn", "education", "seek knowledge",
      "wisdom", "intellect", "reason"],
     "Seeking knowledge in Islam. Seek knowledge from cradle to grave. "
     "Importance of learning in Islam. First revelation iqra read. "
     "Scholars inherit prophets. Knowledge as obligation. "
     "Pen qalam first creation of Allah."),

    (["marriage", "nikah", "husband", "wife", "family", "children",
      "divorce", "talaq", "spouse"],
     "Marriage nikah in Islam. Importance of marriage. "
     "Rights of husband and wife. Mahr dowry. Treating spouse well. "
     "Children rights in Islam. Divorce talaq last resort. "
     "Family as foundation of society."),

    (["death", "dying", "soul", "ruh", "grave", "barzakh",
      "angel of death", "malak ul maut"],
     "Death in Islam. Soul ruh. Angel of death Malak ul-Maut. "
     "What happens after death. Life in grave barzakh. "
     "Every soul shall taste death. Preparation for death. "
     "Good death husn ul-khatimah. Remembrance of death."),

    (["jihad", "struggle", "strive", "way of allah", "fight",
      "greater jihad", "lesser jihad"],
     "Jihad in Islam. Striving in way of Allah. "
     "Greater jihad against one's own nafs soul. "
     "Struggle against evil. Defending truth. "
     "Justice fighting oppression. Islam religion of peace."),

    (["halal", "haram", "forbidden", "permitted", "lawful", "unlawful",
      "pork", "alcohol", "riba", "interest"],
     "Halal and haram in Islam. Permitted and forbidden. "
     "Pork forbidden. Alcohol forbidden. Riba interest forbidden. "
     "Eat what is halal and good. Islamic law sharia. "
     "Avoiding what Allah has prohibited."),
]


def _get_synonym_expansion(text: str) -> str:
    text_lower = text.lower()
    additions  = []
    for triggers, expansion in SYNONYM_EXPANSIONS:
        if any(t in text_lower for t in triggers):
            additions.append(expansion)
    return " ".join(additions)


# ─────────────────────────────────────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Islam Quran Chunk & Embed Pipeline")
print("=" * 60)

print(f"\nLoading corpus from {CORPUS_PATH.name} ...")
corpus: list[dict] = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(corpus):,} passage entries")

by_category: dict[str, int] = {}
by_surah:    dict[str, int] = {}
for item in corpus:
    cat   = item.get("category", "Unknown")
    surah = item.get("section",  "Unknown")
    by_category[cat]   = by_category.get(cat, 0) + 1
    by_surah[surah]    = by_surah.get(surah, 0) + 1

print(f"\n  By category:")
for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
    print(f"    {cat:<12} {count:>6,} passages")
print(f"\n  Surahs represented: {len(by_surah)}")

# ─────────────────────────────────────────────────────────────────────────────
# Build verse-level chunks with neighbouring-verse context in embed_text
# ─────────────────────────────────────────────────────────────────────────────
# Strategy:
#   stored text  = the single verse  (precise retrieval)
#   embed_text   = preamble + prev verse(s) + THIS verse + next verse(s) + synonyms
#                  (rich semantic embedding so FAISS finds it by topic/theme)
# This mirrors how Bible pipelines work: 1 chunk = 1 verse, ~6 236 total.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Building verse-level chunks (context window ±{CONTEXT_WINDOW}) ---")
all_chunks: list[dict] = []

# Group corpus by surah so we can look up neighbours within the same surah
from collections import defaultdict
surah_verses: dict[int, list[dict]] = defaultdict(list)
for item in corpus:
    surah_verses[item.get("surah", 0)].append(item)
# Sort each surah's verses by ayah number
for num in surah_verses:
    surah_verses[num].sort(key=lambda x: x.get("ayah", 0))

for item in tqdm(corpus, desc="Building chunks"):
    raw_text = item.get("text", "").strip()
    section  = item.get("section",  "Unknown")
    category = item.get("category", "Unknown")
    surah    = item.get("surah",    0)
    ayah     = item.get("ayah",     0)
    language = item.get("language", "en")
    source   = item.get("source",   "quran")

    if not raw_text:
        continue

    # ── Hadith entries: simple embed_text (no verse-context window needed) ────
    if category == "Hadith":
        preamble = ""
        parts = [
            f"Source: {section}. Religion: Islam. Hadith {ayah}.",
            f"Hadith: {raw_text}",
        ]
        synonyms = _get_synonym_expansion(raw_text)
        if synonyms:
            parts.append(synonyms)

        all_chunks.append({
            "text":       raw_text,
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
        continue

    # ── Quran entries: verse-level with neighbouring-verse context ────────────
    preamble = _get_topic_preamble(section)

    siblings   = surah_verses[surah]
    idx        = next((i for i, v in enumerate(siblings)
                       if v.get("ayah") == ayah), None)

    prev_texts = []
    next_texts = []
    if idx is not None:
        for offset in range(1, CONTEXT_WINDOW + 1):
            if idx - offset >= 0:
                prev_texts.insert(0, siblings[idx - offset].get("text", "").strip())
            if idx + offset < len(siblings):
                next_texts.append(siblings[idx + offset].get("text", "").strip())

    # ── Assemble embed_text ───────────────────────────────────────────────────
    parts = []

    if preamble:
        parts.append(preamble)

    parts.append(
        f"Surah: {section}. Revelation: {category}. Religion: Islam. "
        f"Quran {surah}:{ayah}."
    )

    if prev_texts:
        parts.append("Context before: " + " ".join(prev_texts))

    parts.append(f"Verse: {raw_text}")

    if next_texts:
        parts.append("Context after: " + " ".join(next_texts))

    synonyms = _get_synonym_expansion(raw_text)
    if synonyms:
        parts.append(synonyms)

    embed_text = " ".join(parts)

    all_chunks.append({
        "text":       raw_text,        # single clean verse — what gets returned
        "embed_text": embed_text,      # rich context — what gets embedded
        "book":       section,
        "section":    section,
        "category":   category,
        "surah":      surah,
        "ayah":       ayah,
        "religion":   "Islam",
        "language":   language,
        "source":     source,
    })

print(f"Total chunks produced: {len(all_chunks):,}  (1 chunk = 1 verse)")

# Chunk count by surah
chunk_by_surah: dict[str, int] = {}
for c in all_chunks:
    chunk_by_surah[c["section"]] = chunk_by_surah.get(c["section"], 0) + 1
print(f"\n  Verses by surah (top 20):")
for s, count in sorted(chunk_by_surah.items(), key=lambda x: -x[1])[:20]:
    print(f"    {s:<28} {count:>5,}")

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
    CREATE INDEX idx_book      ON chunks (book);
    CREATE INDEX idx_surah     ON chunks (surah);
    CREATE INDEX idx_category  ON chunks (category);
    CREATE INDEX idx_source    ON chunks (source);
""")

con.executemany(
    "INSERT INTO chunks (id, text, book, section, category, surah, ayah, religion, language, source) "
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
con.close()

db_mb = CHUNKS_DB_PATH.stat().st_size / 1_048_576
print(f"Saved {CHUNKS_DB_PATH.name}  ({db_mb:.1f} MB, {len(all_chunks):,} rows)")

# chunks.json backup (no embed_text)
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
print(f"\n  Input passages (quran_raw.json) : {len(corpus):,}")
print(f"  Total chunks embedded           : {len(all_chunks):,}")
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
    print(f"    {cat:<14} {count:>7,}")
print("=" * 60)