import os
import re
import time
import requests
from dotenv import load_dotenv
from retrieve import search, MIN_CHUNKS_FOR_NATIVE

# ────────────────────────────────────────────────────────────────
# Groq API settings
# ────────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# Models
MODEL_DEFAULT   = "llama-3.1-8b-instant"    # English answers
MODEL_TRANSLATE = "llama-3.3-70b-versatile"  # Translation & nuance verification (faster than Qwen3)
MODEL_SINHALA   = "qwen/qwen3-32b"           # Final SI/TA answer generation + terminology review

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "Create a .env file with:  GROQ_API_KEY=your_key_here\n"
        "Get your free key at https://console.groq.com"
    )

# ════════════════════════════════════════════════════════════════
# Language detection
# ════════════════════════════════════════════════════════════════

# Sinhala Unicode block: U+0D80–U+0DFF
_SINHALA_RE = re.compile(r"[\u0D80-\u0DFF]")
# Tamil Unicode block: U+0B80–U+0BFF
_TAMIL_RE   = re.compile(r"[\u0B80-\u0BFF]")

def _is_sinhala(text: str) -> bool:
    return bool(_SINHALA_RE.search(text))

def _is_tamil(text: str) -> bool:
    return bool(_TAMIL_RE.search(text))

def _detect_language(question: str, language: str) -> str:
    """
    Resolve the effective language code.
    Explicit selection OR presence of script characters in the question text.
    Returns: "en" | "si" | "ta"
    """
    lang = language.lower()
    if lang in ("si", "sinhala"):
        return "si"
    if lang in ("ta", "tamil"):
        return "ta"
    if _is_sinhala(question):
        return "si"
    if _is_tamil(question):
        return "ta"
    return "en"

# ════════════════════════════════════════════════════════════════
# Per-religion persona prompts  (English)
# ════════════════════════════════════════════════════════════════

_PERSONAS_EN = {
    "Buddhism": """You are a knowledgeable and compassionate Buddhist guide speaking to someone new to Buddhism.

Rules you must follow without exception:
- Answer ONLY using the scripture context provided between [Source: ...] tags. Do not use your own knowledge under any circumstances.
- If the context does not contain enough information, say exactly:
  "I do not have enough reliable scriptural context to answer this accurately."
- Use simple, clear, everyday language. Avoid Pali technical terms unless you explain them.
- Begin with the most human, relatable aspect of the teaching before going deeper.
- Keep answers concise — 3 to 5 sentences unless the question requires more.
- Do not provide personal opinions or moral judgments.
- Do not compare Buddhism with other religions.
- Do not mix teachings from other traditions.
- Maintain a calm, warm, and welcoming tone at all times.
- When referencing scripture, mention the source book naturally (e.g. "As taught in the Digha Nikaya...").

CRITICAL — Quoting rules:
- Do NOT quote specific verse references (e.g. SN 56.11, MN 22) unless those exact references appear word-for-word in the provided context.
- Do NOT reproduce or paraphrase text not present in the provided context.
- Never invent or recall quotes from memory — only use what is explicitly in the context.""",

    "Christianity": """You are a knowledgeable and compassionate Christian guide speaking to someone seeking to understand the Bible and Christian teachings.

Rules you must follow without exception:
- Answer ONLY using the scripture context provided between [Source: ...] tags. Do not use your own knowledge under any circumstances.
- If the context does not contain enough information, say exactly:
  "I do not have enough reliable scriptural context to answer this accurately."
- Use simple, clear, everyday language that is welcoming to newcomers.
- Begin with the most human, relatable aspect of the teaching before going deeper.
- Keep answers concise — 3 to 5 sentences unless the question requires more detail.
- Do not provide personal opinions or moral judgments.
- Do not compare Christianity with other religions.
- Do not mix teachings from other traditions.
- Maintain a warm, hopeful, and welcoming tone at all times.
- NEVER restate or paraphrase the user's question in your answer. Start directly with the answer.
- When referencing scripture, ONLY mention a book name if that book appears in the [Source: ...] tags above. Never cite a book from memory.

CRITICAL — Quoting rules:
- Do NOT cite specific verse references (e.g. John 3:16, Romans 8:28) unless those exact references appear word-for-word in the provided context.
- Do NOT cite or mention any book name (e.g. "1 Corinthians", "Proverbs") unless it appears in the [Source: ...] tags above.
- Do NOT reproduce or paraphrase text not present in the provided context.
- Never invent or recall verse references or book names from memory — only use what is explicitly in the context.""",
}

# ════════════════════════════════════════════════════════════════
# Per-religion persona prompts  (Sinhala)
# ════════════════════════════════════════════════════════════════

_PERSONAS_SI = {
    "Buddhism": """ඔබ බෞද්ධ ධර්මය පිළිබඳ දැනුවත්, කරුණාවන්ත මාර්ගෝපදේශකයෙකි. බෞද්ධ ධර්මයට අලුතින් හඳුනා ගන්නා අයට ඔබ කතා කරයි.

ඔබ අනිවාර්යෙන් අනුගමනය කළ යුතු නීති:
- [Source: ...] ටැග් අතර ඇති ශාස්ත්‍රීය සන්දර්භය පමණක් භාවිත කර පිළිතුරු දෙන්න. ඔබේ ස්වකීය දැනුම කිසිසේත් භාවිත නොකරන්න.
- සන්දර්භය ප්‍රමාණවත් නොවේ නම්, හරියටම මෙසේ කියන්න:
  "මෙය නිවැරදිව පිළිතුරු දීමට ප්‍රමාණවත් විශ්වාසදායක ශාස්ත්‍රීය සන්දර්භයක් මා සතු නොවේ."
- සරල, පැහැදිලි, දෛනික භාෂාව භාවිත කරන්න. පාලි තාක්ෂණික පද භාවිත කරන්නේ නම් ඒවා පැහැදිලි කරන්න.
- ගැඹුරට යාමට පෙර ඉගැන්වීමේ වඩාත්ම මානවීය, සාපේක්ෂ අංශයෙන් ආරම්භ කරන්න.
- ප්‍රශ්නයට වැඩිපුර අවශ්‍ය නැතිනම් පිළිතුරු සංක්ෂිප්ත රාඛන්න — වාක්‍ය 3 සිට 5 දක්වා.
- පෞද්ගලික මත හෝ සදාචාරාත්මක විනිශ්චය ඉදිරිපත් නොකරන්න.
- බුදු දහම වෙනත් ආගම් සමඟ සසසඳන්න එපා.
- වෙනත් සම්ප්‍රදායන්ගේ ඉගැන්වීම් මිශ්‍ර නොකරන්න.
- සෑම විටෙකම සන්සුන්, උණුසුම් හා සාදරයෙන් පිළිගන්නා ස්වරයක් පවත්වා ගන්න.
- ශාස්ත්‍රය යොමු කරන විට ප්‍රභව ග්‍රන්ථය ස්වාභාවිකව සඳහන් කරන්න (උදා. "දීඝ නිකාය හි දැක්වෙන පරිදි...").

තීරණාත්මක — උද්ධෘත නීති:
- ලබා දී ඇති සන්දර්භයේ හරියටම ඒ යොමු දිස් නොවේ නම්, නිශ්චිත වාක්‍ය යොමු (SN 56.11, MN 22 වැනි) උද්ධෘත නොකරන්න.
- ලබා දී ඇති සන්දර්භයේ නැති පාඨ නැවත නිෂ්පාදනය නොකරන්න හෝ ප්‍රතිනිර්මාණය නොකරන්න.
- කිසිදා මතකයෙන් උද්ධෘත නිර්මාණය නොකරන්න — සන්දර්භයේ ඇති දෙය පමණක් භාවිත කරන්න.""",

    "Christianity": """ඔබ ක්‍රිස්තියානි ධර්මය පිළිබඳ දැනුවත්, කරුණාවන්ත මාර්ගෝපදේශකයෙකි. බයිබලය හා ක්‍රිස්තියානි ඉගැන්වීම් තේරුම් ගැනීමට ඉල්ලා සිටින අයට ඔබ කතා කරයි.

ඔබ අනිවාර්යෙන් අනුගමනය කළ යුතු නීති:
- [Source: ...] ටැග් අතර ඇති ශාස්ත්‍රීය සන්දර්භය පමණක් භාවිත කර පිළිතුරු දෙන්න. ඔබේ ස්වකීය දැනුම කිසිසේත් භාවිත නොකරන්න.
- සන්දර්භය ප්‍රමාණවත් නොවේ නම්, හරියටම මෙසේ කියන්න:
  "මෙය නිවැරදිව පිළිතුරු දීමට ප්‍රමාණවත් විශ්වාසදායක ශාස්ත්‍රීය සන්දර්භයක් මා සතු නොවේ."
- අලුත් අය සාදරයෙන් පිළිගන්නා සරල, පැහැදිලි, දෛනික භාෂාව භාවිත කරන්න.
- ගැඹුරට යාමට පෙර ඉගැන්වීමේ වඩාත්ම මානවීය, සාපේක්ෂ අංශයෙන් ආරම්භ කරන්න.
- ප්‍රශ්නයට වැඩිපුර අවශ්‍ය නැතිනම් පිළිතුරු සංක්ෂිප්ත රාඛන්න — වාක්‍ය 3 සිට 5 දක්වා.
- පෞද්ගලික මත හෝ සදාචාරාත්මක විනිශ්චය ඉදිරිපත් නොකරන්න.
- ක්‍රිස්තියානි ධර්මය වෙනත් ආගම් සමඟ සසඳන්න එපා.
- වෙනත් සම්ප්‍රදායන්ගේ ඉගැන්වීම් මිශ්‍ර නොකරන්න.
- සෑම විටෙකම උණුසුම්, බලාපොරොත්තු සහිත හා සාදරයෙන් පිළිගන්නා ස්වරයක් පවත්වා ගන්න.
- පිළිතුරේ ආරම්භයේ පරිශීලකයාගේ ප්‍රශ්නය නැවත සඳහන් කිරීම හෝ පරිවර්තනය කිරීම කිසිවිටෙකත් නොකරන්න. පිළිතුරෙන් කෙලින්ම ආරම්භ කරන්න.
- ශාස්ත්‍රය යොමු කරන විට ඉහත [Source: ...] ටැග් වල දිස්වන ග්‍රන්ථ නාම පමණක් සඳහන් කරන්න. මතකයෙන් කිසිදු ග්‍රන්ථ නාමයක් සඳහන් නොකරන්න.

තීරණාත්මක — උද්ධෘත නීති:
- ලබා දී ඇති සන්දර්භයේ හරියටම ඒ යොමු දිස් නොවේ නම්, නිශ්චිත වාක්‍ය යොමු (John 3:16, Romans 8:28 වැනි) උද්ධෘත නොකරන්න.
- ඉහත [Source: ...] ටැග් වල දිස් නොවන කිසිදු ග්‍රන්ථ නාමයක් (1 කොරින්ති, හිතෝපදේශ, 1 යොහන් ආදී) සඳහන් නොකරන්න.
- ලබා දී ඇති සන්දර්භයේ නැති පාඨ නැවත නිෂ්පාදනය නොකරන්න හෝ ප්‍රතිනිර්මාණය නොකරන්න.
- කිසිදා මතකයෙන් උද්ධෘත හෝ ග්‍රන්ථ නාම නිර්මාණය නොකරන්න — සන්දර්භයේ ඇති දෙය පමණක් භාවිත කරන්න.""",
}

# ════════════════════════════════════════════════════════════════
# Moderation config
# ════════════════════════════════════════════════════════════════

_UNSAFE_PHRASES = [
    "which religion is correct", "which religion is better",
    "which religion is true", "which religion is superior",
    "best religion", "true religion", "only true religion",
    "hate", "kill", "destroy", "inferior religion", "false religion",
]

_COMPARATIVE_PATTERNS = [
    r"\b(buddhism|christianity|islam|hinduism)\b.{0,30}\b(better|worse|superior|inferior|correct|wrong|true|false)\b",
    r"\b(better|worse|superior|inferior|correct|wrong|true|false)\b.{0,30}\b(buddhism|christianity|islam|hinduism)\b",
    r"compare.{0,20}(religion|buddhism|christianity|islam|hinduism)",
    r"which (god|religion|faith|belief).{0,20}(right|correct|true|real|better)",
]

_HATE_PATTERNS = [
    r"\b(hate|despise|destroy|eliminate|eradicate).{0,20}(religion|muslim|christian|hindu|buddhist|jew)\b",
    r"\b(religion|faith).{0,20}(cancer|disease|evil|poison|lie|scam|fraud)\b",
]

_OTHER_RELIGION_TERMS = {
    "Buddhism": [
        "christianity", "islam", "hinduism", "judaism", "sikhism",
        "quran", "bible", "torah", "vedas", "gita",
        "jesus", "allah", "brahma", "vishnu", "shiva", "moses",
        "church", "mosque",
    ],
    "Christianity": [
        "buddhism", "islam", "hinduism", "judaism", "sikhism",
        "quran", "torah", "vedas", "gita", "tipitaka",
        "buddha", "allah", "brahma", "vishnu", "shiva",
        "mosque", "synagogue",
    ],
}

# ── Cross-religion INPUT detection ────────────────────────────────
# Blocks questions that ask about OTHER religions in the query itself.
_CROSS_RELIGION_QUERY_TERMS = {
    "Buddhism": [
        "islam", "muslim", "quran", "christianity", "christian", "bible",
        "hinduism", "hindu", "vedas", "gita", "judaism", "jewish", "torah",
        "sikhism", "sikh", "jesus", "allah", "vishnu", "brahma", "shiva",
        "prophet muhammad", "mohammed",
    ],
    "Christianity": [
        "islam", "muslim", "quran", "buddhism", "buddhist", "tipitaka",
        "hinduism", "hindu", "vedas", "gita", "judaism", "jewish", "torah",
        "sikhism", "sikh", "allah", "buddha", "vishnu", "brahma", "shiva",
        "prophet muhammad", "mohammed",
    ],
}

# Patterns that explicitly ask what another religion says/teaches
_CROSS_RELIGION_ASK_PATTERNS = [
    r"\bwhat does (islam|buddhism|hinduism|judaism|sikhism|christianity)\b.{0,40}(say|teach|believe|think|claim|state)",
    r"\baccording to (islam|buddhism|hinduism|judaism|sikhism|the quran|the bible|the torah|the vedas)\b",
    r"\b(islam|buddhism|hinduism|judaism|sikhism)'s (view|teaching|belief|perspective|stance)\b",
    r"\bin (islam|buddhism|hinduism|judaism|sikhism)\b.{0,30}(jesus|god|salvation|heaven|sin|prayer)",
    r"\b(quran|torah|vedas|gita|tipitaka)\b.{0,30}(say|teach|mention|describe|speak)",
]

_FALLBACK_MESSAGES = {
    "en": {
        "comparative_religion": (
            "Questions comparing religions fall outside the scope of this guide. "
            "Please ask about a specific teaching or scripture passage."
        ),
        "hate_speech": (
            "This question contains language that is not appropriate for a respectful "
            "religious discussion. Please rephrase your question."
        ),
        "comparative_or_unsafe": (
            "This question may cause religious conflict and cannot be answered. "
            "Please ask about the selected religion's teachings specifically."
        ),
        "cross_religion_query": (
            "This guide is focused on the selected religion's own scripture and teachings. "
            "Questions about other religions or their scriptures cannot be answered here. "
            "Please ask about this religion's specific teachings."
        ),
        "no_context": "I do not have enough reliable scriptural context to answer this accurately.",
    },
    "si": {
        "comparative_religion": (
            "ආගම් සසසඳන ප්‍රශ්න මෙම මාර්ගෝපදේශයේ විෂය පථයෙන් බැහැරය. "
            "කරුණාකර නිශ්චිත ඉගැන්වීමක් හෝ ශාස්ත්‍රීය ඡේදයක් ගැන අසන්න."
        ),
        "hate_speech": (
            "මෙම ප්‍රශ්නය ගෞරවනීය ආගමික සාකච්ඡාවකට නුසුදුසු භාෂාව අඩංගු කරයි. "
            "කරුණාකර ඔබේ ප්‍රශ්නය නැවත සකස් කරන්න."
        ),
        "comparative_or_unsafe": (
            "මෙම ප්‍රශ්නය ආගමික ගැටුම් ඇති කළ හැකි බැවින් පිළිතුරු දිය නොහැක. "
            "කරුණාකර තෝරාගත් ආගමේ ඉගැන්වීම් ගැන නිශ්චිතව අසන්න."
        ),
        "cross_religion_query": (
            "මෙම මාර්ගෝපදේශය තෝරාගත් ආගමේ ශාස්ත්‍රය සහ ඉගැන්වීම් කෙරෙහි පමණක් අවධානය යොමු කරයි. "
            "වෙනත් ආගම් ගැන ප්‍රශ්න මෙහිදී පිළිතුරු දිය නොහැක. "
            "කරුණාකර මෙම ආගමේ ඉගැන්වීම් ගැන නිශ්චිතව අසන්න."
        ),
        "no_context": "මෙය නිවැරදිව පිළිතුරු දීමට ප්‍රමාණවත් විශ්වාසදායක ශාස්ත්‍රීය සන්දර්භයක් මා සතු නොවේ.",
    },
    "ta": {
        "comparative_religion": (
            "மதங்களை ஒப்பிடும் கேள்விகள் இந்த வழிகாட்டியின் எல்லைக்கு வெளியே உள்ளன. "
            "குறிப்பிட்ட போதனை அல்லது மறைநூல் பகுதியைப் பற்றி கேட்கவும்."
        ),
        "hate_speech": (
            "இந்தக் கேள்வி மரியாதையான மத விவாதத்திற்கு பொருத்தமற்ற மொழியை கொண்டுள்ளது. "
            "உங்கள் கேள்வியை மறுசீரமைக்கவும்."
        ),
        "comparative_or_unsafe": (
            "இந்தக் கேள்வி மத மோதலை ஏற்படுத்தக்கூடும், எனவே பதிலளிக்க முடியாது. "
            "தேர்ந்தெடுக்கப்பட்ட மதத்தின் போதனைகளைப் பற்றி குறிப்பாகக் கேட்கவும்."
        ),
        "cross_religion_query": (
            "இந்த வழிகாட்டி தேர்ந்தெடுக்கப்பட்ட மதத்தின் மறைநூல் மற்றும் போதனைகளில் மட்டுமே கவனம் செலுத்துகிறது. "
            "மற்ற மதங்களைப் பற்றிய கேள்விகளுக்கு இங்கே பதிலளிக்க முடியாது. "
            "இந்த மதத்தின் குறிப்பிட்ட போதனைகளைப் பற்றி கேட்கவும்."
        ),
        "no_context": "இதை சரியாக பதிலளிக்க போதுமான நம்பகமான மறைநூல் சூழல் என்னிடம் இல்லை.",
    },
}

# ════════════════════════════════════════════════════════════════
# Greeting detection
# ════════════════════════════════════════════════════════════════

_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"(hi|hello|hey|hiya|howdy|yo)[\s,!.]*"
    r")?"
    r"("
    r"hi|hello|hey|hiya|howdy|greetings|salutations|namaste|"
    r"good\s*(morning|afternoon|evening|night|day)|"
    r"what'?s\s+up|sup|"
    # Sinhala greetings
    r"ආයුබෝවන්|හෙලෝ|හායි|ශුභ\s*(උදෑසන|සවස|රාත්‍රී)|"
    # Tamil greetings
    r"வணக்கம்|ஹலோ|ஹாய்|காலை\s*வணக்கம்|மாலை\s*வணக்கம்"
    r")"
    r"[\s!.,]*$",
    re.IGNORECASE | re.UNICODE,
)

_GREETING_RESPONSES = {
    "Buddhism": {
        "en": "Hi! 🙏 Feel free to ask any question about Buddhism.",
        "si": "ආයුබෝවන්! 🙏 බෞද්ධ ධර්මය පිළිබඳ ඕනෑම ප්‍රශ්නයක් අසන්න.",
        "ta": "வணக்கம்! 🙏 பௌத்தம் பற்றி எதுவும் கேட்கலாம்.",
    },
    "Christianity": {
        "en": "Hi! 🙏 Feel free to ask any question about Christianity.",
        "si": "ආයුබෝවන්! 🙏 ක්‍රිස්තියානි ධර්මය පිළිබඳ ඕනෑම ප්‍රශ්නයක් අසන්න.",
        "ta": "வணக்கம்! 🙏 கிறிஸ்தவம் பற்றி எதுவும் கேட்கலாம்.",
    },
}


def _is_greeting(question: str) -> bool:
    """Return True if the message is a simple greeting with no substantive question."""
    q = question.strip()
    if _GREETING_PATTERNS.match(q):
        return True
    # Catch "hi good morning", "hello good evening" — strip leading hi/hello/hey
    # then re-check the remainder against the same pattern.
    stripped = re.sub(
        r"^(hi|hello|hey|hiya|howdy|yo)[,\s!.]+",
        "", q, flags=re.IGNORECASE
    ).strip()
    if stripped and stripped != q:
        return bool(_GREETING_PATTERNS.match(stripped))
    return False


_VERSE_REF_PATTERNS = {
    "Buddhism": re.compile(
        r"\b(SN|MN|DN|AN|KN|Dhp|Ud|Iti|Snp|Thag|Thig|Ja|Cp|Mil|Pv|Vv|Vb|Ds|Kv|Pp|Ps|Ya)\s*\d+[\.\d]*\b",
        re.IGNORECASE,
    ),
    "Christianity": re.compile(
        # ── English book names with chapter:verse ──────────────────────────
        r"\b(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|"
        r"1\s*Samuel|2\s*Samuel|1\s*Kings|2\s*Kings|1\s*Chronicles|2\s*Chronicles|"
        r"Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs?|Ecclesiastes|Song\s*of\s*Solomon|"
        r"Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|"
        r"Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|"
        r"Matthew|Mark|Luke|John|Acts|Romans|"
        r"1\s*Corinthians|2\s*Corinthians|Galatians|Ephesians|Philippians|"
        r"Colossians|1\s*Thessalonians|2\s*Thessalonians|1\s*Timothy|2\s*Timothy|"
        r"Titus|Philemon|Hebrews|James|1\s*Peter|2\s*Peter|"
        r"1\s*John|2\s*John|3\s*John|Jude|Revelation|Rev)\s+\d+:\d+(?:-\d+)?\b"
        # ── Sinhala book names followed by chapter:verse ───────────────────
        r"|(?:"
        r"උත්පත්ති|පිටවීම|ලෙවිවිවරණය|ගණනය කිරීම|ද්විතීය කථාව|"
        r"යෝෂුවා|විනිසුරන්|රූත්|1\s*සාමுවෙල්|2\s*සාමුවෙල්|"
        r"1\s*රාජාවලිය|2\s*රාජාවලිය|ගීතාවලිය|හිතෝපදේශ|ප්‍රේරිතයන්ගේ|"
        r"රෝමවරුන්ට|1\s*කොරින්ති|2\s*කොරින්ති|ගලාතිවරුන්ට|එපෙසි|"
        r"පිලිප්පිවරුන්ට|කොලොස්සි|1\s*තෙස්සලෝනිකෙවරුන්ට|"
        r"2\s*තෙස්සලෝනිකෙවරුන්ට|1\s*තිමෝතිවරුන්ට|2\s*තිමෝතිවරුන්ට|"
        r"තීතස්|ෆිලේමොන්|හෙබ්‍රෙව්|යාකොබ්|1\s*පේතෘස්|2\s*පේතෘස්|"
        r"1\s*යොහන්|2\s*යොහන්|3\s*යොහන්|යූදස්|එළිදරව්ව|"
        r"මත්තෙව්|මාර්ක|ලූකස්|යොහන්|ඊශාය|යෙරෙමියා|එසෙකියෙල්|දානියෙල්"
        r")\s+\d+[:\.\s]\d+(?:-\d+)?"
        # ── Tamil book names followed by chapter:verse ─────────────────────
        r"|(?:"
        r"ஆதியாகமம்|யாத்திராகமம்|லேவியராகமம்|எண்ணாகமம்|உபாகமம்|"
        r"யோசுவா|நியாயாதிபதிகள்|ரூத்|1\s*சாமுவேல்|2\s*சாமுவேல்|"
        r"1\s*இராஜாக்கள்|2\s*இராஜாக்கள்|சங்கீதம்|நீதிமொழிகள்|"
        r"மத்தேயு|மாற்கு|லூக்கா|யோவான்|அப்போஸ்தலர்|ரோமர்|"
        r"1\s*கொரிந்தியர்|2\s*கொரிந்தியர்|கலாத்தியர்|எபேசியர்|"
        r"பிலிப்பியர்|கொலோசெயர்|1\s*தெசலோனிக்கேயர்|"
        r"2\s*தெசலோனிக்கேயர்|1\s*தீமோத்தேயு|2\s*தீமோத்தேயு|"
        r"தீத்து|பிலேமோன்|எபிரெயர்|யாக்கோபு|1\s*பேதுரு|2\s*பேதுரு|"
        r"1\s*யோவான்|2\s*யோவான்|3\s*யோவான்|யூதா|வெளிப்படுத்தல்|"
        r"ஏசாயா|எரேமியா|எசேக்கியேல்|தானியேல்"
        r")\s+\d+[:\.\s]\d+(?:-\d+)?",
        re.UNICODE,
    ),
}

# Matches [Source: ...] or [மூலாශ்‍ர: ...] or [මූලාශ්‍රය: ...] tags that
# leak from the context prompt into the LLM answer.
_SOURCE_TAG_RE = re.compile(
    r"\[(?:Source|මූලාශ්‍රය|மூலம்)[^\]]*\]",
    re.UNICODE,
)

def _scrub_source_tags(text: str) -> str:
    """Remove any [Source: ...] / [මූලාශ්‍රය: ...] / [மூலம்: ...] tags
    that the LLM echoed back from the context prompt into its answer."""
    cleaned = _SOURCE_TAG_RE.sub("", text)
    # Tidy up any double-spaces or trailing whitespace left behind
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()

_OPINION_SIGNALS = [
    r"\bi (think|believe|feel|personally|would say)\b",
    r"\bmy (view|opinion|take|perspective)\b",
]

# ════════════════════════════════════════════════════════════════
# Moderation functions
# ════════════════════════════════════════════════════════════════

def moderate_input(query: str, religion: str = "Buddhism") -> tuple[bool, str]:
    q = query.lower().strip()

    # 1. Unsafe / hate phrases
    for phrase in _UNSAFE_PHRASES:
        if phrase in q:
            return False, "comparative_or_unsafe"

    # 2. Comparative patterns (e.g. "is Christianity better than Islam")
    for pattern in _COMPARATIVE_PATTERNS:
        if re.search(pattern, q):
            return False, "comparative_religion"

    # 3. Hate speech
    for pattern in _HATE_PATTERNS:
        if re.search(pattern, q):
            return False, "hate_speech"

    # 4. Cross-religion ask patterns (e.g. "what does Islam say about X")
    for pattern in _CROSS_RELIGION_ASK_PATTERNS:
        if re.search(pattern, q):
            return False, "cross_religion_query"

    # 5. Query contains terms belonging to a DIFFERENT religion
    #    e.g. asking "what does the Quran say" inside the Christianity chatbot
    blocked_terms = _CROSS_RELIGION_QUERY_TERMS.get(religion, [])
    for term in blocked_terms:
        if re.search(r"" + re.escape(term) + r"", q):
            return False, "cross_religion_query"

    return True, "ok"


def _check_fabricated_references(response: str, context: str, religion: str) -> list[str]:
    pattern  = _VERSE_REF_PATTERNS.get(religion)
    warnings = []
    if not pattern:
        return warnings
    for match in pattern.finditer(response):
        ref_str = match.group(0)
        if ref_str.lower() not in context.lower():
            warnings.append(f"fabricated_reference:{ref_str}")
    return warnings


def moderate_output(
    response: str, context: str, religion: str, lang: str = "en"
) -> tuple[str, list[str]]:
    # Always strip any [Source: ...] tags the LLM echoed from the context prompt
    response = _scrub_source_tags(response)
    warnings = []
    r        = response.lower()

    for term in _OTHER_RELIGION_TERMS.get(religion, []):
        if term in r:
            warnings.append(f"cross_religion_term:{term}")
            msg = (
                "ශාස්ත්‍රීය සන්දර්භයෙන් සම්පූර්ණ ප්‍රතිචාරයක් ජනනය කිරීමට නොහැකි විය. "
                "කරුණාකර ඔබේ ප්‍රශ්නය නැවත සකස් කරන්න."
                if lang == "si" else
                "I was unable to generate a fully grounded response from the "
                "scripture context. Please rephrase your question."
            )
            return msg, warnings

    for pattern in _OPINION_SIGNALS:
        if re.search(pattern, r):
            warnings.append("personal_opinion_detected")
            break

    if len(response.strip()) < 30:
        warnings.append("response_too_short")
        fb = _FALLBACK_MESSAGES.get(lang, _FALLBACK_MESSAGES["en"])
        return fb["no_context"], warnings

    ref_warnings = _check_fabricated_references(response, context, religion)
    if ref_warnings:
        warnings.extend(ref_warnings)
        pattern = _VERSE_REF_PATTERNS.get(religion)
        cleaned = pattern.sub("", response) if pattern else response
        cleaned = re.sub(r"\(\s*\)", "", cleaned).strip()
        return cleaned, warnings

    return response, warnings

# ════════════════════════════════════════════════════════════════
# Query translation for retrieval
# The FAISS index was built with English embeddings only.
# Translate Sinhala questions to English before embedding so scores
# are meaningful, while still returning Sinhala chunks from the DB.
# ════════════════════════════════════════════════════════════════

def _translate_query_to_english(question: str, religion: str = "Buddhism") -> str:
    """
    Translate a Sinhala or Tamil question to English for FAISS retrieval.

    Two-step process:
      Step 1 — Fast translation via llama (MODEL_DEFAULT).
      Step 2 — Nuance verification via Qwen3 (MODEL_SINHALA).
                Qwen3 checks whether the key concept/term in the original
                question was translated with its correct religious meaning.
                If not, it returns a corrected English query.

    This prevents mistranslations like:
      Buddhism:
        ஏக்கம்   → "suffering"  (wrong) should be → "longing / craving / tanha"
        ශූන්‍යතාව → "emptiness" (ok)   or → "sunyata / void" (also ok)
      Christianity:
        ගැලවීම   → "escape / rescue" (wrong) should be → "salvation"
        ත්‍රිත්වය → "three groups" (wrong) should be → "Trinity"
        இரட்சிப்பு → "rescue" (wrong) should be → "salvation"
        மன்னிப்பு → "forgiveness / pardon" — keep as "forgiveness / grace"

    Falls back to the Step 1 result if Step 2 fails.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    # ── Step 1: Fast translation ─────────────────────────────────
    _step1_system = {
        "Buddhism": (
            "You are a translator. Translate the following question into "
            "concise English suitable for a Buddhist scripture search. "
            "Preserve the precise meaning of every religious or philosophical term. "
            "Output ONLY the English translation — no explanation, no extra text."
        ),
        "Christianity": (
            "You are a translator. Translate the following Sinhala or Tamil question into "
            "concise English suitable for a Christian Bible scripture search. "
            "You MUST use the correct Christian theological English term — never a generic word.\n\n"
            "CRITICAL term mappings (use these EXACTLY):\n"
            "  ගැලවීම / இரட்சிப்பு        → salvation  (NOT escape, rescue, relief)\n"
            "  ත්‍රිත්වය / திரித்துவம்     → Trinity\n"
            "  ඇදහිල්ල / விசுவாசம்        → faith\n"
            "  පාපය / பாவம்               → sin\n"
            "  දෙවිඳු / கடவுள்            → God\n"
            "  ශුද්ධ ලියවිල්ල / பைபிள்   → Bible / scripture\n"
            "  මිදීම / மீட்பு             → redemption\n"
            "  කරුණාව / கிருபை            → grace\n"
            "  සදාකාලික ජීවිතය / நித்திய ஜீவன் → eternal life\n"
            "  ශුද්ධ ආත්මය / பரிசுத்த ஆவி → Holy Spirit\n"
            "  යාච්ඤාව / ஜெபம்            → prayer\n"
            "  සමාව / மன்னிப்பு           → forgiveness\n"
            "  පශ්චාත්තාපය / மனந்திரும்புதல் → repentance\n"
            "  ජේසුස් / இயேசு             → Jesus\n\n"
            "Output ONLY the English translation — no explanation, no extra text."
        ),
    }
    step1_system = _step1_system.get(religion, _step1_system["Buddhism"])
    payload_1 = {
        "model":    MODEL_DEFAULT,
        "messages": [
            {
                "role":    "system",
                "content": step1_system,
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens":  80,
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload_1, timeout=15)
        resp.raise_for_status()
        step1 = resp.json()["choices"][0]["message"]["content"].strip()
        step1 = re.sub(r"<think>.*?</think>", "", step1, flags=re.DOTALL).strip()
        step1 = step1 or question
    except Exception:
        step1 = question   # will be caught by the script-check below

    # ── Step 1b: Retry with a stronger model if llama returned untranslated text ──
    # Happens when llama echoes the Sinhala/Tamil question unchanged or empty.
    if not step1 or _is_sinhala(step1) or _is_tamil(step1) or step1 == question:
        print(f"  [translate] Step1 returned untranslated text — retrying with {MODEL_TRANSLATE!r}")
        retry_system = (
            "You are a professional translator. "
            "The user will give you a question written in Sinhala or Tamil script. "
            "Your ONLY job is to output the English translation of that question. "
            "Output ONLY English — absolutely no Sinhala, Tamil, or other scripts. "
            "No explanations, no original text, no quotes. Just the English translation."
        )
        payload_retry = {
            "model":    MODEL_TRANSLATE,
            "messages": [
                {"role": "system", "content": retry_system},
                {"role": "user",   "content": question},
            ],
            "temperature": 0.0,
            "max_tokens":  80,
        }
        try:
            resp2 = requests.post(GROQ_URL, headers=headers, json=payload_retry, timeout=20)
            resp2.raise_for_status()
            retry_result = resp2.json()["choices"][0]["message"]["content"].strip()
            retry_result = re.sub(r"<think>.*?</think>", "", retry_result, flags=re.DOTALL).strip()
            if retry_result and not _is_sinhala(retry_result) and not _is_tamil(retry_result):
                print(f"  [translate] Retry succeeded: {retry_result!r}")
                step1 = retry_result
            else:
                print(f"  [translate] Retry also failed — using original question for search")
                return question
        except Exception:
            print(f"  [translate] Retry exception — using original question for search")
            return question

    # ── Step 2: Nuance verification ──────────────────────────────
    # Detect source language for the verification prompt
    src_lang = "Tamil" if _is_tamil(question) else "Sinhala"

    if religion == "Christianity":
        _christianity_glossary = (
            "═══ TAMIL → ENGLISH CHRISTIAN GLOSSARY ═══\n\n"

            "CORE THEOLOGY:\n"
            "  இரட்சிப்பு       → salvation — NOT 'rescue' or 'escape'\n"
            "  திரித்துவம்      → Trinity — Father, Son, Holy Spirit — NOT 'three groups'\n"
            "  விசுவாசம்        → faith / belief — NOT just 'trust'\n"
            "  கிருபை           → grace — unmerited divine favour — NOT 'kindness' alone\n"
            "  பாவம்            → sin — NOT 'mistake' or 'fault'\n"
            "  மன்னிப்பு        → forgiveness — NOT 'pardon' in a legal sense\n"
            "  மீட்பு           → redemption — NOT 'recovery'\n"
            "  நித்திய ஜீவன்    → eternal life — NOT 'immortality'\n"
            "  நரகம்            → hell — NOT 'fire'\n"
            "  சொர்க்கம்        → heaven — NOT 'sky'\n\n"

            "GOD AND JESUS:\n"
            "  கடவுள்           → God\n"
            "  இயேசு கிறிஸ்து  → Jesus Christ\n"
            "  கிறிஸ்து         → Christ — the anointed one — NOT a name alone\n"
            "  ஆண்டவர்         → Lord\n"
            "  பரிசுத்த ஆவி    → Holy Spirit — NOT 'holy ghost' in modern context\n"
            "  தேவன்            → God (formal/literary)\n"
            "  மகன்             → Son (of God) — NOT just 'boy' or 'child'\n\n"

            "SACRAMENTS AND PRACTICE:\n"
            "  ஞானஸ்நானம்      → baptism — NOT 'bath' or 'washing'\n"
            "  திருவிருந்து     → Holy Communion / Eucharist — NOT 'feast'\n"
            "  ஜெபம்            → prayer — NOT 'request'\n"
            "  வழிபாடு         → worship\n"
            "  உபவாசம்         → fasting\n"
            "  மனந்திரும்புதல் → repentance — NOT 'changing one's mind'\n\n"

            "SCRIPTURE AND CHURCH:\n"
            "  பைபிள்           → Bible\n"
            "  புதிய ஏற்பாடு   → New Testament\n"
            "  பழைய ஏற்பாடு   → Old Testament\n"
            "  சுவிசேஷம்       → Gospel — NOT 'good news story'\n"
            "  திருச்சபை       → Church — NOT 'building'\n"
            "  போதகர்           → pastor / preacher\n\n"

            "═══ SINHALA → ENGLISH CHRISTIAN GLOSSARY ═══\n\n"

            "CORE THEOLOGY:\n"
            "  ගැලවීම           → salvation — NOT 'escape', 'rescue', or 'relief'\n"
            "  ත්‍රිත්වය        → Trinity — Father, Son, Holy Spirit — NOT 'three groups' or 'trio'\n"
            "  ඇදහිල්ල         → faith — NOT just 'belief' or 'trust'\n"
            "  කරුණාව           → grace — divine unmerited favour — NOT just 'mercy'\n"
            "  පාපය             → sin — NOT 'mistake' or 'wrongdoing' generically\n"
            "  සමාව             → forgiveness — NOT 'excuse'\n"
            "  මිදීම            → redemption — NOT 'freedom' alone\n"
            "  සදාකාලික ජීවිතය → eternal life — NOT 'everlasting living'\n"
            "  නිරය             → hell — NOT 'underworld'\n"
            "  ස්වර්ගය          → heaven — NOT 'sky'\n\n"

            "GOD AND JESUS:\n"
            "  දෙවිඳු / දෙවියන් → God\n"
            "  යේසුස් ක්‍රිස්තුස් → Jesus Christ\n"
            "  ක්‍රිස්තුස්      → Christ — the anointed one\n"
            "  ස්වාමීන්         → Lord\n"
            "  පරිශුද්ධ ආත්මය  → Holy Spirit — NOT 'holy soul' or 'pure soul'\n"
            "  පුත්‍රයා         → Son (of God) — NOT just 'son' or 'child'\n\n"

            "SACRAMENTS AND PRACTICE:\n"
            "  බාප්තිස්මය       → baptism — NOT 'bathing' or 'washing'\n"
            "  ශුද්ධ කොමියුනියන් → Holy Communion / Eucharist — NOT 'holy meal'\n"
            "  යාච්ඤාව          → prayer — NOT 'request' or 'begging'\n"
            "  නමස්කාරය         → worship\n"
            "  උපවාසය           → fasting\n"
            "  පශ්චාත්තාපය      → repentance — NOT 'regret' alone\n\n"

            "SCRIPTURE AND CHURCH:\n"
            "  ශුද්ධ ලියවිල්ල  → Bible / scripture — NOT 'holy writing' generically\n"
            "  නව ගිවිසුම        → New Testament\n"
            "  පැරණි ගිවිසුම    → Old Testament\n"
            "  සුවිශේෂය          → Gospel — NOT 'good news' alone\n"
            "  සභාව              → Church — NOT just 'assembly' or 'meeting'\n"
            "  පාස්තෝරතුමා       → pastor\n"
        )

        step2_system = (
            f"You are a Christian theology scholar fluent in English, Sinhala, and Tamil.\n\n"
            f"Your task: verify that an English translation of a Christian question "
            f"correctly preserves the theological meaning of the key term.\n\n"
            f"Rules:\n"
            f"1. Identify the main Christian concept or term in the original {src_lang} question.\n"
            f"2. Check if the English translation captures its correct Christian theological meaning "
            f"using the reference glossary below.\n"
            f"3. If the translation is correct, return it UNCHANGED.\n"
            f"4. If the translation is wrong or loses important theological nuance, return a corrected "
            f"English query that will retrieve the right Bible passages.\n"
            f"5. Output ONLY the final English query — no explanation, no commentary.\n\n"
            f"{_christianity_glossary}"
        )
    else:
        # Buddhism (original prompt)
        step2_system = (
            "You are a Buddhist scholar fluent in Pali, English, Sinhala, and Tamil.\n\n"
            "Your task: verify that an English translation of a Buddhist question "
            "correctly preserves the nuance of the key concept or term.\n\n"
            "Rules:\n"
            "1. Identify the main Buddhist concept or term in the original question.\n"
            "2. Check if the English translation captures its correct Buddhist meaning "
            "using the reference glossary below.\n"
            "3. If the translation is correct, return it UNCHANGED.\n"
            "4. If the translation is wrong or loses important nuance, return a corrected "
            "English query that will retrieve the right Buddhist scripture passages.\n"
            "5. Output ONLY the final English query — no explanation, no commentary.\n\n"

            "═══ TAMIL → PALI / ENGLISH GLOSSARY ═══\n\n"

                    "THREE MARKS OF EXISTENCE:\n"
                    "  அனிச்சை       → impermanence (anicca) — NOT craving, NOT suffering\n"
                    "  துக்கம்       → suffering / unsatisfactoriness (dukkha) — NOT craving\n"
                    "  அனாத்தா      → no-self / non-self (anattā) — NOT soul\n\n"

                    "FOUR NOBLE TRUTHS:\n"
                    "  நான்கு ஆரிய சத்தியங்கள் → Four Noble Truths (cattāri ariyasaccāni)\n"
                    "  துக்க சத்தியம்           → truth of suffering (dukkha-sacca)\n"
                    "  சமுதய சத்தியம்           → truth of the origin of suffering (samudaya-sacca)\n"
                    "  நிரோத சத்தியம்           → truth of the cessation of suffering (nirodha-sacca)\n"
                    "  மார்க்க சத்தியம்         → truth of the path (magga-sacca)\n\n"

                    "CRAVING AND CLINGING:\n"
                    "  தண்ணா / ஏக்கம்  → craving / taṇhā — longing that leads to rebirth\n"
                    "  உபாதானம்        → clinging / attachment (upādāna)\n"
                    "  ராகம்           → passion / greed / lust (rāga)\n"
                    "  லோபம்           → greed (lobha)\n"
                    "  துவேஷம்         → hatred / aversion (dosa)\n"
                    "  மோகம்           → delusion / ignorance (moha)\n"
                    "  அவிஜ்ஜா        → ignorance (avijjā) — root of suffering\n\n"

                    "THE PATH:\n"
                    "  அஷ்டாங்க மார்க்கம் → Noble Eightfold Path (aṭṭhaṅgika-magga)\n"
                    "  சம்மா தீட்டி      → right view (sammā-diṭṭhi)\n"
                    "  சம்மா சங்கப்பம்   → right intention (sammā-saṅkappa)\n"
                    "  சம்மா வாக்கு      → right speech (sammā-vācā)\n"
                    "  சம்மா கம்மந்தம்   → right action (sammā-kammanta)\n"
                    "  சம்மா ஆஜீவம்     → right livelihood (sammā-ājīva)\n"
                    "  சம்மா வாயாமம்    → right effort (sammā-vāyāma)\n"
                    "  சம்மா சதி        → right mindfulness (sammā-sati)\n"
                    "  சம்மா சமாதி      → right concentration (sammā-samādhi)\n\n"

                    "MEDITATION:\n"
                    "  விபஸ்ஸனா        → insight meditation (vipassanā)\n"
                    "  சமதா            → calm / concentration meditation (samatha)\n"
                    "  சமாதி           → concentration / meditative absorption (samādhi)\n"
                    "  ஜானம்           → meditative absorption / jhāna\n"
                    "  சதி             → mindfulness (sati)\n"
                    "  பாவனா           → mental cultivation / meditation (bhāvanā)\n"
                    "  அனாபானசதி       → mindfulness of breathing (ānāpānasati)\n\n"

                    "LIBERATION AND AWAKENING:\n"
                    "  நிர்வாணம்       → nibbāna / nirvana — liberation, NOT a place or heaven\n"
                    "  போதி            → awakening / enlightenment (bodhi)\n"
                    "  விமுக்தி        → liberation / release (vimutti)\n"
                    "  அரஹந்த்         → arahant — fully enlightened being\n"
                    "  போதிசத்வம்      → bodhisattva — being on path to full Buddhahood\n"
                    "  பரிநிர்வாணம்    → parinibbāna — final liberation at death\n\n"

                    "DEPENDENT ORIGINATION AND REALITY:\n"
                    "  பதிச்சசமுப்பாதம் → dependent origination (paṭicca-samuppāda)\n"
                    "  சூன்யதா         → emptiness (suññatā) — NOT nothingness\n"
                    "  சங்காரம்        → formations / conditioned phenomena (saṅkhāra)\n"
                    "  நாமரூபம்        → name-and-form / mind-matter (nāmarūpa)\n"
                    "  விஞ்ஞானம்       → consciousness (viññāna)\n"
                    "  வேதனா           → feeling / sensation (vedanā) — pleasant/unpleasant/neutral\n"
                    "  சஞ்ஞா           → perception (saññā)\n"
                    "  சேதனா           → volition / intention (cetanā)\n"
                    "  பஞ்சக்கந்தங்கள் → five aggregates (pañcakkhandhā)\n"
                    "  ரூபம்           → form / materiality (rūpa)\n\n"

                    "ETHICS AND CONDUCT:\n"
                    "  கர்மா / கம்மம்  → karma — intentional action, NOT fate\n"
                    "  சீலம்           → virtue / ethical conduct (sīla)\n"
                    "  பஞ்சசீலம்       → five precepts (pañcasīla)\n"
                    "  தானம்           → generosity / giving (dāna)\n"
                    "  அஹிம்சா        → non-harming / non-violence (ahiṃsā)\n"
                    "  மெட்டா          → loving-kindness (mettā)\n"
                    "  கருணை           → compassion (karuṇā)\n"
                    "  முதிதா          → sympathetic joy (muditā)\n"
                    "  உபேக்கா         → equanimity (upekkhā)\n"
                    "  பிரஜ்ஞா         → wisdom / insight (paññā)\n\n"

                    "REBIRTH AND EXISTENCE:\n"
                    "  மறுபிறவி        → rebirth (punabbhava)\n"
                    "  சம்சாரம்        → cycle of rebirth (saṃsāra)\n"
                    "  புனர்ஜன்மம்     → rebirth / reincarnation\n"
                    "  கதி             → destination / realm of rebirth (gati)\n"
                    "  பவம்            → existence / becoming (bhava)\n\n"

                    "THE THREE JEWELS:\n"
                    "  திரிரத்தினம்    → Three Jewels (tiratana)\n"
                    "  புத்தர்         → the Buddha — the awakened one\n"
                    "  தம்மம்          → Dhamma — the teaching / the truth\n"
                    "  சங்கம்          → Sangha — the community of practitioners\n\n"

                    "SCRIPTURE:\n"
                    "  திரிபிடகம்      → Tipitaka — the Pali Canon (three baskets)\n"
                    "  சுத்த பிடகம்    → Sutta Pitaka — discourses of the Buddha\n"
                    "  வினய பிடகம்     → Vinaya Pitaka — monastic rules\n"
                    "  அபிதம்ம பிடகம் → Abhidhamma Pitaka — higher teaching / philosophy\n"
                    "  நிகாயம்         → Nikāya — collection of suttas\n\n"

                    "═══ SINHALA → PALI / ENGLISH GLOSSARY ═══\n\n"

                    "THREE MARKS:\n"
                    "  අනිත්‍ය (anithya)   → impermanence (anicca) — NOT craving\n"
                    "  දුක්ඛ (dukkha)     → suffering / unsatisfactoriness\n"
                    "  අනාත්ම (anathma)   → no-self (anattā)\n\n"

                    "CRAVING AND DEFILEMENTS:\n"
                    "  තණ්හාව (tanhawa)   → craving (taṇhā)\n"
                    "  උපාදාන (upadana)   → clinging (upādāna)\n"
                    "  අවිද්‍යා (avidya)  → ignorance (avijjā)\n"
                    "  ලෝභ (lobha)        → greed\n"
                    "  ද්වේෂ (dwesha)     → hatred / aversion (dosa)\n"
                    "  මෝහ (moha)         → delusion\n\n"

                    "LIBERATION:\n"
                    "  නිර්වාණය (nirvanaya) → nibbāna — liberation, NOT a place\n"
                    "  ශූන්‍යතාව (shunyata) → emptiness (suññatā)\n"
                    "  විමුක්තිය (vimukthi)  → liberation (vimutti)\n\n"

                    "MEDITATION:\n"
                    "  සමාධිය (samadhiya)   → concentration (samādhi)\n"
                    "  සතිය (sathiya)       → mindfulness (sati)\n"
                    "  විදර්ශනා (vidarshana) → insight meditation (vipassanā)\n"
                    "  සමථ (samatha)        → calm meditation\n\n"

                    "ETHICS:\n"
                    "  කර්මය (karmaya)      → karma — intentional action\n"
                    "  ශීලය (shilaya)       → virtue / ethics (sīla)\n"
                    "  මෛත්‍රී (maithri)    → loving-kindness (mettā)\n"
                    "  කරුණා (karuna)       → compassion\n"
                    "  ප්‍රඥා (pragna)      → wisdom (paññā)\n\n"

                    "PATH:\n"
                    "  ආර්ය අෂ්ටාංගික මාර්ගය → Noble Eightfold Path\n"
                    "  සංසාරය (sansaraya)     → cycle of rebirth (saṃsāra)\n"
                    "  පටිච්චසමුප්පාද        → dependent origination\n"
                    "  ධර්මය (dharmaya)       → Dhamma — the teaching\n"
                    "  ත්‍රිපිටකය             → Tipitaka — Pali Canon\n"
        )

    payload_2 = {
        "model":    MODEL_TRANSLATE,
        "messages": [
            {
                "role":    "system",
                "content": step2_system,
            },
            {
                "role":    "user",
                "content": (
                    f"Original {src_lang} question: {question}\n"
                    f"English translation to verify: {step1}"
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens":  150,
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload_2, timeout=20)
        resp.raise_for_status()
        step2 = resp.json()["choices"][0]["message"]["content"].strip()
        step2 = re.sub(r"<think>.*?</think>", "", step2, flags=re.DOTALL).strip()
        if step2 and len(step2) > 5:
            if step2 != step1:
                print(f"  [translate] Step1: {step1!r}")
                print(f"  [translate] Step2 corrected: {step2!r}")
            return step2
    except Exception:
        pass   # step 2 failed — use step 1 result

    return step1


# ════════════════════════════════════════════════════════════════
# Groq API call
# ════════════════════════════════════════════════════════════════

def _parse_retry_after(resp: requests.Response) -> int:
    """Extract retry-after seconds from Groq 429 response headers or body."""
    # Check Retry-After header first
    retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if retry_after:
        try:
            return int(retry_after)
        except ValueError:
            pass
    # Fall back to parsing the JSON error body (Groq often includes it there)
    try:
        body = resp.json()
        msg  = str(body.get("error", {}).get("message", ""))
        # e.g. "Please try again in 30s" or "retry after 60 seconds"
        m = re.search(r"(\d+)\s*s(?:ec(?:ond)?s?)?", msg, re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 60   # safe default


def _call_groq(system_prompt: str, user_message: str, model: str, max_tokens: int = 4096) -> str:
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens":  max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)

        if resp.status_code == 401:
            return "The API key is invalid. Please check the configuration."

        if resp.status_code == 429:
            wait = _parse_retry_after(resp)
            print(f"  [info] Rate limit hit. Retry-After: {wait}s. Waiting before retry...")
            time.sleep(min(wait, 15))   # wait up to 15s inline, then retry once
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            # If still 429 after the wait, return a user-friendly message with the time
            if resp.status_code == 429:
                wait2 = _parse_retry_after(resp)
                minutes, seconds = divmod(wait2, 60)
                if minutes > 0:
                    time_str = f"{minutes} minute{'s' if minutes != 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}" if seconds else f"{minutes} minute{'s' if minutes != 1 else ''}"
                else:
                    time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
                return (
                    f"The service is currently busy due to high demand. "
                    f"Please try again in {time_str}."
                )

        resp.raise_for_status()
        data          = resp.json()
        finish_reason = data["choices"][0].get("finish_reason", "")
        if finish_reason == "length":
            print(f"  [warn] Groq response cut off at max_tokens={max_tokens} (finish_reason=length)")
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "Cannot connect to the service. Please check your internet connection and try again."
    except Exception as e:
        return f"Something went wrong while generating the answer: {e}"

# ════════════════════════════════════════════════════════════════
# Retrieval post-processing
# ════════════════════════════════════════════════════════════════

def _unique_sources(results: list[dict]) -> tuple[list[str], list[float]]:
    """
    Return deduplicated (sources, scores) lists — one entry per unique book name,
    keeping the highest score for that book.  Preserves order of first appearance.
    """
    seen   = {}   # book -> best score
    order  = []   # insertion order
    for r in results:
        book  = r["book"]
        score = r["score"]
        if book not in seen:
            seen[book] = score
            order.append(book)
        else:
            seen[book] = max(seen[book], score)
    return order, [seen[b] for b in order]

def _refine_results(results: list[dict], religion: str) -> list[dict]:
    book_counts      = {}
    seen_text_hashes = set()
    refined          = []
    MAX_PER_BOOK     = 2
    MAX_TOTAL        = 6

    def _priority(r: dict) -> int:
        if religion == "Buddhism":
            return 0 if r.get("pitaka", "").lower() == "sutta pitaka" else 1
        elif religion == "Christianity":
            genre = r.get("pitaka", r.get("testament", "")).lower()
            if "gospel" in genre:
                return 0
            if r.get("testament") == "New Testament":
                return 1
            return 2
        return 0

    for r in sorted(results, key=lambda r: (_priority(r), -r["score"])):
        text_hash = hash(r["text"][:200])
        if text_hash in seen_text_hashes:
            continue
        book = r["book"]
        if book_counts.get(book, 0) >= MAX_PER_BOOK:
            continue
        seen_text_hashes.add(text_hash)
        book_counts[book] = book_counts.get(book, 0) + 1
        refined.append(r)
        if len(refined) >= MAX_TOTAL:
            break

    return refined

# ════════════════════════════════════════════════════════════════
# Intent detection
# ════════════════════════════════════════════════════════════════

# Only match when the user explicitly asks for a list or an enumeration of books/texts.
# Deliberately excludes broad triggers like "what are" or "give me" which appear
# in normal questions (e.g. "what are the differences between...").
_LIST_PATTERNS = [
    # Explicit list/enumeration requests
    r"\b(list|enumerate)\b",
    # Asking specifically for book/text/sutta names
    r"\blist (of )?(books?|texts?|suttas?|passages?|sources?)\b",
    r"\bwhich (books?|texts?|suttas?|passages?)\b",
    r"\b(name|show) (the |all )?(books?|texts?|suttas?|passages?)\b",
    r"\ball (the )?(books?|texts?|suttas?|passages?)\b",
    r"\bbooks? (about|on|that|related)\b",
    # Sinhala explicit list requests
    r"ලැයිස්තුව",        # "list" in Sinhala
    r"ග්‍රන්ථ මොනවාද",   # "what are the books"
    r"ග්‍රන්ථ ලැයිස්තු",  # "list of books"
    r"පොත් මොනවාද",      # "what are the books"
    r"සූත්‍ර මොනවාද",    # "what are the suttas"
]

def _is_list_request(query: str) -> bool:
    return any(re.search(p, query.lower()) for p in _LIST_PATTERNS)

# ════════════════════════════════════════════════════════════════
# Format instructions
# ════════════════════════════════════════════════════════════════

def _format_instructions(religion: str, is_list: bool, lang: str) -> str:
    if is_list:
        if lang == "si":
            return (
                "- පරිශීලකයා ලැයිස්තුවක් ඉල්ලා සිටී. සන්දර්භයේ ඇති ග්‍රන්ථ/ප්‍රභව නම් "
                "සිංහලෙන් පැහැදිලි, සෘජු ලැයිස්තුවක් ලෙස ලබා දෙන්න.\n"
                "- ආකෘතිය: එක් ග්‍රන්ථයක් එක් පේළියකට, ඒකාරෝගී වාක්‍යයක් සමඟ.\n"
                "- දිගු ගද්‍ය ලිවීමෙන් වළකින්න. සංක්ෂිප්ත හා සෘජු වන්න.\n"
                "- ලබා දී ඇති ශාස්ත්‍රීය සන්දර්භයේ ඇති ග්‍රන්ථ පමණක් ඇතුළත් කරන්න."
            )
        return (
            "- The user is asking for a list. Respond with a clear, direct list of "
            "book/source names found in the context.\n"
            "- Format: one book per line, with a one-sentence description.\n"
            "- Do not write long prose. Be concise and direct.\n"
            "- Only include books present in the provided scripture context."
        )

    if lang == "si":
        if religion == "Buddhism":
            return (
                "- සිංහල භාෂාවෙන් පිළිතුරු දෙන්න.\n"
                "- ආරම්භකයෙකුට තේරෙන සරල, මානවීය පැහැදිලි කිරීමකින් ආරම්භ කරන්න.\n"
                "- ශාස්ත්‍රය ප්‍රකාශ කරන දේ ස්වාභාවිකව ප්‍රභව ග්‍රන්ථය සඳහන් කරමින් "
                "ඒ සහාය ලබා දෙන්න (උදා. \"සංයුත්ත නිකාය හි දක්නට ලැබෙන ලෙස...\").\n"
                "- ඉගැන්වීමේ ප්‍රායෝගික මානයක් ඇත්නම්, එය කෙටියෙන් සඳහන් කරන්න.\n"
                "- ලිතිකාංක හෝ ලැයිස්තු භාවිත නොකරන්න. ගලා යන, උණුසුම් ගද්‍යයෙන් ලියන්න."
            )
        return (
            "- සිංහල භාෂාවෙන් පිළිතුරු දෙන්න.\n"
            "- අලුත් අයෙකුට තේරෙන සරල, මානවීය පැහැදිලි කිරීමකින් ආරම්භ කරන්න.\n"
            "- ශාස්ත්‍රය ප්‍රකාශ කරන දේ ස්වාභාවිකව ප්‍රභව ග්‍රන්ථය සඳහන් කරමින් "
            "ඒ සහාය ලබා දෙන්න (උදා. \"යොහාන් ශුභාරංචිය හි ලියා ඇති ලෙස...\").\n"
            "- ඉගැන්වීමේ ප්‍රායෝගික හෝ අධ්‍යාත්මික මානයක් ඇත්නම්, එය කෙටියෙන් සඳහන් කරන්න.\n"
            "- ලිතිකාංක හෝ ලැයිස්තු භාවිත නොකරන්න. ගලා යන, උණුසුම් ගද්‍යයෙන් ලියන්න."
        )

    if lang == "ta":
        if religion == "Buddhism":
            return (
                "- தமிழ் மொழியில் பதில் அளிக்கவும்.\n"
                "- ஒரு தொடக்கநிலையினர் புரிந்துகொள்ளக்கூடிய எளிமையான, மனித விளக்கத்துடன் தொடங்கவும்.\n"
                "- மறைநூல் கூறுவதை இயற்கையாக ஆதாரமாக மேற்கோள் காட்டவும் "
                "(எ.கா. 'சம்யுத்த நிகாயத்தில் காணப்படுவது போல்...').\n"
                "- போதனையில் நடைமுறை பரிமாணம் இருந்தால், சுருக்கமாக குறிப்பிடவும்.\n"
                "- புள்ளிகள் அல்லது பட்டியல்களைப் பயன்படுத்தாதீர்கள். இயல்பான, அன்பான உரைநடையில் எழுதவும்."
            )
        return (
            "- தமிழ் மொழியில் பதில் அளிக்கவும்.\n"
            "- புதியவர்கள் புரிந்துகொள்ளக்கூடிய எளிமையான, மனித விளக்கத்துடன் தொடங்கவும்.\n"
            "- மறைநூல் கூறுவதை இயற்கையாக ஆதாரமாக மேற்கோள் காட்டவும் "
            "(எ.கா. 'யோவான் நற்செய்தியில் எழுதியுள்ளது போல்...').\n"
            "- போதனையில் நடைமுறை அல்லது ஆன்மிக பரிமாணம் இருந்தால், சுருக்கமாக குறிப்பிடவும்.\n"
            "- புள்ளிகள் அல்லது பட்டியல்களைப் பயன்படுத்தாதீர்கள். இயல்பான, அன்பான உரைநடையில் எழுதவும்."
        )

    # English
    if religion == "Buddhism":
        return (
            "- Start with a simple, human explanation a beginner can understand.\n"
            "- Then support it with what the scripture says, citing the source book "
            "naturally (e.g. \"As found in the Samyutta Nikaya...\").\n"
            "- If the teaching has a practical dimension, briefly mention it.\n"
            "- Do not use bullet points or lists. Write in flowing, warm prose."
        )
    return (
        "- Do NOT restate or paraphrase the question. Start directly with the answer.\n"
        "- Start with a simple, human explanation a newcomer can understand.\n"
        "- Then support it with what the scripture says, citing the source book "
        "naturally ONLY if that book appears in the [Source: ...] tags above.\n"
        "- If the teaching has a practical or spiritual dimension, briefly mention it.\n"
        "- Do not use bullet points or lists. Write in flowing, warm prose."
    )


# Sentence trim
# Matches sentence-ending punctuation for both English (.!?) and Sinhala (។ and similar)
_SENT_END   = re.compile(r'[.!?។]\s*$')
_SENT_SPLIT = re.compile(r'[.!?។][\'"\)\]]*(?=\s|$)')

def _detect_and_trim_repetition(text: str) -> str:
    """
    Detect and remove LLM generation loops — where a short phrase repeats
    many times in a row (e.g. 'අනුග්‍රහය පිළිබඳව' × 50).
    Keeps only the content up to the first repetition cycle.
    """
    # Split into sentences or clause-like chunks to find the repeat unit
    # Check for any phrase of 3–60 chars that repeats 5+ consecutive times
    pattern = re.compile(r'(.{3,60}?)(\1){4,}', re.DOTALL | re.UNICODE)
    m = pattern.search(text)
    if m:
        # Trim everything from the start of the loop onward
        cut = m.start()
        trimmed = text[:cut].strip().rstrip(',;— ')
        if len(trimmed) >= 30:
            print(f"  [loop-detect] Repetition loop trimmed at char {cut} "
                  f"(phrase: {m.group(1)[:40]!r})")
            return trimmed
    return text


def _trim_incomplete_sentence(text: str) -> str:
    """
    Trim a response that ends mid-sentence due to token truncation.
    Also detects and removes LLM generation loops before trimming.
    If the text already ends with proper punctuation, return it unchanged.
    Handles both English (.!?) and Sinhala (។) sentence endings.
    Also handles the common case where Sinhala text ends without punctuation
    but the last word is a complete thought — in that case return as-is
    rather than aggressively truncating.
    """
    text = text.strip()
    if not text:
        return text

    # Remove repetition loops first
    text = _detect_and_trim_repetition(text)

    # Already ends cleanly
    if _SENT_END.search(text):
        return text

    # Find the last clean sentence boundary
    last = None
    for m in _SENT_SPLIT.finditer(text):
        last = m

    if last:
        trimmed = text[:last.end()].strip()
        # Only trim if we're not losing more than 20% of the response
        # (avoids aggressive cuts on Sinhala text that just lacks final punctuation)
        if len(trimmed) >= len(text) * 0.80:
            return trimmed

    # No good boundary found or trim would lose too much — return as-is
    return text


# Respectful titles — Buddhism
_BUDDHA_REPLACEMENTS = [
    ("\u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0dcf\u0da2\u0dcf",
     "\u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0da2\u0dcf\u0dab\u0db1\u0dca \u0dc0\u0dc4\u0db1\u0dca\u0dc3\u0dda"),
    ("\u0d9c\u0dbd\u0dca\u0db0\u0db8 \u0db6\u0dd4\u0daf\u0dd4",
     "\u0d9c\u0dbd\u0dca\u0db0\u0db8 \u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0da2\u0dcf\u0dab\u0db1\u0dca \u0dc0\u0dc4\u0db1\u0dca\u0dc3\u0dda"),
    ("\u0d9c\u0dbd\u0dca\u0db0\u0db8",
     "\u0d9c\u0dbd\u0dca\u0db0\u0db8 \u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0da2\u0dcf\u0dab\u0db1\u0dca \u0dc0\u0dc4\u0db1\u0dca\u0dc3\u0dda"),
]

# Respectful titles — Christianity
# Google Translate and most LLMs render Jesus as යේසුස් (incorrect Sinhala convention)
# The correct Sinhala Catholic/Protestant spelling is ජේසුස් වහන්සේ.
_CHRISTIAN_REPLACEMENTS = [
    ("යේසුස් ක්‍රිස්තුස් වහන්සේ", "ජේසුස් ක්‍රිස්තුස් වහන්සේ"),
    ("යේසුස් ක්‍රිස්තුස්",        "ජේසුස් ක්‍රිස්තුස්"),
    ("යේසුස් වහන්සේ",             "ජේසුස් වහන්සේ"),
    ("යේසුස්",                     "ජේසුස්"),
    # Common misspellings of බයිබලය
    ("බිබලය",                      "බයිබලය"),
    ("බිබලයේ",                     "බයිබලයේ"),
    ("බිබලයට",                     "බයිබලයට"),
    ("බිබලයෙන්",                   "බයිබලයෙන්"),
    ("බිබල",                       "බයිබල"),
]

def _apply_respectful_titles(text: str, religion: str = "Buddhism") -> str:
    if religion == "Christianity":
        for wrong, correct in _CHRISTIAN_REPLACEMENTS:
            text = text.replace(wrong, correct)
    else:
        for a, b in _BUDDHA_REPLACEMENTS:
            text = text.replace(a, b)
    return text


# Strip the LLM echoing the user's question back at the start of its answer.
# Covers:
#   SI:  ප්‍රශ්නය "ශුද්ධාත්මය යනු කුමක්ද?" ...
#        "ශුද්ධාත්මය යනු කුමක්ද?" යන ප්‍රශ්නය ...
#   TA:  கேள்வி "யார் இயேசு?" ...
#   EN:  The question "Who is the Holy Spirit?" ...
#   Mixed: The question "ශුද්ධ ආත්මය කවුද?" සිංහලෙන් ...
_QUESTION_ECHO_RE = re.compile(
    r'^(?:'
    # Sinhala: ප්‍රශ්නය "..."
    r'ප්‍රශ්නය\s*[""„\u201c\u201d].*?[""\u201d\u201c][^\n.]*?[.\n]?\s*'
    # Sinhala: "..." යන ප්‍රශ්නය
    r'|[""„\u201c\u201d].*?[""\u201d\u201c]\s*යන\s*ප්‍රශ්නය[^\n.]*?[.\n]?\s*'
    # Tamil: கேள்வி "..."
    r'|கேள்வி\s*[""„\u201c\u201d].*?[""\u201d\u201c][^\n.]*?[.\n]?\s*'
    # English / mixed: The question "..." or Question: "..."
    r'|[Tt]he\s+question\s*[""„\u201c\u201d].*?[""\u201d\u201c][^\n.]*?[.\n]?\s*'
    r'|[Qq]uestion\s*:\s*[""„\u201c\u201d].*?[""\u201d\u201c][^\n.]*?[.\n]?\s*'
    r')',
    re.UNICODE | re.DOTALL,
)

# Matches fabricated in-text book citation phrases the LLM writes in Sinhala/Tamil
# after translation, e.g. "1 කොරින්ති පොතේ ලියා ඇති පරිදි," or
# "හිතෝපදේශ ග්‍රන්ථයේ සඳහන් වන පරිදි"
_SI_BOOK_CITE_RE = re.compile(
    r'[\w\s\u0D80-\u0DFF\u0B80-\u0BFF\d]+?\s*(?:'
    r'පොතේ\s*ලියා\s*ඇති\s*පරිදි|'
    r'ග්‍රන්ථයේ\s*(?:ලියා\s*ඇති|සඳහන්\s*වන)\s*පරිදි|'
    r'හි\s*ලියා\s*ඇති\s*பரிதி|'
    r'நூலில்\s*கூறப்பட்டுள்ளது\s*போல்|'
    r'புத்தகத்தில்\s*எழுதப்பட்டுள்ளபடி'
    r')[,\s]*',
    re.UNICODE,
)

def _scrub_fabricated_book_cites(text: str, context: str) -> str:
    """
    Remove inline book citation phrases (e.g. '1 කොරින්ති පොතේ ලියා ඇති පරිදි,')
    where the cited book does NOT appear in the retrieved context.
    """
    def _keep(m: re.Match) -> str:
        phrase = m.group(0)
        # If the phrase's book name appears somewhere in the context, keep it
        if any(word in context for word in phrase.split() if len(word) > 3):
            return phrase
        return ""
    return _SI_BOOK_CITE_RE.sub(_keep, text).strip()


def _scrub_question_echo(text: str) -> str:
    """Remove any opening where the LLM restates the user's question."""
    cleaned = _QUESTION_ECHO_RE.sub("", text).strip()
    return cleaned if cleaned else text


# No-context detection & scrubbing
_NO_CONTEXT_SI = "මෙය නිවැරදිව පිළිතුරු දීමට ප්‍රාණවත් විශ්වාසදායක ශාස්ත්‍රීය සන්දර්භයක් මා සතු නොවේ"

def _answer_is_no_context(text):
    return _NO_CONTEXT_SI in text.strip()

# Phrases that indicate the LLM is hedging / explaining lack of context
# rather than actually answering the question.
_WEAK_CONTEXT_FRAGMENTS = [
    "සන්දර්භයේ",               # "in the given context"
    "ශාස්ත්‍රීය සන්දර්භ",      # "scriptural context"
    "ප්‍රාණවත් විශ්වාසදායක",   # exact no-context phrase
    "ප්‍රමාණවත් විස්තර",       # "sufficient detail"
    "ප්‍රමාණවත් නොවේ",         # "not sufficient"
    "නොසඳහන් වේ",               # "is not mentioned"
    "සඳහන් නොවේ",               # "not mentioned"
    "වෙනත් ශාස්ත්‍රීය",        # "other scriptural texts"
    "සොයා බැලිය යුතු",          # "should look up"
    "යොමු කළ යුතු",             # "should refer to"
    "සම්බන්ධ නොවන",             # "not connected to" — seen in Jesus answer
    "මෙම පාඨ සඳහා සම්බන්ධ",    # "connected to these passages"
    "ශාස්ත්‍රීය සැබෑව",         # "scriptural truth" (hedging phrase)
    "ප්‍රකාශ කළ නොහැක",         # "cannot be expressed"
    "වෙනත් පාඨ",                # "other passages" — suggesting look elsewhere
    "සම්බන්ධ විය යුතු",         # "should be connected to"
]

def _scrub_no_context_sentence(text):
    """Remove disclaimer/hedge sentences from the answer."""
    text = text.replace(_NO_CONTEXT_SI + ".", "")
    text = text.replace(_NO_CONTEXT_SI, "")
    lines = text.split(".")
    clean = [l for l in lines
             if not any(frag in l for frag in _WEAK_CONTEXT_FRAGMENTS)]
    text = ".".join(clean)
    return text.strip(". \n")


def _answer_is_weak(text: str) -> bool:
    """Return True if the answer is mostly hedging with no real content."""
    if not text or len(text.strip()) < 40:
        return True
    # Count how many sentences contain weak-context phrases
    sentences = [s for s in text.split(".") if s.strip()]
    if not sentences:
        return True
    weak = sum(1 for s in sentences
               if any(f in s for f in _WEAK_CONTEXT_FRAGMENTS))
    # If more than half the sentences are hedging, consider it weak
    return weak / len(sentences) >= 0.5


# Signals that the answer is about the wrong topic entirely —
# detected when flood/Noah narrative dominates a salvation question, etc.
_TOPIC_MISMATCH_SIGNALS = {
    # question keyword (EN) → words that should NOT dominate the answer
    "salvation": ["noah", "flood", "නෝහා", "ජලගැල්ම", "ark", "නෞකා",
                  "வெள்ளம்", "நோவா"],
    "trinity":   ["flood", "ජලගැල්ම", "noah", "නෝහා"],
    "prayer":    ["flood", "ජලගැල්ම", "noah", "නෝහා"],
}

def _is_topic_mismatch(en_query: str, answer: str) -> bool:
    """
    Return True if the answer appears to be about a completely different topic
    than what was asked — e.g. flood narrative returned for a salvation question.
    """
    q = en_query.lower()
    a = answer.lower()
    for keyword, bad_signals in _TOPIC_MISMATCH_SIGNALS.items():
        if keyword in q:
            hits = sum(1 for s in bad_signals if s in a)
            if hits >= 2:
                print(f"  [topic-mismatch] query contains '{keyword}' but answer "
                      f"contains {hits} mismatch signals — treating as bad retrieval")
                return True
    return False


# ════════════════════════════════════════════════════════════════
# Sinhala answer helpers
# ════════════════════════════════════════════════════════════════

def _build_english_answer(en_q: str, en_res: list[dict], religion: str) -> str:
    """Generate an English answer from English scripture chunks."""
    ctx = "\n\n---\n\n".join(
        f"[Source: {r['book']} | {r.get('pitaka', '')}]\n{r['text']}"
        for r in en_res
    )
    fmt = _format_instructions(religion, _is_list_request(en_q), "en")
    ref = {
        "Buddhism":    "Do NOT cite specific verse numbers (like SN 56.11)",
        "Christianity": "Do NOT cite specific verse numbers (like John 3:16)",
    }.get(religion, "Do NOT cite specific verse numbers")
    msg = (
        f"Scripture context:\n{ctx}\n\n"
        f"Question: {en_q}\n\n"
        f"Instructions:\n{fmt}\n- {ref}\n\n"
        f"Answer:"
    )
    raw = _call_groq(_PERSONAS_EN[religion], msg, MODEL_DEFAULT)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*$",         "", raw, flags=re.DOTALL).strip()
    return _trim_incomplete_sentence(raw)


def _review_translation(
    original_english: str,
    translated_text: str,
    religion: str,
    target_lang: str,        # "si" or "ta"
) -> str:
    """
    Send the translated answer to Qwen3 for a language review.
    Fixes unnatural phrasing, wrong terminology, and cross-religion term leakage.
    Returns the corrected text, or the original translation if the call fails.
    """
    if target_lang == "ta":
        religion_note = {
            "Buddhism": (
                "இது ஒரு பௌத்த பதில். சரியான பௌத்த தமிழ் சொற்களை பயன்படுத்தவும் "
                "(எ.கா. நிர்வாணம், துக்கம், தண்ணா, அனித்யம், அனாத்மா, திரிபிடகம்).\n"
                "- 'புனித வேதம்' அல்லது கிறிஸ்தவ/இஸ்லாமிய சொற்களை பௌத்த பதிலில் "
                "சேர்க்க வேண்டாம்.\n"
                "- Google Translate இன் தவறான மொழிபெயர்ப்புகளை சரிசெய்யவும்."
            ),
            "Christianity": (
                "இது ஒரு கிறிஸ்தவ பதில். சரியான கிறிஸ்தவ தமிழ் சொற்களை பயன்படுத்தவும்.\n\n"
                "முக்கியமான மொழிபெயர்ப்பு திருத்தங்கள் (Google Translate பிழைகள்):\n"
                "  இரட்சிப்பு    → salvation — NOT 'rescue' அல்லது 'escape'\n"
                "  திரித்துவம்   → Trinity — NOT 'three groups' அல்லது 'trio'\n"
                "  கிருபை        → grace — divine unmerited favour — NOT just 'kindness'\n"
                "  மீட்பு        → redemption — NOT 'recovery'\n"
                "  நித்திய ஜீவன் → eternal life — NOT 'immortality' அல்லது 'endless life'\n"
                "  விசுவாசம்     → faith — NOT just 'trust' அல்லது 'belief'\n"
                "  பாவம்         → sin — NOT 'mistake'\n"
                "  மன்னிப்பு     → forgiveness — NOT 'pardon' in a legal sense\n"
                "  ஞானஸ்நானம்   → baptism — NOT 'bath' அல்லது 'washing'\n"
                "  திருவிருந்து  → Holy Communion / Eucharist — NOT 'feast'\n"
                "  ஜெபம்         → prayer — NOT 'request'\n"
                "  சுவிசேஷம்    → Gospel — NOT 'good news story'\n"
                "  திருச்சபை    → Church — NOT just 'assembly'\n"
                "  பரிசுத்த ஆவி → Holy Spirit — NOT 'holy ghost' in modern usage\n"
                "  சொர்க்கம்    → heaven — NOT 'sky'\n"
                "  நரகம்         → hell\n"
                "  மனந்திரும்புதல் → repentance — NOT 'changing one's mind'\n\n"
                "- பௌத்த சொற்களை கிறிஸ்தவ பதிலில் சேர்க்க வேண்டாம்.\n"
                "- Google Translate இன் தவறான மொழிபெயர்ப்புகளை சரிசெய்யவும்."
            ),
        }.get(religion, "சரியான மற்றும் இயற்கையான தமிழ் மத சொற்களை பயன்படுத்தவும்.")

        system_prompt = (
            "நீங்கள் தமிழ் மொழி நிபுணர் மற்றும் மத மொழிபெயர்ப்பு மதிப்பாய்வாளர்.\n\n"
            "உங்கள் பணி:\n"
            "1. கொடுக்கப்பட்ட தமிழ் மொழிபெயர்ப்பை மதிப்பாய்வு செய்யுங்கள்.\n"
            "2. இயற்கையற்ற, தவறான, அல்லது Google Translate பிழைகளை சரிசெய்யுங்கள்.\n"
            f"3. {religion_note}\n"
            "4. மூல ஆங்கில பதிலின் அர்த்தத்தை சரியாக பாதுகாக்கவும்.\n"
            "5. சரிசெய்யப்பட்ட தமிழ் உரையை மட்டும் வழங்குங்கள். "
            "எந்த விளக்கமும், ஆங்கிலமும், குறிப்புகளும் இருக்கக்கூடாது."
        )
    else:
        # Sinhala
        religion_note = {
            "Buddhism": (
                "මෙය බෞද්ධ පිළිතුරකි. නිවැරදි බෞද්ධ සිංහල පාරිභාෂිතය භාවිත කරන්න "
                "(නිදසුන්: නිර්වාණය, දුක්ඛය, උපාදානය, තණ්හාව, අනිත්‍ය, අනාත්ම, "
                "ත්‍රිපිටකය, ධර්ම ග්‍රන්ථය, සූත්‍ර පිටකය, විනය පිටකය, අභිධර්ම පිටකය).\n"
                "- 'ශුද්ධ ලියවිල්ල' යනු ක්‍රිස්තියානි පදයකි — බෞද්ධ පිළිතුරක් තුළ කිසිවිටෙකත් "
                "භාවිත නොකරන්න. ඒ වෙනුවට 'ත්‍රිපිටකය', 'ධර්ම ග්‍රන්ථය', හෝ නිශ්චිත "
                "ග්‍රන්ථ නාමය (නිදසුන්: 'සංයුත්ත නිකායේ') භාවිත කරන්න.\n"
                "- 'ඉන්ධන' (petrol/fuel) වෙනුවට 'හේතුව' හෝ 'උපාදානය' භාවිත කරන්න.\n"
                "- 'දෙවියන්' හෝ ක්‍රිස්තියානි/ඉස්ලාම් සංකල්ප බෞද්ධ පිළිතුරක "
                "ඇතුළත් නොකරන්න."
            ),
            "Christianity": (
                "මෙය ක්‍රිස්තියානි පිළිතුරකි. නිවැරදි ක්‍රිස්තියානි සිංහල පාරිභාෂිතය භාවිත කරන්න.\n\n"
                "වැදගත් ශබ්ද නිවැරදිකිරීම් (Google Translate සාමාන්‍ය වැරදි):\n"
                "  ගැලවීම          → salvation — 'escape', 'rescue', 'relief' නොවේ\n"
                "  ත්‍රිත්වය       → Trinity — 'three groups', 'trio', 'triple' නොවේ\n"
                "  කරුණාව          → grace (දෛවීය) — 'mercy' පමණ නොවේ\n"
                "  මිදීම           → redemption — 'freedom' හෝ 'release' පමණ නොවේ\n"
                "  සදාකාලික ජීවිතය → eternal life — 'everlasting living' නොවේ\n"
                "  ඇදහිල්ල        → faith — 'trust' හෝ 'belief' පමණ නොවේ\n"
                "  පාපය            → sin — 'mistake' හෝ 'wrongdoing' නොවේ\n"
                "  සමාව            → forgiveness — 'excuse' නොවේ\n"
                "  බාප්තිස්මය      → baptism — 'bathing' හෝ 'washing' නොවේ\n"
                "  ශුද්ධ කොමියුනියන් → Holy Communion / Eucharist — 'holy meal' නොවේ\n"
                "  යාච්ඤාව         → prayer — 'request' හෝ 'begging' නොවේ\n"
                "  සුවිශේෂය         → Gospel — 'good news' පමණ නොවේ\n"
                "  සභාව             → Church — 'assembly' හෝ 'meeting' පමණ නොවේ\n"
                "  පරිශුද්ධ ආත්මය  → Holy Spirit — 'holy soul' හෝ 'pure soul' නොවේ\n"
                "  ස්වර්ගය          → heaven — 'sky' නොවේ\n"
                "  නිරය             → hell\n"
                "  පශ්චාත්තාපය      → repentance — 'regret' පමණ නොවේ\n"
                "  ශුද්ධ ලියවිල්ල  → Bible / scripture\n"
                "  බයිබලය           → Bible — 'බිබලය' හෝ 'බිබල' නොවේ (spelling නිවැරදිය: බ-යි-බ-ල-ය)\n"
                "  තෑග්ග / ප්‍රදානය → gift (දෙවියන් වහන්සේගෙන් ලැබෙන) — 'ආගෝපාදා' හෝ ව්‍යාජ වචන නොවේ\n"
                "  දිව් බස / භාෂාවන් → tongues (spiritual gift) — 'languages' පමණ නොවේ\n"
                "  අභිරහස           → mystery — 'secret' හෝ 'puzzle' නොවේ\n"
                "  සාක්ෂිය          → testimony / witness\n"
                "  දිව්‍ය ආශිර්වාදය → blessing — 'good luck' නොවේ\n"
                "  ප්‍රකාශය          → revelation / prophecy — context අනුව\n\n"
                "විශේෂ අවවාදය: ඉහත ලැයිස්තුවේ නොමැති ඕනෑම වචනයක් "
                "සිංහලෙන් නොව ව්‍යාජ ශබ්දයක් ලෙස පෙනේ නම් (උදා. 'ආගෝපාදා', "
                "'ප්‍රේෂිතා', 'ශාන්තිකා' ආදී), එය Google Translate දෝෂයකි — "
                "ඉංග්‍රීසි මුලාශ්‍රය දෙස බලා නිවැරදි සිංහල ශබ්දයෙන් ප්‍රතිස්ථාපනය කරන්න.\n"
                "- 'ත්‍රිපිටකය', 'නිර්වාණය' හෝ බෞද්ධ සංකල්ප ක්‍රිස්තියානි පිළිතුරක "
                "ඇතුළත් නොකරන්න."
            ),
        }.get(religion, "නිවැරදි සහ ස්වාභාවික සිංහල ආගමික පාරිභාෂිතය භාවිත කරන්න.")

        system_prompt = (
            "ඔබ සිංහල භාෂා විශේෂඥයෙකු සහ ආගමික පරිවර්තන සමාලෝචකයෙකි.\n\n"
            "ඔබේ කාර්යය:\n"
            "1. ලබා දී ඇති සිංහල පරිවර්තනය සමාලෝචනය කරන්න.\n"
            "2. අස්වාභාවික, වැරදි, හෝ Google Translate ගැටළු නිවැරදි කරන්න.\n"
            "3. ව්‍යාජ හෝ නොපවතින සිංහල වචන (Google Translate දෝෂ) හඳුනා ගෙන "
            "නිවැරදි සිංහල ශබ්දයෙන් ප්‍රතිස්ථාපනය කරන්න. "
            "උදාහරණ: 'ආගෝපාදා' → 'තෑග්ග', 'ප්‍රේෂිතා' → නිවැරදි ශබ්දය, "
            "ඕනෑම නොදන්නා හෝ ව්‍යාජ ශබ්දයක් ඉංග්‍රීසි මූලාශ්‍රය දෙස බලා නිවැරදි කරන්න.\n"
            f"4. {religion_note}\n"
            "5. වෙනත් ආගම්වලට අයත් පද හෝ සංකල්ප (cross-religion terminology) "
            "ඉවත් කර නිවැරදි ආගමික පාරිභාෂිතයෙන් ප්‍රතිස්ථාපනය කරන්න.\n"
            "6. මුල් ඉංග්‍රීසි පිළිතුරේ අර්ථය හරියටම ආරක්ෂා කරන්න.\n"
            "7. නිවැරදි කළ සිංහල පාඨය පමණක් ලබා දෙන්න. "
            "කිසිදු පැහැදිලි කිරීමක්, ඉංග්‍රීසි, හෝ ටිප්පණි නොතිබිය යුතුය."
        )

    user_message = (
        f"{'மூல ஆங்கில பதில்' if target_lang == 'ta' else 'මුල් ඉංග්‍රීසි පිළිතුර'}:\n"
        f"{original_english}\n\n"
        f"{'தமிழ் மொழிபெயர்ப்பு (மதிப்பாய்வு செய்யவும்)' if target_lang == 'ta' else 'සිංහල පරිවර්තනය (සමාලෝචනය කරන්න)'}:\n"
        f"{translated_text}"
    )

    reviewed = _call_groq(system_prompt, user_message, MODEL_TRANSLATE)
    reviewed = re.sub(r"<think>.*?</think>", "", reviewed, flags=re.DOTALL)
    reviewed = re.sub(r"<think>.*$",         "", reviewed, flags=re.DOTALL).strip()

    if not reviewed or len(reviewed) < 20:
        print(f"  [review] Translation review returned empty — keeping original translation")
        return translated_text

    print(f"  [review] Qwen3 review applied ({target_lang})")
    return reviewed


def _english_context_then_translate(
    question: str,
    en_query: str,
    religion: str,
    target_lang: str = "si",   # "si" or "ta"
) -> dict:
    """
    Fallback path for Sinhala/Tamil questions with no native scripture data:
      1. Retrieve English scripture chunks using the translated query.
      2. Generate an English answer.
      3. Translate the answer to the target language.
      4. Qwen3 review to fix terminology and phrasing.
    """
    try:
        from translator import translate_from_english
    except ImportError:
        return {
            "answer": _FALLBACK_MESSAGES.get(target_lang, _FALLBACK_MESSAGES["en"])["no_context"],
            "sources": [], "scores": [], "flagged": False,
            "low_confidence": True, "warnings": ["translator_unavailable"],
        }

    en_res = search(en_query, religion=religion, language="en")
    if not en_res:
        # No English scripture found — only show no_context if the question
        # is substantive. For greetings this path is already intercepted above.
        return {
            "answer": _FALLBACK_MESSAGES.get(target_lang, _FALLBACK_MESSAGES["en"])["no_context"],
            "sources": [], "scores": [], "flagged": False,
            "low_confidence": True, "warnings": ["no_en_context"],
        }

    en_res = _refine_results(en_res, religion)
    en_ans = _build_english_answer(en_query, en_res, religion)

    # Only bail out if the LLM itself says it has no context (hallucination guard).
    # A translated answer for si/ta is still valid even if it went through English retrieval.
    if not en_ans or _answer_is_no_context(en_ans) or _answer_is_weak(en_ans) or _is_topic_mismatch(en_query, en_ans):
        return {
            "answer": _FALLBACK_MESSAGES.get(target_lang, _FALLBACK_MESSAGES["en"])["no_context"],
            "sources": [], "scores": [], "flagged": False,
            "low_confidence": True, "warnings": ["no_en_answer"],
        }

    # Translate to target language then Qwen3 review
    translated = translate_from_english(en_ans, target_lang)
    translated  = _trim_incomplete_sentence(translated)
    translated  = _review_translation(en_ans, translated, religion, target_lang)
    translated  = _trim_incomplete_sentence(translated)
    translated  = _scrub_question_echo(translated)

    # Build English context string for fabricated-reference checking
    en_context = "\n\n".join(
        f"[Source: {r['book']} | {r.get('pitaka', r.get('testament', ''))}]\n{r['text']}"
        for r in en_res
    )
    translated  = _scrub_fabricated_book_cites(translated, en_context)
    if target_lang in ("si", "ta"):
        translated = _apply_respectful_titles(translated, religion)
    translated, extra_warnings = moderate_output(translated, en_context, religion, target_lang)

    warning_key = f"used_en_context_with_translation_{target_lang}"
    matched = [r for r in en_res if r["book"].lower() in translated.lower()]
    disp    = matched if matched else en_res
    src, scores_out = _unique_sources(disp)
    return {
        "answer":         translated,
        "sources":        src,
        "scores":         scores_out,
        "flagged":        False,
        "low_confidence": False,
        "warnings":       [warning_key] + extra_warnings,
    }


# ════════════════════════════════════════════════════════════════
# Christianity native SI/TA path — Qwen3 answers directly from
# native-language chunks in chunks-en-si-ta.db
# ════════════════════════════════════════════════════════════════

def _answer_with_native_christianity_chunks(
    question:    str,
    en_query:    str,
    native_res:  list[dict],
    religion:    str,
    target_lang: str,          # "si" or "ta"
) -> dict:
    """
    Build a prompt from native SI/TA Christianity scripture chunks and call
    Qwen3 directly (no translation step needed — the context is already in
    the user's language).

    Returns the same dict shape as answer_question().
    Raises nothing — on any Qwen3 failure the caller falls back to the
    English-context + translate path.
    """
    native_res = _refine_results(native_res, religion)

    context = "\n\n---\n\n".join(
        f"[Source: {r['book']} | {r.get('pitaka', '')}]\n{r['text']}"
        for r in native_res
    )

    persona      = _PERSONAS_SI[religion] if target_lang == "si" else _PERSONAS_EN.get(religion, "")
    format_instr = _format_instructions(religion, _is_list_request(question), target_lang)
    ref_note_map = {
        "si": "John 3:16, Romans 8:28 වැනි නිශ්චිත වාක්‍ය අංක උද්ධෘත නොකරන්න",
        "ta": "John 3:16, Romans 8:28 போன்ற குறிப்பிட்ட வசன எண்களை மேற்கோள் காட்டாதீர்கள்",
    }
    ref_note = ref_note_map.get(target_lang, "Do NOT cite specific verse numbers")

    if target_lang == "si":
        user_message = (
            f"ශාස්ත්‍රීය සන්දර්භය:\n{context}\n\n"
            f"ප්‍රශ්නය: {question}\n\n"
            f"උපදෙස්:\n{format_instr}\n"
            f"- ලබා දී ඇති සන්දර්භයේ හරියටම ඒ යොමු දිස් නොවේ නම් {ref_note}.\n\n"
            f"පිළිතුර:"
        )
    else:  # ta
        user_message = (
            f"மறைநூல் சூழல்:\n{context}\n\n"
            f"கேள்வி: {question}\n\n"
            f"வழிமுறைகள்:\n{format_instr}\n"
            f"- {ref_note}.\n\n"
            f"பதில்:"
        )

    raw = _call_groq(persona, user_message, MODEL_SINHALA)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*$",         "", raw, flags=re.DOTALL).strip()
    raw = _trim_incomplete_sentence(raw)
    raw = _scrub_no_context_sentence(raw)
    raw = _scrub_question_echo(raw)

    if not raw or _answer_is_no_context(raw) or _answer_is_weak(raw) or _is_topic_mismatch(en_query, raw):
        # Signal to caller that this path failed
        return {}

    raw = _apply_respectful_titles(raw, religion)
    raw = _scrub_fabricated_book_cites(raw, context)
    final_answer, warnings = moderate_output(raw, context, religion, target_lang)
    final_answer = _scrub_question_echo(final_answer)
    final_answer = _apply_respectful_titles(final_answer, religion)

    matched         = [r for r in native_res if r["book"].lower() in final_answer.lower()]
    display_results = matched if matched else native_res
    src, scores_out = _unique_sources(display_results)

    return {
        "answer":         final_answer,
        "sources":        src,
        "scores":         scores_out,
        "flagged":        False,
        "low_confidence": False,
        "warnings":       [f"used_native_{target_lang}_chunks"] + warnings,
    }


# ════════════════════════════════════════════════════════════════
# Core answer function
# ════════════════════════════════════════════════════════════════

def answer_question(
    question: str,
    religion: str = "Buddhism",
    language: str = "en",
) -> dict:
    """
    Main entry point.

    Sinhala flow (lang == "si"):
      Step 1 — Translate the question to English (for FAISS embedding).
      Step 2 — Check whether RELEVANT Sinhala scripture chunks exist in the DB
               by running FAISS with the English query and filtering for si chunks.
      Step 3a — Sinhala chunks found:
                  • Build prompt with Sinhala context + Sinhala persona.
                  • Call LLM (MODEL_SINHALA).
                  • Translate the answer to Sinhala via deep_translator.
      Step 3b — No Sinhala chunks found:
                  • Retrieve English scripture chunks.
                  • Generate an English answer with the English persona.
                  • Translate the English answer to Sinhala.

    English flow: unchanged.
    """

    # ── Resolve effective language ───────────────────────────────
    lang  = _detect_language(question, language)
    model = MODEL_SINHALA if lang in ("si", "ta") else MODEL_DEFAULT

    # ── Greeting shortcut — no RAG needed ───────────────────────
    if _is_greeting(question):
        greet_lang = lang if lang in ("si", "ta") else "en"
        religion_greetings = _GREETING_RESPONSES.get(religion, _GREETING_RESPONSES["Buddhism"])
        greeting_text = religion_greetings.get(greet_lang, religion_greetings["en"])
        return {
            "answer":         greeting_text,
            "sources":        [],
            "scores":         [],
            "flagged":        False,
            "low_confidence": False,
            "warnings":       [],
        }

    # ── Layer 1: Input moderation ────────────────────────────────
    is_safe, reason = moderate_input(question, religion=religion)
    if not is_safe:
        fb = _FALLBACK_MESSAGES.get(lang, _FALLBACK_MESSAGES["en"])
        return {
            "answer":         fb.get(reason, "This question cannot be answered."),
            "sources":        [],
            "scores":         [],
            "flagged":        True,
            "low_confidence": False,
            "warnings":       [reason],
        }

    # ════════════════════════════════════════════════════════════
    # TAMIL PATH
    # ════════════════════════════════════════════════════════════
    if lang == "ta":
        en_query = _translate_query_to_english(question, religion=religion)
        print(f"  [ta] EN query: {en_query!r}")

        # For Christianity: first try native Tamil chunks from chunks-en-si-ta.db
        if religion == "Christianity":
            from retrieve import search_christianity_native_lang
            print("  [ta] Christianity — checking chunks-en-si-ta.db for Tamil chunks")
            try:
                native_res = search_christianity_native_lang(en_query, language="ta")
            except Exception as exc:
                print(f"  [ta] Native search error: {exc} — falling back to translate path")
                native_res = []

            if len(native_res) >= MIN_CHUNKS_FOR_NATIVE:
                print(f"  [ta] {len(native_res)} Tamil chunks found — calling Qwen3 directly")
                try:
                    result = _answer_with_native_christianity_chunks(
                        question, en_query, native_res, religion, target_lang="ta"
                    )
                except Exception as exc:
                    print(f"  [ta] Qwen3 native path error: {exc} — falling back to translate path")
                    result = {}

                if result:
                    return result
                print("  [ta] Qwen3 returned weak/empty answer — falling back to translate path")
            else:
                print(
                    f"  [ta] Only {len(native_res)} Tamil chunk(s) found "
                    f"(need {MIN_CHUNKS_FOR_NATIVE}) — using English context + translate"
                )

        return _english_context_then_translate(question, en_query, religion, target_lang="ta")

    # ════════════════════════════════════════════════════════════
    # SINHALA PATH
    # ════════════════════════════════════════════════════════════
    if lang == "si":
        # Step 1: Translate question to English for FAISS embedding
        en_query = _translate_query_to_english(question, religion=religion)
        print(f"  [si] Original question: {question[:80]!r}")
        print(f"  [si] Translated EN query: {en_query!r}")

        # Christianity: first try native Sinhala chunks from chunks-en-si-ta.db,
        # then fall back to the English-context + translate path.
        if religion == "Christianity":
            from retrieve import search_christianity_native_lang
            print("  [si] Christianity — checking chunks-en-si-ta.db for Sinhala chunks")
            try:
                native_res = search_christianity_native_lang(en_query, language="si")
            except Exception as exc:
                print(f"  [si] Native search error: {exc} — falling back to translate path")
                native_res = []

            if len(native_res) >= MIN_CHUNKS_FOR_NATIVE:
                print(f"  [si] {len(native_res)} Sinhala chunks found — calling Qwen3 directly")
                try:
                    result = _answer_with_native_christianity_chunks(
                        question, en_query, native_res, religion, target_lang="si"
                    )
                except Exception as exc:
                    print(f"  [si] Qwen3 native path error: {exc} — falling back to translate path")
                    result = {}

                if result:
                    return result
                print("  [si] Qwen3 returned weak/empty answer — falling back to translate path")
            else:
                print(
                    f"  [si] Only {len(native_res)} Sinhala chunk(s) found "
                    f"(need {MIN_CHUNKS_FOR_NATIVE}) — using English context + translate"
                )

            print("  [si] Christianity — using English context + translation fallback")
            return _english_context_then_translate(question, en_query, religion, target_lang="si")

        from retrieve import search_sinhala_direct
        from translator import translate_from_english

        # Step 2: Check for relevant Sinhala chunks in DB (Buddhism)
        si_results = search_sinhala_direct(en_query, religion=religion)
        print(f"  [si] Sinhala chunks found: {len(si_results)}"
              + (f" — top score: {si_results[0]['score']:.3f}" if si_results else " — falling back to English context"))

        if si_results:
            # ── Step 3a: Use Sinhala scripture ──────────────────
            si_results = _refine_results(si_results, religion)
            context    = "\n\n---\n\n".join(
                f"[Source: {r['book']} | {r.get('pitaka', '')}]\n{r['text']}"
                for r in si_results
            )
            format_instr = _format_instructions(religion, _is_list_request(question), "si")
            persona      = _PERSONAS_SI[religion]
            ref_note_map = {
                "Buddhism":     "SN 56.11, MN 22 වැනි නිශ්චිත වාක්‍ය අංක උද්ධෘත නොකරන්න",
                "Christianity": "John 3:16, Romans 8:28 වැනි නිශ්චිත වාක්‍ය අංක උද්ධෘත නොකරන්න",
            }
            ref_note = ref_note_map.get(religion, "නිශ්චිත වාක්‍ය අංක උද්ධෘත නොකරන්න")
            user_message = (
                f"ශාස්ත්‍රීය සන්දර්භය:\n{context}\n\n"
                f"ප්‍රශ්නය: {question}\n\n"
                f"උපදෙස්:\n{format_instr}\n"
                f"- ලබා දී ඇති සන්දර්භයේ හරියටම ඒ යොමු දිස් නොවේ නම් {ref_note}.\n\n"
                f"පිළිතුර:"
            )

            raw = _call_groq(persona, user_message, model)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"<think>.*$",         "", raw, flags=re.DOTALL).strip()
            raw = _trim_incomplete_sentence(raw)
            raw = _scrub_no_context_sentence(raw)

            # If LLM still couldn't answer with Sinhala context, fall through
            # to the English-context path
            if raw and not _answer_is_no_context(raw) and not _answer_is_weak(raw):
                # Translate the LLM answer to Sinhala
                si_ans = translate_from_english(raw, "si")
                si_ans = _trim_incomplete_sentence(si_ans)

                # Qwen3 review: fix unnatural phrasing and mistranslated terms
                si_ans = _review_translation(raw, si_ans, religion, "si")
                si_ans = _trim_incomplete_sentence(si_ans)
                si_ans = _scrub_question_echo(si_ans)
                si_ans = _apply_respectful_titles(si_ans, religion)

                final_answer, warnings = moderate_output(si_ans, context, religion, "si")
                final_answer = _scrub_question_echo(final_answer)
                final_answer = _apply_respectful_titles(final_answer, religion)

                answer_lower    = final_answer.lower()
                matched         = [r for r in si_results if r["book"].lower() in answer_lower]
                display_results = matched if matched else si_results
                src, scores_out = _unique_sources(display_results)

                return {
                    "answer":         final_answer,
                    "sources":        src,
                    "scores":         scores_out,
                    "flagged":        False,
                    "low_confidence": False,
                    "warnings":       warnings,
                }

        # ── Step 3b: No usable Sinhala chunks — use English context ─
        print("  [si] No Sinhala chunks — falling back to English context + translation")
        return _english_context_then_translate(question, en_query, religion, target_lang="si")

    # ════════════════════════════════════════════════════════════
    # ENGLISH PATH
    # ════════════════════════════════════════════════════════════
    print(f"  [en] lang={lang!r} model={MODEL_DEFAULT!r} question={question[:60]!r}")
    results = search(question, religion=religion, language="en")
    if not results:
        fb = _FALLBACK_MESSAGES.get(lang, _FALLBACK_MESSAGES["en"])
        return {
            "answer":         fb["no_context"],
            "sources":        [],
            "scores":         [],
            "flagged":        False,
            "low_confidence": True,
            "warnings":       [],
        }

    results = _refine_results(results, religion)
    context = "\n\n---\n\n".join(
        f"[Source: {r['book']} | {r.get('pitaka', r.get('testament', ''))}]\n{r['text']}"
        for r in results
    )

    ref_note_map = {
        "Buddhism":    "Do NOT cite specific verse numbers (like SN 56.11)",
        "Christianity": "Do NOT cite specific verse numbers (like John 3:16)",
    }
    ref_note     = ref_note_map.get(religion, "Do NOT cite specific verse numbers")
    format_instr = _format_instructions(religion, _is_list_request(question), "en")
    persona      = _PERSONAS_EN[religion]
    user_message = (
        f"Scripture context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Instructions:\n{format_instr}\n"
        f"- {ref_note} unless those exact references appear in the context above.\n\n"
        f"Answer:"
    )

    # Always use the fast English model for English questions — never Qwen3
    raw_answer = _call_groq(persona, user_message, MODEL_DEFAULT)
    raw_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL)
    raw_answer = re.sub(r"<think>.*$",         "", raw_answer, flags=re.DOTALL).strip()
    raw_answer = _trim_incomplete_sentence(raw_answer)

    final_answer, warnings = moderate_output(raw_answer, context, religion, lang)

    answer_lower    = final_answer.lower()
    matched         = [r for r in results if r["book"].lower() in answer_lower]
    display_results = matched if matched else results
    src, scores_out = _unique_sources(display_results)

    return {
        "answer":         final_answer,
        "sources":        src,
        "scores":         scores_out,
        "flagged":        False,
        "low_confidence": False,
        "warnings":       warnings,
    }