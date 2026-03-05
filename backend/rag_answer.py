import os
import re
import time
import requests
from dotenv import load_dotenv
from retrieve import search

# ────────────────────────────────────────────────────────────────
# Groq API settings
# ────────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# Models
MODEL_DEFAULT  = "llama-3.1-8b-instant"   # English
MODEL_SINHALA  = "qwen/qwen3-32b"         # Sinhala and Tamil (review + generation)

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
- When referencing scripture, mention the source book naturally (e.g. "As written in the Gospel of John...").

CRITICAL — Quoting rules:
- Do NOT cite specific verse references (e.g. John 3:16, Romans 8:28) unless those exact references appear word-for-word in the provided context.
- Do NOT reproduce or paraphrase text not present in the provided context.
- Never invent or recall verse references from memory — only use what is explicitly in the context.""",
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
- ශාස්ත්‍රය යොමු කරන විට ප්‍රභව ග්‍රන්ථය ස්වාභාවිකව සඳහන් කරන්න (උදා. "යොහාන් ශුභාරංචිය හි ලියා ඇති පරිදි...").

තීරණාත්මක — උද්ධෘත නීති:
- ලබා දී ඇති සන්දර්භයේ හරියටම ඒ යොමු දිස් නොවේ නම්, නිශ්චිත වාක්‍ය යොමු (John 3:16, Romans 8:28 වැනි) උද්ධෘත නොකරන්න.
- ලබා දී ඇති සන්දර්භයේ නැති පාඨ නැවත නිෂ්පාදනය නොකරන්න හෝ ප්‍රතිනිර්මාණය නොකරන්න.
- කිසිදා මතකයෙන් උද්ධෘත නිර්මාණය නොකරන්න — සන්දර්භයේ ඇති දෙය පමණක් භාවිත කරන්න.""",
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
        "no_context": "இதை சரியாக பதிலளிக்க போதுமான நம்பகமான மறைநூல் சூழல் என்னிடம் இல்லை.",
    },
}

_VERSE_REF_PATTERNS = {
    "Buddhism": re.compile(
        r"\b(SN|MN|DN|AN|KN|Dhp|Ud|Iti|Snp|Thag|Thig|Ja|Cp|Mil|Pv|Vv|Vb|Ds|Kv|Pp|Ps|Ya)\s*\d+[\.\d]*\b",
        re.IGNORECASE,
    ),
    "Christianity": re.compile(
        r"\b(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|"
        r"1\s*Samuel|2\s*Samuel|1\s*Kings|2\s*Kings|1\s*Chronicles|2\s*Chronicles|"
        r"Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs?|Ecclesiastes|Song\s*of\s*Solomon|"
        r"Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|"
        r"Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|"
        r"Matthew|Mark|Luke|John|Acts|Romans|"
        r"1\s*Corinthians|2\s*Corinthians|Galatians|Ephesians|Philippians|"
        r"Colossians|1\s*Thessalonians|2\s*Thessalonians|1\s*Timothy|2\s*Timothy|"
        r"Titus|Philemon|Hebrews|James|1\s*Peter|2\s*Peter|"
        r"1\s*John|2\s*John|3\s*John|Jude|Revelation|Rev)\s+\d+:\d+(?:-\d+)?\b",
        re.IGNORECASE,
    ),
}

_OPINION_SIGNALS = [
    r"\bi (think|believe|feel|personally|would say)\b",
    r"\bmy (view|opinion|take|perspective)\b",
]

# ════════════════════════════════════════════════════════════════
# Moderation functions
# ════════════════════════════════════════════════════════════════

def moderate_input(query: str) -> tuple[bool, str]:
    q = query.lower().strip()
    for phrase in _UNSAFE_PHRASES:
        if phrase in q:
            return False, "comparative_or_unsafe"
    for pattern in _COMPARATIVE_PATTERNS:
        if re.search(pattern, q):
            return False, "comparative_religion"
    for pattern in _HATE_PATTERNS:
        if re.search(pattern, q):
            return False, "hate_speech"
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

def _translate_query_to_english(question: str) -> str:
    """
    Translate a Sinhala or Tamil question to English for FAISS retrieval.

    Two-step process:
      Step 1 — Fast translation via llama (MODEL_DEFAULT).
      Step 2 — Nuance verification via Qwen3 (MODEL_SINHALA).
                Qwen3 checks whether the key concept/term in the original
                question was translated with its correct Buddhist meaning.
                If not, it returns a corrected English query.

    This prevents mistranslations like:
      ஏக்கம்  → "suffering"  (wrong) should be → "longing / craving / tanha"
      ශූන්‍යතාව → "emptiness" (ok)   or → "sunyata / void" (also ok)

    Falls back to the Step 1 result if Step 2 fails.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    # ── Step 1: Fast translation ─────────────────────────────────
    payload_1 = {
        "model":    MODEL_DEFAULT,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a translator. Translate the following question into "
                    "concise English suitable for a Buddhist scripture search. "
                    "Preserve the precise meaning of every religious or philosophical term. "
                    "Output ONLY the English translation — no explanation, no extra text."
                ),
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
        return question   # total failure — return original

    # ── Step 2: Nuance verification ──────────────────────────────
    # Detect source language for the verification prompt
    src_lang = "Tamil" if _is_tamil(question) else "Sinhala"

    payload_2 = {
        "model":    MODEL_SINHALA,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a Buddhist scholar fluent in Pali, English, Sinhala, and Tamil.\n\n"
                    "Your task: verify that an English translation of a Buddhist question "
                    "correctly preserves the nuance of the key concept or term.\n\n"
                    "Rules:\n"
                    "1. Identify the main Buddhist concept or term in the original question.\n"
                    "2. Check if the English translation captures its correct Buddhist meaning.\n"
                    "   - Example of BAD translation: ஏக்கம் → 'suffering' "
                    "(ஏக்கம் means longing/yearning, closer to taṇhā, not dukkha)\n"
                    "   - Example of GOOD translation: ஏக்கம் → 'longing or craving (tanha) in Buddhism'\n"
                    "   - Example of GOOD translation: ශූන්‍යතාව → 'sunyata (emptiness) in Buddhism'\n"
                    "3. If the translation is correct, return it UNCHANGED.\n"
                    "4. If the translation is wrong or loses important nuance, return a corrected "
                    "English query that will retrieve the right Buddhist scripture passages.\n"
                    "5. Output ONLY the final English query — no explanation, no commentary."
                ),
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
        "max_tokens":  120,
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
        "- Start with a simple, human explanation a newcomer can understand.\n"
        "- Then support it with what the scripture says, citing the source book "
        "naturally (e.g. \"As written in the Gospel of John...\").\n"
        "- If the teaching has a practical or spiritual dimension, briefly mention it.\n"
        "- Do not use bullet points or lists. Write in flowing, warm prose."
    )


# Sentence trim
# Matches sentence-ending punctuation for both English (.!?) and Sinhala (។ and similar)
_SENT_END   = re.compile(r'[.!?។]\s*$')
_SENT_SPLIT = re.compile(r'[.!?។][\'"\)\]]*(?=\s|$)')

def _trim_incomplete_sentence(text: str) -> str:
    """
    Trim a response that ends mid-sentence due to token truncation.
    If the text already ends with proper punctuation, return it unchanged.
    Handles both English (.!?) and Sinhala (។) sentence endings.
    Also handles the common case where Sinhala text ends without punctuation
    but the last word is a complete thought — in that case return as-is
    rather than aggressively truncating.
    """
    text = text.strip()
    if not text:
        return text

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


# Respectful titles
_BUDDHA_REPLACEMENTS = [
    ("\u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0dcf\u0da2\u0dcf",
     "\u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0da2\u0dcf\u0dab\u0db1\u0dca \u0dc0\u0dc4\u0db1\u0dca\u0dc3\u0dda"),
    ("\u0d9c\u0dbd\u0dca\u0db0\u0db8 \u0db6\u0dd4\u0daf\u0dd4",
     "\u0d9c\u0dbd\u0dca\u0db0\u0db8 \u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0da2\u0dcf\u0dab\u0db1\u0dca \u0dc0\u0dc4\u0db1\u0dca\u0dc3\u0dda"),
    ("\u0d9c\u0dbd\u0dca\u0db0\u0db8",
     "\u0d9c\u0dbd\u0dca\u0db0\u0db8 \u0db6\u0dd4\u0daf\u0dd4\u0dbb\u0da2\u0dcf\u0dab\u0db1\u0dca \u0dc0\u0dc4\u0db1\u0dca\u0dc3\u0dda"),
]

def _apply_respectful_titles(text):
    for a, b in _BUDDHA_REPLACEMENTS:
        text = text.replace(a, b)
    return text


# No-context detection & scrubbing
_NO_CONTEXT_SI = "මෙය නිවැරදිව පිළිතුරු දීමට ප්‍රාණවත් විශ්වාසදායක ශාස්ත්‍රීය සන්දර්භයක් මා සතු නොවේ"

def _answer_is_no_context(text):
    return _NO_CONTEXT_SI in text.strip()

# Phrases that indicate the LLM is hedging / explaining lack of context
# rather than actually answering the question.
_WEAK_CONTEXT_FRAGMENTS = [
    "සන්දර්භයේ",           # "in the given context"
    "ශාස්ත්‍රීය සන්දර්භ",  # "scriptural context"
    "ප්‍රාණවත් විශ්වාසදායක", # exact no-context phrase
    "ප්‍රමාණවත් විස්තර",   # "sufficient detail"
    "ප්‍රමාණවත් නොවේ",     # "not sufficient"
    "නොසඳහන් වේ",           # "is not mentioned"
    "සඳහන් නොවේ",           # "not mentioned"
    "වෙනත් ශාස්ත්‍රීය",    # "other scriptural texts"
    "සොයා බැලිය යුතු",      # "should look up"
    "යොමු කළ යුතු",         # "should refer to"
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
                "இது ஒரு கிறிஸ்தவ பதில். சரியான கிறிஸ்தவ தமிழ் சொற்களை பயன்படுத்தவும் "
                "(எ.கா. திரித்துவம், இரட்சிப்பு, விசுவாசம், அன்பு, பைபிள்).\n"
                "- பௌத்த சொற்களை கிறிஸ்தவ பதிலில் சேர்க்க வேண்டாம்."
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
                "මෙය ක්‍රිස්තියානි පිළිතුරකි. නිවැරදි ක්‍රිස්තියානි සිංහල පාරිභාෂිතය භාවිත කරන්න "
                "(නිදසුන්: ත්‍රිත්වය, ගැලවීම, ඇදහිල්ල, කරුණාව, ශුද්ධ ලියවිල්ල, බයිබලය).\n"
                "- 'ත්‍රිපිටකය', 'නිර්වාණය' හෝ බෞද්ධ සංකල්ප ක්‍රිස්තියානි පිළිතුරක "
                "ඇතුළත් නොකරන්න."
            ),
        }.get(religion, "නිවැරදි සහ ස්වාභාවික සිංහල ආගමික පාරිභාෂිතය භාවිත කරන්න.")

        system_prompt = (
            "ඔබ සිංහල භාෂා විශේෂඥයෙකු සහ ආගමික පරිවර්තන සමාලෝචකයෙකි.\n\n"
            "ඔබේ කාර්යය:\n"
            "1. ලබා දී ඇති සිංහල පරිවර්තනය සමාලෝචනය කරන්න.\n"
            "2. අස්වාභාවික, වැරදි, හෝ Google Translate ගැටළු නිවැරදි කරන්න.\n"
            f"3. {religion_note}\n"
            "4. වෙනත් ආගම්වලට අයත් පද හෝ සංකල්ප (cross-religion terminology) "
            "ඉවත් කර නිවැරදි ආගමික පාරිභාෂිතයෙන් ප්‍රතිස්ථාපනය කරන්න.\n"
            "5. මුල් ඉංග්‍රීසි පිළිතුරේ අර්ථය හරියටම ආරක්ෂා කරන්න.\n"
            "6. නිවැරදි කළ සිංහල පාඨය පමණක් ලබා දෙන්න. "
            "කිසිදු පැහැදිලි කිරීමක්, ඉංග්‍රීසි, හෝ ටිප්පණි නොතිබිය යුතුය."
        )

    user_message = (
        f"{'மூல ஆங்கில பதில்' if target_lang == 'ta' else 'මුල් ඉංග්‍රීසි පිළිතුර'}:\n"
        f"{original_english}\n\n"
        f"{'தமிழ் மொழிபெயர்ப்பு (மதிப்பாய்வு செய்யவும்)' if target_lang == 'ta' else 'සිංහල පරිවර්තනය (සමාලෝචනය කරන්න)'}:\n"
        f"{translated_text}"
    )

    reviewed = _call_groq(system_prompt, user_message, MODEL_SINHALA)
    reviewed = re.sub(r"<think>.*?</think>", "", reviewed, flags=re.DOTALL)
    reviewed = re.sub(r"<think>.*$",         "", reviewed, flags=re.DOTALL).strip()

    if not reviewed or len(reviewed) < 20:
        print(f"  [review] Qwen3 review returned empty — keeping original translation")
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
        return {
            "answer": _FALLBACK_MESSAGES.get(target_lang, _FALLBACK_MESSAGES["en"])["no_context"],
            "sources": [], "scores": [], "flagged": False,
            "low_confidence": True, "warnings": ["no_en_context"],
        }

    en_res = _refine_results(en_res, religion)
    en_ans = _build_english_answer(en_query, en_res, religion)

    if not en_ans or "I do not have enough" in en_ans:
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
    if target_lang == "si":
        translated = _apply_respectful_titles(translated)

    warning_key = f"used_en_context_with_translation_{target_lang}"
    matched = [r for r in en_res if r["book"].lower() in translated.lower()]
    disp    = matched if matched else en_res
    return {
        "answer":         translated,
        "sources":        [r["book"]  for r in disp],
        "scores":         [r["score"] for r in disp],
        "flagged":        False,
        "low_confidence": False,
        "warnings":       [warning_key],
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

    # ── Layer 1: Input moderation ────────────────────────────────
    is_safe, reason = moderate_input(question)
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
    # TAMIL PATH — no Tamil scripture in DB, always use English
    # context + translate to Tamil
    # ════════════════════════════════════════════════════════════
    if lang == "ta":
        en_query = _translate_query_to_english(question)
        print(f"  [ta] EN query: {en_query!r}")
        return _english_context_then_translate(question, en_query, religion, target_lang="ta")

    # ════════════════════════════════════════════════════════════
    # SINHALA PATH
    # ════════════════════════════════════════════════════════════
    if lang == "si":
        from retrieve import search_sinhala_direct
        from translator import translate_from_english

        # Step 1: Translate question to English for FAISS embedding
        en_query = _translate_query_to_english(question)
        print(f"  [si] EN query: {en_query!r}")

        # Step 2: Check for relevant Sinhala chunks in DB
        si_results = search_sinhala_direct(en_query, religion=religion)
        print(f"  [si] Sinhala chunks found: {len(si_results)}")

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
                si_ans = _apply_respectful_titles(si_ans)

                final_answer, warnings = moderate_output(si_ans, context, religion, "si")
                final_answer = _apply_respectful_titles(final_answer)

                answer_lower    = final_answer.lower()
                matched         = [r for r in si_results if r["book"].lower() in answer_lower]
                display_results = matched if matched else si_results

                return {
                    "answer":         final_answer,
                    "sources":        [r["book"]  for r in display_results],
                    "scores":         [r["score"] for r in display_results],
                    "flagged":        False,
                    "low_confidence": False,
                    "warnings":       warnings,
                }

        # ── Step 3b: No usable Sinhala chunks — use English context ─
        print("  [si] No Sinhala chunks — falling back to English context + translation")
        return _english_context_then_translate(question, en_query, religion, target_lang="si")

    # ════════════════════════════════════════════════════════════
    # ENGLISH PATH  (unchanged)
    # ════════════════════════════════════════════════════════════
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

    raw_answer = _call_groq(persona, user_message, model)
    raw_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL)
    raw_answer = re.sub(r"<think>.*$",         "", raw_answer, flags=re.DOTALL).strip()
    raw_answer = _trim_incomplete_sentence(raw_answer)

    final_answer, warnings = moderate_output(raw_answer, context, religion, lang)

    answer_lower    = final_answer.lower()
    matched         = [r for r in results if r["book"].lower() in answer_lower]
    display_results = matched if matched else results

    return {
        "answer":         final_answer,
        "sources":        [r["book"]  for r in display_results],
        "scores":         [r["score"] for r in display_results],
        "flagged":        False,
        "low_confidence": False,
        "warnings":       warnings,
    }