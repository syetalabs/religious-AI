from deep_translator import GoogleTranslator
from langdetect import detect

# ─── Language code map ───────────────────────────────────────
# Maps the language name used in the frontend to Google Translate codes
LANGUAGE_CODES = {
    "English": "en",
    "Sinhala": "si",
    "Tamil":   "ta",
}

def translate_to_english(text: str, source_language: str) -> str:
    """
    Translate user input from Sinhala or Tamil into English
    before passing to the RAG pipeline.
    Returns the original text unchanged if language is English.
    """
    source_code = LANGUAGE_CODES.get(source_language, "en")

    if source_code == "en":
        return text

    try:
        translated = GoogleTranslator(
            source=source_code,
            target="en"
        ).translate(text)
        return translated or text
    except Exception as exc:
        print(f"  [translator] to-English failed: {exc}")
        return text  # Fall back to original — RAG will still attempt an answer


def translate_from_english(text: str, target_language: str) -> str:
    """
    Translate the RAG answer from English into the user's chosen language.
    Returns the original text unchanged if target is English.
    """
    target_code = LANGUAGE_CODES.get(target_language, "en")

    if target_code == "en":
        return text

    try:
        # Google Translate has a 5000-char limit per call.
        # Split on sentence boundaries if the answer is long.
        if len(text) <= 4500:
            translated = GoogleTranslator(
                source="en",
                target=target_code
            ).translate(text)
            return translated or text
        else:
            # Split into sentences and translate in chunks
            sentences = text.replace(".\n", ". ").split(". ")
            chunks, current = [], ""
            for sentence in sentences:
                if len(current) + len(sentence) < 4500:
                    current += sentence + ". "
                else:
                    chunks.append(current.strip())
                    current = sentence + ". "
            if current:
                chunks.append(current.strip())

            translated_chunks = []
            for chunk in chunks:
                t = GoogleTranslator(source="en", target=target_code).translate(chunk)
                translated_chunks.append(t or chunk)
            return " ".join(translated_chunks)

    except Exception as exc:
        print(f"  [translator] from-English failed: {exc}")
        return text  # Fall back to English answer