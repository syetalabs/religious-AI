from deep_translator import GoogleTranslator

LANGUAGE_CODES = {
    "English": "en",
    "Sinhala": "si",
    "Tamil":   "ta",
    "en": "en",
    "si": "si",
    "ta": "ta",
}

def translate_to_english(text: str, source_language: str) -> str:
    source_code = LANGUAGE_CODES.get(source_language, "en")
    if source_code == "en":
        return text
    try:
        translated = GoogleTranslator(source=source_code, target="en").translate(text)
        return translated or text
    except Exception as exc:
        print(f"  [translator] to-English failed: {exc}")
        return text


def translate_from_english(text: str, target_language: str) -> str:
    target_code = LANGUAGE_CODES.get(target_language, "en")
    if target_code == "en":
        return text
    try:
        if len(text) <= 4500:
            translated = GoogleTranslator(source="en", target=target_code).translate(text)
            return translated or text
        else:
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
        return text