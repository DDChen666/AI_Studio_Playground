"""Language utilities for the Gradio playground."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict


class LanguageCode(str, Enum):
    """Supported UI languages."""

    EN = "en"
    ZH = "zh"


LANGUAGE_LABELS: Dict[LanguageCode, str] = {
    LanguageCode.EN: "English",
    LanguageCode.ZH: "中文",
}

LABEL_TO_LANGUAGE = {label: code for code, label in LANGUAGE_LABELS.items()}

TRANSLATIONS_PATH = Path(__file__).with_name("translations_map.json")


def _load_translations() -> Dict[str, Dict[str, str]]:
    try:
        payload = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - configuration guard
        raise RuntimeError(
            "Missing translations_map.json. Run the project from the repository root."
        ) from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration guard
        raise RuntimeError(f"Invalid translations_map.json: {exc.msg} (line {exc.lineno}).") from exc

    if not isinstance(payload, dict):  # pragma: no cover - configuration guard
        raise RuntimeError("Translation map must be a JSON object keyed by string identifiers.")

    normalized: Dict[str, Dict[str, str]] = {}
    for key, bucket in payload.items():
        if not isinstance(bucket, dict):  # pragma: no cover - configuration guard
            raise RuntimeError(f"Translation bucket for '{key}' must be a JSON object.")
        normalized[key] = {str(lang): str(message) for lang, message in bucket.items()}
    return normalized


TRANSLATIONS: Dict[str, Dict[str, str]] = _load_translations()


def get_text(key: str, language: LanguageCode) -> str:
    lang_key = language.value if isinstance(language, LanguageCode) else str(language)
    if key not in TRANSLATIONS:
        raise KeyError(f"Missing translation key: {key}")
    bucket = TRANSLATIONS[key]
    if lang_key not in bucket:
        raise KeyError(f"Missing translation for key '{key}' and language '{lang_key}'.")
    return bucket[lang_key]


def bilingual_tab_label(key: str) -> str:
    return f"{get_text(key, LanguageCode.EN)} / {get_text(key, LanguageCode.ZH)}"


__all__ = [
    "LanguageCode",
    "LANGUAGE_LABELS",
    "LABEL_TO_LANGUAGE",
    "TRANSLATIONS",
    "TRANSLATIONS_PATH",
    "bilingual_tab_label",
    "get_text",
]
