from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise RuntimeError(
        "google-genai SDK is required. Install it via `pip install google-genai google-auth`."
    ) from exc

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

ENV_PATHS = [PACKAGE_ROOT / ".env", PROJECT_ROOT / ".env"]
for env_path in ENV_PATHS:
    if env_path.exists():
        load_dotenv(env_path, override=False)
load_dotenv(override=False)

# The order matters: We prioritize the Hugging Face standard `GOOGLE_API_KEY` first.
API_KEY_ENV_CANDIDATES: tuple[str, ...] = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GENAI_API_KEY",
)


def _find_api_key() -> str | None:
    """Finds the API key from the environment variables."""
    for env_name in API_KEY_ENV_CANDIDATES:
        raw_value = os.getenv(env_name)
        if raw_value and raw_value.strip():
            return raw_value.strip()
    return None


def get_api_key(*, raise_error: bool = True) -> str:
    """
    Return the configured Gemini API key.

    This function supports multiple candidate names for the Gemini API key.
    We prioritize `GOOGLE_API_KEY` as it aligns with the standard secret
    naming convention on Hugging Face Spaces, but fall back to legacy
    names for flexibility when they are present.
    """
    api_key = _find_api_key()
    if api_key:
        return api_key
    if not raise_error:
        return ""
    raise RuntimeError(
        f"No Google Gemini API key found. Please set one of the following "
        f"environment variables: {', '.join(API_KEY_ENV_CANDIDATES)}"
    )



@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Return a cached Gemini client instance."""

    return genai.Client(api_key=get_api_key())


class _ClientProxy:
    """Lazily create the Google Gemini client on first access."""

    def __getattr__(self, name: str):
        return getattr(get_client(), name)


CLIENT = _ClientProxy()

SUMMARY_PATH = PROJECT_ROOT / "test_outputs" / "aistudio_tests_summary.json"
PLAYGROUND_OUTPUT_DIR = PROJECT_ROOT / "playground_outputs"
PLAYGROUND_OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_IMAGE_PATH = PROJECT_ROOT / "test_outputs" / "generated_image.png"
SAMPLE_AUDIO_PATH = PROJECT_ROOT / "test_outputs" / "generated_tts.wav"
SAMPLE_REPORT_PATH = PROJECT_ROOT / "report.pdf"

SAMPLE_IMAGE = str(SAMPLE_IMAGE_PATH) if SAMPLE_IMAGE_PATH.exists() else None
SAMPLE_AUDIO = str(SAMPLE_AUDIO_PATH) if SAMPLE_AUDIO_PATH.exists() else None
SAMPLE_REPORT = str(SAMPLE_REPORT_PATH) if SAMPLE_REPORT_PATH.exists() else None

__all__ = [
    "CLIENT",
    "ClientError",
    "PLAYGROUND_OUTPUT_DIR",
    "PROJECT_ROOT",
    "SAMPLE_AUDIO",
    "SAMPLE_IMAGE",
    "SAMPLE_REPORT",
    "SUMMARY_PATH",
    "get_api_key",
    "get_client",
    "types",
]
