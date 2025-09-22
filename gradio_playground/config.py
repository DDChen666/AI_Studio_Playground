from __future__ import annotations

import os
from pathlib import Path

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

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env or the shell environment before launching the playground.")

CLIENT = genai.Client(api_key=API_KEY)

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
    "types",
]
