# AI Studio API Playground

[繁體中文](README.zh-TW.md) | English

## Overview
AI Studio API Playground is a modular Gradio application for exploring Google Gemini
models interactively. The project bundles reusable backend utilities, a structured
UI layer with bilingual support (English/Traditional Chinese), and clear
configuration boundaries suitable for production-grade experimentation.

## Features
- **Bilingual UI:** Seamlessly switch between English and Traditional Chinese via
the in-app language selector backed by `translations_map.json`.
- **Modular architecture:** Discrete modules for UI, API orchestration, utilities,
and configuration under the `gradio_playground/` package.
- **Extensible scenarios:** Reuses outputs from `test_outputs/aistudio_tests_summary.json`
to pre-populate prompts and defaults.
- **Asset management:** Generated files are written to `playground_outputs/` with
helpers to convert PCM audio into WAV containers when required.

## Project Layout
```
gradio_playground/
├── __init__.py
├── api_calls.py        # Backend orchestration for Gemini API calls
├── config.py           # Environment loading and client bootstrap
├── main.py             # Application entry point
├── translations.py     # Runtime loader for the bilingual text catalogue
├── translations_map.json
├── ui.py               # Gradio Blocks layout (build_demo)
├── utils.py            # Shared helper functions
└── .env                # Local environment template (keep values private)
```
Other notable files:
- `gradio_api_playground.py`: Backwards-compatible launcher importing the modular
  implementation.
- `tests/run_aistudio_tests.py`: Regression harness for automated API checks.

## Prerequisites
- Python 3.10 or later (tested with CPython 3.13.x).
- A Gemini API key with access to the desired models.
- `pip` for dependency management.

## Quick Start
1. **Clone and enter the repository.**
   ```bash
   git clone <repo-url>
   cd AI_Studio_Playground
   ```
2. **Create and activate a virtual environment.**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\Activate.ps1
   ```
3. **Install dependencies.**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt  # or install gradio google-genai google-auth python-dotenv
   ```
   If `requirements.txt` is not available, install the packages shown in the comment.
4. **Configure environment variables.**
   Update `gradio_playground/.env` (or export variables in your shell) with a
   valid Gemini API key:
   ```env
   GEMINI_API_KEY=your-real-api-key
   ```
   The application uses [`python-dotenv`](https://pypi.org/project/python-dotenv/)
   to load values from `.env` automatically.

## Running the Playground
Launch the UI using either entry point:
```bash
python -m gradio_playground.main
# or
python gradio_api_playground.py
```
Gradio will queue background tasks and open a local interface. Generated assets
are written to `playground_outputs/` at the repository root.

## Automated Tests
A lightweight regression script is provided for scenarios captured in
`test_outputs/aistudio_tests_summary.json`:
```bash
GEMINI_API_KEY=your-real-api-key python tests/run_aistudio_tests.py
```

## Localization Workflow
- All UI strings reside in `gradio_playground/translations_map.json`.
- Add new keys to the JSON map and reference them via `get_text` in
  `gradio_playground/translations.py`.
- When adding tabs or controls in `ui.py`, register the components so they
  receive live updates when the language switch changes.

## Support
For issues or feature requests, please open a GitHub issue describing the
behaviour, expected outcome, and reproduction steps.
