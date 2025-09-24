---
title: AI Studio API Playground
emoji: 🎛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# AI Studio API Playground!

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
   pip install -r requirements.txt
   ```
#### 4. Configure Environment Variables

This application requires a Google Gemini API key to function.

**A) For Local Development:**

1.  In the root directory of the project, create a file named `.env`.
2.  Add your API key to this file. The application will recognize `GOOGLE_API_KEY`.

    ```
    # Inside your .env file
    GOOGLE_API_KEY="your-real-api-key-here"
    ```
The application uses the `python-dotenv` library to automatically load this key when you run it locally.

**B) For Hugging Face Space Deployment:**

Do **not** upload your `.env` file. Instead, you must set the API key using Hugging Face's built-in secure storage:

1.  Go to your Space's **Settings** tab.
2.  Find the **Repository secrets** section.
3.  Create a new secret with:
    *   **Name:** `GOOGLE_API_KEY`
    *   **Value:** Paste your actual Google Gemini API key here.

The Space will automatically load this secret as an environment variable when it runs.

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
# or equivalently
GOOGLE_API_KEY=your-real-api-key python tests/run_aistudio_tests.py
```

## Deploying to Hugging Face Spaces
1. **Create a new Gradio Space.** Choose the “Gradio” SDK and link it to a Git
   repository (public or private depending on your needs).
2. **Push the project files.** The provided `app.py` exports the `demo` object
   expected by Spaces, and `requirements.txt` lists the runtime dependencies.
3. **Configure secrets.** Navigate to *Settings → Secrets* and add
   `GOOGLE_API_KEY` (or any other supported name listed above) with your Gemini
   API key.
4. **Select hardware (optional).** Free CPU tiers are sufficient for testing.
   Ensure the Space type allows outbound internet access so requests can reach
   the Gemini API.
5. **Trigger a build.** When the Space restarts it installs dependencies from
   `requirements.txt` and launches the Gradio app defined in `app.py`.
6. **Smoke test the UI.** Exercise several tabs, verify bilingual switching,
   and ensure generated assets appear under the `playground_outputs/` directory.

## Localization Workflow
- All UI strings reside in `gradio_playground/translations_map.json`.
- Add new keys to the JSON map and reference them via `get_text` in
  `gradio_playground/translations.py`.
- When adding tabs or controls in `ui.py`, register the components so they
  receive live updates when the language switch changes.

## Support
For issues or feature requests, please open a GitHub issue describing the
behaviour, expected outcome, and reproduction steps.
