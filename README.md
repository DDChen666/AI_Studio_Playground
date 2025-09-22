# Epic AI AutoVideoMachine Agent

This project bundles local utilities for exercising Google AI Studio APIs and a Gradio-based playground for manual testing. The repository expects a dedicated Python virtual environment in `.venv` and relies on a handful of lightweight Python packages.

## Prerequisites
- Python 3.13 (the existing `.venv` targets CPython 3.13.7).
- A valid Gemini API key with access to the desired models.
- PowerShell (on Windows) or a POSIX shell (macOS/Linux) for activating the virtual environment.

## Create or Reuse the Virtual Environment
If `.venv` already exists you can reuse it; otherwise create it with:

```powershell
python -m venv .venv
```

### Activation
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS/Linux (bash/zsh):**
  ```bash
  source .venv/bin/activate
  ```

After activation, the prompt should show the `(.venv)` prefix.

## Initial Dependency Installation
Upgrade `pip`, then install runtime and tooling dependencies:

```bash
pip install --upgrade pip
pip install gradio google-genai google-auth python-dotenv
```

> Tip: capture the installed set with `pip freeze > requirements.txt` if you need to share the exact environment snapshot.

## Environment Variables
Create a `.env` file (or export variables in your shell) so the automated tests can authenticate:

```
GEMINI_API_KEY=your_api_key_here
```

The test harness loads `.env` automatically via `python-dotenv`.

## Useful Commands
- Run the Gradio playground (writes generated assets to `playground_outputs/`):
  ```bash
  python gradio_api_playground.py
  ```
- Execute the AI Studio regression tests (writes JSON results under `test_outputs/`):
  ```bash
  python tests/run_aistudio_tests.py
  ```

Deactivate the virtual environment any time with `deactivate`.
