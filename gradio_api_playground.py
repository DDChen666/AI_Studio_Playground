"""Backwards compatible launcher for the AI Studio Gradio playground."""
from __future__ import annotations

from gradio_playground.main import launch
from gradio_playground.ui import build_demo

__all__ = ["build_demo", "launch"]


if __name__ == "__main__":
    launch()
