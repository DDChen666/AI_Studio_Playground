"""Application entry point for the AI Studio Gradio playground."""
from __future__ import annotations

from typing import Any, Optional

import gradio as gr

from .ui import build_demo


def launch(*, inbrowser: Optional[bool] = None, share: Optional[bool] = None, **kwargs: Any) -> gr.Blocks:
    """Construct and launch the Gradio Blocks app.

    Parameters
    ----------
    inbrowser:
        Forwarded to :meth:`gradio.Blocks.launch` to control whether the UI should
        open a browser tab automatically.
    share:
        Forwarded to :meth:`gradio.Blocks.launch` to request a public Gradio
        share link.
    **kwargs:
        Additional keyword arguments are forwarded directly to
        :meth:`gradio.Blocks.launch` to keep parity with the original script.

    Returns
    -------
    gradio.Blocks
        The instantiated Blocks application. Returning the object makes it
        simple to compose in tests if needed.
    """

    demo = build_demo()
    demo.queue()
    demo.launch(inbrowser=inbrowser, share=share, **kwargs)
    return demo


def main() -> None:
    """CLI entry point used by ``python -m gradio_playground.main``."""

    launch()


__all__ = ["launch", "main"]


if __name__ == "__main__":
    main()
