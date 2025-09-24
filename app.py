"""Entry point for Hugging Face Spaces deployment."""

from gradio_playground.ui import build_demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
