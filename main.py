"""Entry point for CareerCraft.

Usage:
    python main.py                          # Interactive mode
    python main.py --resume resume.pdf       # Start with PDF resume
    python main.py -r resume.pdf -j "JD..."  # Resume + JD
    python main.py --model qwen-max          # Use specific model
"""

from app.cli.app import cli

if __name__ == "__main__":
    cli()
