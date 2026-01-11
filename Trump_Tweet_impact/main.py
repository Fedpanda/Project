"""
main.py (project root)

Submission entry point.

This file exists to satisfy the required repository structure:
- main.py at repo root

It forwards execution to the package orchestrator:
- src.main

Examples
--------
python main.py --models --aggregate --analyze
python main.py --events --features --build --models --aggregate --analyze --with-llm
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    # Run src.main as if: python -m src.main
    runpy.run_module("src.main", run_name="__main__")


if __name__ == "__main__":
    main()
