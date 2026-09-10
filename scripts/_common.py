"""Put the repo root on sys.path so ``import pyntaz`` works from any
working directory. Every script imports this first; nothing else here."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
