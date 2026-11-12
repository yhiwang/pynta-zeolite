"""Put the repo root and scripts/ on sys.path so ``import pyntaz`` and
``import layout`` work from any working directory. Every demo script
imports this first; nothing else here."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.join(REPO, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
