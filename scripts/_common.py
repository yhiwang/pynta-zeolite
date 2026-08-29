"""Shared bits for the step scripts: put the repo on sys.path so the scripts
run from anywhere, and the argparse options every step takes."""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pyntaz import config  # noqa: E402


def step_parser(description):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=config.DEFAULT_RUN_DIR,
                        help="run directory (default: %(default)s)")
    return parser


def layout_from(args):
    return config.RunLayout(args.run_dir)
