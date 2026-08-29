#!/usr/bin/env python
"""Relax one adsorbate config with MACE, framework frozen. This is the worker
that job.sh runs inside every relaxation directory (step 2 copies it there).

    python relax_one.py <stem>_init.xyz

Writes relax.xyz, relax.traj and relax.log next to the input. The repo must
be on PYTHONPATH (job.sh exports it from pyntaz/config.py REPO; set
PYNTAZ_REPO if the checkout lives elsewhere).
"""

import os
import sys

try:
    from pyntaz.relax import relax_structure
except ImportError:
    sys.exit("cannot import pyntaz -- put the repo on PYTHONPATH, e.g.\n"
             "    export PYTHONPATH=%s:$PYTHONPATH"
             % os.environ.get("PYNTAZ_REPO", "/path/to/pyntaz-repo"))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    relax_structure(sys.argv[1])
