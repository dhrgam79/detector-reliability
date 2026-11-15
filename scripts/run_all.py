"""Reproduce every table and figure from a fresh clone: ``python scripts/run_all.py``.

Extra arguments are passed to the CLI before the subcommand, e.g. ``--config my.yaml``.
"""

import sys

from detrel.cli import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "sweep"]))
