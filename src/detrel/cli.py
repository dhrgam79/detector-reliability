"""Command-line entry points. Subcommands are added as milestones land."""

from __future__ import annotations

import argparse
import logging

from detrel.config import load_config


def main(argv: list[str] | None = None) -> int:
    """Run the ``detrel`` command-line interface.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="detrel")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    logging.getLogger("detrel").info("Config OK (seed=%s)", cfg["seed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
