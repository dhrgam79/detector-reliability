"""Command-line entry points. Subcommands are added as milestones land."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from detrel import data
from detrel.config import load_config

logger = logging.getLogger("detrel")


def prepare_data(cfg: dict[str, Any]) -> data.Split:
    """Download annotations and subset images, then create or verify the split manifest.

    Args:
        cfg: Validated config.

    Returns:
        The calibration/evaluation split.
    """
    ann_path = data.ensure_annotations(cfg)
    coco = data.load_annotations(ann_path)
    split = data.make_split(cfg, coco, data.sha256_file(ann_path))
    data.ensure_images(cfg, coco, split.calibration + split.evaluation)
    logger.info(
        "Data ready: %d calibration + %d evaluation images",
        len(split.calibration),
        len(split.evaluation),
    )
    return split


COMMANDS = {"prepare-data": prepare_data}


def main(argv: list[str] | None = None) -> int:
    """Run the ``detrel`` command-line interface.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="detrel")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    COMMANDS[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
