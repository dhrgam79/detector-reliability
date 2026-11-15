"""Load and validate the experiment configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

REQUIRED_SECTIONS = (
    "seed",
    "paths",
    "data",
    "corruptions",
    "model",
    "matching",
    "metrics",
    "calibration",
    "plots",
)


class ConfigError(ValueError):
    """Raised when the configuration is missing keys or has invalid values."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file and validate it.

    Args:
        path: Path to the YAML file. Defaults to ``configs/default.yaml`` in the repo.

    Returns:
        The parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If required sections are missing or values are inconsistent.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    validate_config(cfg)
    logger.info("Loaded config from %s", config_path)
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path: absolute paths are kept, relative ones are taken from the repo root.

    Args:
        path: Path from the config.

    Returns:
        An absolute path.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that a config dictionary is complete and internally consistent.

    Args:
        cfg: Parsed configuration dictionary.

    Raises:
        ConfigError: On the first problem found.
    """
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    data = cfg["data"]
    if data["calibration_size"] + data["evaluation_size"] != data["subset_size"]:
        raise ConfigError("calibration_size + evaluation_size must equal subset_size")
    if not data["driving_classes"]:
        raise ConfigError("driving_classes must not be empty")
    grouped = [c for ids in data["class_groups"].values() for c in ids]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(data["driving_classes"].values()):
        raise ConfigError("class_groups must partition driving_classes (disjoint, complete)")

    severities = cfg["corruptions"]["severities"]
    if not severities or any(s not in range(1, 6) for s in severities):
        raise ConfigError("corruption severities must be a non-empty subset of 1..5")

    for t in cfg["metrics"]["dece_score_thresholds"]:
        if not 0.0 <= t < 1.0:
            raise ConfigError(f"D-ECE score threshold out of [0, 1): {t}")
    if min(cfg["metrics"]["dece_score_thresholds"]) < cfg["model"]["score_threshold"]:
        raise ConfigError("D-ECE thresholds cannot be below the inference score_threshold")

    if not 0.0 < cfg["matching"]["iou_threshold"] <= 1.0:
        raise ConfigError("iou_threshold must be in (0, 1]")
    if cfg["metrics"]["dece_bins"] < 1:
        raise ConfigError("dece_bins must be >= 1")

    lo, hi = cfg["calibration"]["temperature_bounds"]
    if not 0.0 < lo < hi:
        raise ConfigError("temperature_bounds must satisfy 0 < low < high")
