import copy

import pytest

import detrel
from detrel.config import ConfigError, load_config, validate_config


def test_package_imports():
    assert detrel.__version__


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert cfg["data"]["subset_size"] == 500
    assert cfg["model"]["score_threshold"] == 0.05


@pytest.fixture
def cfg():
    return load_config()


def test_split_sizes_must_sum(cfg):
    bad = copy.deepcopy(cfg)
    bad["data"]["evaluation_size"] = 300
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_severity_out_of_range(cfg):
    bad = copy.deepcopy(cfg)
    bad["corruptions"]["severities"] = [0, 6]
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_dece_threshold_below_inference_threshold(cfg):
    bad = copy.deepcopy(cfg)
    bad["metrics"]["dece_score_thresholds"] = [0.01]
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_missing_section(cfg):
    bad = copy.deepcopy(cfg)
    del bad["calibration"]
    with pytest.raises(ConfigError):
        validate_config(bad)
