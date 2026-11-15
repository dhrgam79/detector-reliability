import copy
import csv

import pytest

from detrel import plots
from detrel.config import load_config

CORRUPTIONS = ["fog", "snow", "motion_blur", "gaussian_noise", "brightness"]
SCOPES = ["vru", "vehicle"]
METRICS = ["mAP", "mAP50", "dece_t0.05", "dece_t0.3"]


def write_summary(path):
    rows = []
    for c, sevs in [("clean", [0])] + [(c, [1, 2, 3, 4, 5]) for c in CORRUPTIONS]:
        for s in sevs:
            for scope in SCOPES:
                row = {
                    "corruption": c,
                    "severity": s,
                    "scope": scope,
                    "n_gt": 500,
                    "sufficient": "True",
                }
                for m in METRICS:
                    v = 0.7 - 0.05 * s
                    row |= {m: v, f"{m}_lo": v - 0.02, f"{m}_hi": v + 0.02}
                rows.append(row)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_reliability(path):
    rows = []
    for c, s in [("clean", 0), ("gaussian_noise", 5), ("fog", 5)]:
        for scope in SCOPES:
            for t in (0.05, 0.3):
                for k in range(10):
                    empty = t == 0.3 and k < 3
                    rows.append(
                        {
                            "corruption": c,
                            "severity": s,
                            "scope": scope,
                            "score_threshold": t,
                            "bin": k,
                            "bin_low": k / 10,
                            "bin_high": (k + 1) / 10,
                            "count": 0 if empty else 100,
                            "precision": "nan" if empty else k / 12,
                            "confidence": "nan" if empty else k / 10 + 0.05,
                        }
                    )
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def csvs(tmp_path):
    s, r = tmp_path / "summary.csv", tmp_path / "reliability.csv"
    write_summary(s)
    write_reliability(r)
    return s, r


def test_read_csv_types(csvs):
    rows = plots.read_csv(csvs[0])
    assert rows[0]["corruption"] == "clean" and rows[0]["severity"] == 0
    assert isinstance(rows[0]["mAP"], float) and rows[0]["sufficient"] is True


def test_severity_series_prepends_clean(csvs):
    s = plots.severity_series(plots.read_csv(csvs[0]), "vru", "mAP50", ["fog"])["fog"]
    assert s["severity"] == [0, 1, 2, 3, 4, 5]
    assert s["value"][0] == pytest.approx(0.7) and s["value"][5] == pytest.approx(0.45)
    assert s["low"][3] < s["value"][3] < s["high"][3]


def test_reliability_bins_selection_and_missing(csvs):
    rel = plots.read_csv(csvs[1])
    bins = plots.reliability_bins(rel, "fog", 5, "vru", 0.3)
    assert [b["bin"] for b in bins] == list(range(10)) and bins[0]["count"] == 0
    with pytest.raises(KeyError):
        plots.reliability_bins(rel, "snow", 5, "vru", 0.3)


def test_make_figures_writes_pngs(csvs, tmp_path):
    cfg = copy.deepcopy(load_config())
    out = plots.make_figures(cfg, csvs[0], csvs[1], tmp_path / "fig", n_images=750)
    assert [p.name for p in out] == ["fig1_mAP50_vs_severity.png", "fig3_reliability_vru.png"]
    for p in out:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 20_000


def test_palette_has_a_colour_and_marker_per_corruption():
    assert len(plots.SERIES_COLORS) >= len(CORRUPTIONS)
    assert len(set(plots.MARKERS)) >= len(CORRUPTIONS)
