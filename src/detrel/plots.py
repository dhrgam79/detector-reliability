"""Figures built only from the committed result CSVs (no recomputation).

Figures use an opaque white background so they read in both light and dark GitHub themes.
Series colours are the validated categorical palette, assigned in fixed order; because three
of them sit below 3:1 contrast on white, every series also gets a distinct marker and a direct
end-of-line label (secondary encoding), in addition to the legend.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# Validated categorical palette (light mode), fixed order. Never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SCOPE_TITLES = {
    "driving": "Driving classes (8)",
    "vru": "Vulnerable road users",
    "vehicle": "Vehicles",
    "infrastructure": "Infrastructure",
    "all80": "All 80 COCO classes",
}
METRIC_TITLES = {
    "mAP": "mAP@[.5:.95]",
    "mAP50": "mAP@0.5",
    "dece_t0.05": "D-ECE (score ≥ 0.05)",
    "dece_t0.3": "D-ECE (score ≥ 0.3)",
    "dece_cal_t0.05": "D-ECE after scaling (score ≥ 0.05)",
    "dece_cal_t0.3": "D-ECE after scaling (score ≥ 0.3)",
}


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a result CSV, converting numeric fields to float (and ``severity``, ``bin`` to int)."""
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        row: dict[str, Any] = {}
        for k, v in r.items():
            if k in {"corruption", "scope"}:
                row[k] = v
            elif k in {"severity", "bin", "n_gt"}:
                row[k] = int(v)
            elif v in {"True", "False"}:
                row[k] = v == "True"
            else:
                row[k] = float(v)
        out.append(row)
    return out


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def severity_series(
    summary: Sequence[dict[str, Any]], scope: str, metric: str, corruptions: Sequence[str]
) -> dict[str, dict[str, list[float]]]:
    """Per corruption: severities 0..5 (0 = shared clean point) with value and CI bounds."""
    clean = next(r for r in summary if r["corruption"] == "clean" and r["scope"] == scope)
    out = {}
    for c in corruptions:
        rows = sorted(
            (r for r in summary if r["corruption"] == c and r["scope"] == scope),
            key=lambda r: r["severity"],
        )
        pts = [clean, *rows]
        out[c] = {
            "severity": [float(r["severity"]) for r in pts],
            "value": [r[metric] for r in pts],
            "low": [r[f"{metric}_lo"] for r in pts],
            "high": [r[f"{metric}_hi"] for r in pts],
        }
    return out


@dataclass(frozen=True)
class Panel:
    """One small-multiple panel: a scope and a metric column, with a title."""

    scope: str
    metric: str
    title: str


def plot_metric_vs_severity(
    summary: Sequence[dict[str, Any]],
    panels: Sequence[Panel],
    corruptions: Sequence[str],
    out_path: str | Path,
    dpi: int,
    title: str,
) -> Path:
    """Small multiples of a metric vs severity (shared y), one line per corruption.

    Args:
        summary: Rows of ``summary.csv``.
        panels: One :class:`Panel` per subplot.
        corruptions: Corruptions in fixed colour order.
        out_path: Output PNG path.
        dpi: Resolution.
        title: Figure title.

    Returns:
        The written path.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    fig.patch.set_facecolor(SURFACE)
    all_ends: list[list[tuple[float, str]]] = []
    top = max(
        max(s["high"] + s["value"])
        for p in panels
        for s in severity_series(summary, p.scope, p.metric, corruptions).values()
    )
    for ax, panel in zip(axes, panels, strict=True):
        _style(ax)
        ends = []
        for i, (c, s) in enumerate(
            severity_series(summary, panel.scope, panel.metric, corruptions).items()
        ):
            color = SERIES_COLORS[i]
            ax.fill_between(s["severity"], s["low"], s["high"], color=color, alpha=0.12, lw=0)
            ax.plot(
                s["severity"],
                s["value"],
                color=color,
                lw=2,
                marker=MARKERS[i],
                ms=6,
                mec=SURFACE,
                mew=1.2,
                label=c.replace("_", " "),
            )
            ends.append((s["value"][-1], c.replace("_", " ")))
        all_ends.append(ends)
        ax.set_title(panel.title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.set_xticks(range(6), ["clean", "1", "2", "3", "4", "5"])
        ax.set_xlim(-0.2, 6.3)
        ax.set_xlabel("corruption severity", color=TEXT_SECONDARY, fontsize=9)
    axes[0].set_ylabel(
        METRIC_TITLES.get(panels[0].metric, panels[0].metric), color=TEXT_SECONDARY, fontsize=9
    )
    axes[0].set_ylim(0, top * 1.05)  # final range first, so label spacing is measured correctly
    for ax, ends in zip(axes, all_ends, strict=True):
        _direct_labels(ax, ends)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11, x=0.01, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return _save(fig, out_path, dpi)


def _direct_labels(ax: plt.Axes, ends: list[tuple[float, str]], min_gap: float = 0.05) -> None:
    """Label line ends at x=5, nudging labels apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    placed: list[float] = []
    for y, _ in sorted(ends):
        y_pos = max(y, placed[-1] + gap) if placed else y
        placed.append(y_pos)
    for (y, name), y_pos in zip(sorted(ends), placed, strict=True):
        ax.annotate(
            name, (5, y), xytext=(5.15, y_pos), color=TEXT_SECONDARY, fontsize=8, va="center"
        )


def reliability_bins(
    reliability: Sequence[dict[str, Any]],
    corruption: str,
    severity: int,
    scope: str,
    threshold: float,
    calibrated: bool = False,
) -> list[dict[str, Any]]:
    """Bins of one reliability diagram, ordered by bin index."""
    rows = [
        r
        for r in reliability
        if r["corruption"] == corruption
        and r["severity"] == severity
        and r["scope"] == scope
        and math.isclose(r["score_threshold"], threshold)
        and bool(r.get("calibrated", False)) == calibrated
    ]
    if not rows:
        raise KeyError(
            f"no reliability rows for {corruption}/{severity}/{scope}/{threshold}/{calibrated}"
        )
    return sorted(rows, key=lambda r: r["bin"])


def plot_reliability(
    reliability: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    conditions: Sequence[tuple[str, int]],
    scope: str,
    threshold: float,
    out_path: str | Path,
    dpi: int,
    rows: Sequence[bool] = (False,),
    temperature: float | None = None,
) -> Path:
    """Reliability diagrams: per-bin precision (bars) against the ideal diagonal.

    One column per condition, one row per calibration state (``False`` = raw scores,
    ``True`` = after temperature scaling). Bars below the diagonal mean overconfidence.
    Each panel is annotated with that row's D-ECE at every reported threshold (95% CI).

    Args:
        reliability: Rows of ``reliability.csv``.
        summary: Rows of ``summary.csv`` (for D-ECE annotations).
        conditions: ``(corruption, severity)`` per column.
        scope: Scope to show.
        threshold: Score threshold of the binned detections.
        out_path: Output PNG path.
        dpi: Resolution.
        rows: Calibration state per row.
        temperature: Fitted T, shown in the row label of calibrated rows.

    Returns:
        The written path.
    """
    n_r, n_c = len(rows), len(conditions)
    fig, grid = plt.subplots(
        n_r, n_c, figsize=(4.0 * n_c, 4.3 * n_r + 0.6), sharex=True, sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ri, calibrated in enumerate(rows):
        for ci, (corruption, severity) in enumerate(conditions):
            ax = grid[ri][ci]
            _style(ax)
            bins = reliability_bins(reliability, corruption, severity, scope, threshold, calibrated)
            width = bins[0]["bin_high"] - bins[0]["bin_low"]
            prec = [0.0 if math.isnan(b["precision"]) else b["precision"] for b in bins]
            color = SERIES_COLORS[0] if not calibrated else "#1c5cab"
            ax.bar(
                [b["bin_low"] for b in bins],
                prec,
                width=width * 0.9,
                align="edge",
                color=color,
                lw=0,
            )
            ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, lw=1.2, ls=(0, (4, 3)))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            if ri == 0:
                name = (
                    "clean"
                    if corruption == "clean"
                    else f"{corruption.replace('_', ' ')} · severity {severity}"
                )
                ax.set_title(name, color=TEXT_PRIMARY, fontsize=11, loc="left")
            if ri == n_r - 1:
                ax.set_xlabel("confidence", color=TEXT_SECONDARY, fontsize=9)
            if ci == 0:
                state = (
                    "raw scores"
                    if not calibrated
                    else (
                        f"after scaling (T = {temperature:.2f})" if temperature else "after scaling"
                    )
                )
                ax.set_ylabel(f"{state}\nprecision (fraction TP)", color=TEXT_SECONDARY, fontsize=9)

            row = next(
                r
                for r in summary
                if r["corruption"] == corruption
                and r["severity"] == severity
                and r["scope"] == scope
            )
            prefix = "dece_cal_t" if calibrated else "dece_t"
            lines = []
            for key in sorted(
                k for k in row if k.startswith(prefix) and not k.endswith(("_lo", "_hi"))
            ):
                t = key.removeprefix(prefix)
                lines.append(
                    f"D-ECE (≥{t}) {row[key]:.3f} [{row[key + '_lo']:.3f}, {row[key + '_hi']:.3f}]"
                )
            lines.append(f"n = {int(sum(b['count'] for b in bins)):,} detections")
            ax.text(
                0.03,
                0.97,
                "\n".join(lines),
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                color=TEXT_PRIMARY,
            )
    fig.suptitle(
        f"Reliability: {SCOPE_TITLES.get(scope, scope)} (bars below the diagonal = overconfident)",
        color=TEXT_PRIMARY,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, out_path, dpi)


def _save(fig: plt.Figure, out_path: str | Path, dpi: int) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def make_figures(
    cfg: dict[str, Any],
    summary_path: Path,
    reliability_path: Path,
    out_dir: Path,
    n_images: int | None = None,
    temperature: float | None = None,
) -> list[Path]:
    """Write all figures configured under ``plots`` from the result CSVs.

    Figure 2 and the calibrated row of Figure 3 are drawn only if the CSVs contain
    after-scaling columns.

    Args:
        cfg: Validated config.
        summary_path: ``summary.csv``.
        reliability_path: ``reliability.csv``.
        out_dir: Figure directory.
        n_images: Evaluation-set size (subtitle only).
        temperature: Fitted T (labels only).

    Returns:
        Paths of the written figures.
    """
    p = cfg["plots"]
    summary, reliability = read_csv(summary_path), read_csv(reliability_path)
    ext, dpi = p["format"], p["dpi"]
    corruptions = cfg["corruptions"]["names"]
    sub = f" · {n_images} COCO val2017 images" if n_images else ""
    metric = p["severity_metric"]
    out = [
        plot_metric_vs_severity(
            summary,
            [Panel(sc, metric, SCOPE_TITLES.get(sc, sc)) for sc in p["severity_scopes"]],
            corruptions,
            out_dir / f"fig1_{metric}_vs_severity.{ext}",
            dpi,
            f"{METRIC_TITLES.get(metric, metric)} under camera corruptions "
            f"(Faster R-CNN v2, shaded: 95% bootstrap CI{sub})",
        )
    ]
    calibrated = any("dece_cal_t" in k for k in summary[0])
    if calibrated:
        before = p["calibration_metric"]
        after = before.replace("dece_t", "dece_cal_t")
        t_txt = f" (T = {temperature:.2f})" if temperature else ""
        scope = p["calibration_scope"]
        out.append(
            plot_metric_vs_severity(
                summary,
                [
                    Panel(scope, before, "Before scaling (raw scores)"),
                    Panel(scope, after, f"After temperature scaling{t_txt}"),
                ],
                corruptions,
                out_dir / f"fig2_{before}_before_after.{ext}",
                dpi,
                f"{METRIC_TITLES.get(before, before)}, {SCOPE_TITLES.get(scope, scope)}: "
                f"T fitted on clean data, applied under corruption (shaded: 95% CI{sub})",
            )
        )
    out.append(
        plot_reliability(
            reliability,
            summary,
            [(c, int(s)) for c, s in p["reliability_conditions"]],
            p["reliability_scope"],
            p["reliability_threshold"],
            out_dir / f"fig3_reliability_{p['reliability_scope']}.{ext}",
            dpi,
            rows=(False, True) if calibrated else (False,),
            temperature=temperature,
        )
    )
    return out
