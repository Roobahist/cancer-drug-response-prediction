"""Editorial figures for held-out predictions and historical reported scores.

All numerical labels are loaded from machine-readable CSV/JSON, never invented.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK = "#1e2d3c"
MUTED = "#627386"
TEAL = "#246d7e"
ORANGE = "#c27c53"
PURPLE = "#77709c"
GRID = "#e6eaf0"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        figure.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight",
                       pad_inches=0.3)
    plt.close(figure)


def plot_original_reported_results(
    input_path: str | Path, output: str | Path
) -> None:
    """Compare metrics TRANSCRIBED from the team's README, not reproduced runs."""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if data.get("status") != "reported_not_reproduced":
        raise ValueError("Reported-results status is missing or inconsistent")
    rows = data["models"]
    names = [r["model"] for r in rows]
    mse = [float(r["mse"]) for r in rows]
    r2 = [float(r["r2"]) for r in rows]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 5.5), layout="constrained")
    colors = [TEAL, "#819aaa", "#a0adbb", "#bcc4cc"]
    for ax, vals, title, note, fmt, lim in [
        (axes[0], mse, "Mean squared error", "Lower is better", "{:.3f}", 1.65),
        (axes[1], r2, "R² score", "Higher is better", "{:.3f}", 1.06),
    ]:
        ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.56)
        for i, v in enumerate(vals[::-1]):
            ax.text(v + lim * 0.015, i, fmt.format(v), va="center", ha="left",
                    fontsize=11, weight="bold", color=INK)
        ax.set_xlim(0, lim)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title, loc="left", pad=17)
        ax.set_xlabel(note)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Original project: published model scores", x=0.07, y=1.07,
                 ha="left", fontsize=22, weight="bold", color=INK)
    fig.text(0.07, -0.08,
             "Values transcribed from the original group's README. Not independently "
             "reproduced; data splits and uncertainty are unverified.",
             ha="left", color=MUTED, fontsize=10.5)
    _save(fig, Path(output))


def plot_holdout_predictions(output_dir: str | Path) -> None:
    """Plot truth-versus-prediction for a newly run held-out evaluation."""
    directory = Path(output_dir)
    predictions = pd.read_csv(directory / "holdout_predictions.csv")
    metrics = pd.read_csv(directory / "model_metrics.csv")
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol["evaluation_status"] == "synthetic demonstration":
        status = "SYNTHETIC DEMO — NOT CLINICAL PERFORMANCE"
    else:
        status = "GROUPED HOLDOUT — NOT EXTERNALLY VALIDATED"
    names = list(metrics["model"])
    _style()
    fig, axes = plt.subplots(1, len(names), figsize=(15.2, 5.0),
                             layout="constrained", sharex=True, sharey=True)
    lo = min(predictions["actual"].min(), predictions["predicted"].min())
    hi = max(predictions["actual"].max(), predictions["predicted"].max())
    for i, (ax, name) in enumerate(zip(axes, names, strict=True)):
        frame = predictions[predictions["model"] == name]
        ax.scatter(frame["actual"], frame["predicted"], s=25,
                   color=[TEAL, ORANGE, PURPLE][i], alpha=0.65, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "--", color="#8fa1af", linewidth=1.2)
        ax.set_title(name, loc="left")
        ax.set_xlabel("Actual LN_IC50")
        ax.grid(alpha=0.5, color=GRID)
        ax.set_axisbelow(True)
        score = float(metrics.loc[metrics["model"] == name, "test_r2"].iloc[0])
        ax.text(0.05, 0.95, f"Test R² = {score:.3f}", transform=ax.transAxes,
                va="top", weight="bold", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted LN_IC50")
    fig.suptitle("Holdout predictions by model", x=0.05, y=1.09,
                 ha="left", fontsize=20, weight="bold")
    fig.text(0.05, -0.07, status, ha="left", fontsize=10.5, weight="bold",
             color=MUTED)
    _save(fig, directory / "predicted_vs_actual")
