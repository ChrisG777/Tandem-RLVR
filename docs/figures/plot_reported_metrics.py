#!/usr/bin/env python3
"""Plot documented aggregate values without inventing curves or uncertainty.

Dependencies: matplotlib (rendered with 3.7.1), LaTeX and dvipng on PATH.
From the repository root:
    python3 docs/figures/plot_reported_metrics.py

The JSON beside this script records units, conditions and source documents.
Only the standard library and Matplotlib are imported. Outputs are PNG and SVG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


ARM_COLORS = {"grpo": "#ee771c", "tandem": "#7b4cb8"}
PILOT_COLORS = {"base": "#929aa7", "no_penalty": "#3775b9", "length_penalty": "#c65252"}


def style_axes(ax, *, percentage: bool = True) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce1e7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9ba3ae")
    ax.tick_params(axis="both", pad=7)
    if percentage:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"{value:.0f}\%"))


def save(fig, out: Path) -> None:
    fig.savefig(out.with_suffix(".png"), dpi=180, facecolor="white")
    # Stable SVG IDs and no timestamp make rerenders reproducible in one runtime.
    svg = out.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_qwen(data: dict, out: Path) -> None:
    study = data["qwen3_4b"]
    fig, (accuracy_ax, drift_ax) = plt.subplots(
        1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.88, wspace=0.34)

    metrics = study["accuracy"]["metrics"]
    width = 0.34
    for arm, offset, label in (("grpo", -width / 2, "GRPO"), ("tandem", width / 2, "Tandem")):
        bars = accuracy_ax.bar(
            [i + offset for i in range(len(metrics))],
            [metric[arm] for metric in metrics], width=width,
            color=ARM_COLORS[arm], label=label, zorder=3,
        )
        accuracy_ax.bar_label(bars, labels=[f"{metric[arm]:.2f}" for metric in metrics], padding=5, fontsize=15)
    style_axes(accuracy_ax)
    accuracy_ax.set_title("Benchmark accuracy", fontsize=18, pad=14)
    accuracy_ax.set_ylabel(r"Macro pass@$k$")
    accuracy_ax.set_xticks(range(len(metrics)), [metric["label"] for metric in metrics])
    accuracy_ax.legend(loc="upper left", frameon=False, ncol=2)

    drift = study["drift"]
    values = [drift[arm] for arm in ("grpo", "tandem")]
    bars = drift_ax.bar([0, 1], values, width=0.56, color=list(ARM_COLORS.values()), zorder=3)
    drift_ax.bar_label(bars, labels=[f"{value:.8f}" for value in values], padding=6, fontsize=15)
    style_axes(drift_ax, percentage=False)
    drift_ax.set_ylim(0, 0.0045)
    drift_ax.yaxis.set_major_locator(MultipleLocator(0.001))
    drift_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
    drift_ax.set_xticks([0, 1], ["GRPO", "Tandem"])
    drift_ax.set_ylabel("JSD from base (nats)")
    drift_ax.set_title("Marginal token drift", fontsize=18, pad=14)
    drift_ax.text(0.5, 0.97, rf"{drift['reported_reduction_percent']:.2f}\% reduction",
                  transform=drift_ax.transAxes, ha="center", va="top", fontsize=15, color=ARM_COLORS["tandem"])
    save(fig, out)


def plot_pilots(data: dict, out: Path) -> None:
    study = data["pilots"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.88, wspace=0.25)

    keys = ("base", "no_penalty", "length_penalty")
    labels = ("Base", "Solo RL\nno penalty", "Solo RL\n+ token cost")
    for ax, task in zip(axes, study["tasks"]):
        values = [task["values"][key] for key in keys]
        bars = ax.bar(range(3), values, width=0.6, color=[PILOT_COLORS[key] for key in keys], zorder=3)
        ax.bar_label(bars, labels=[rf"{value:.2f}\%" for value in values], padding=6, fontsize=15)
        style_axes(ax)
        ax.set_xticks(range(3), labels)
        budget = task["evaluation_response_ceiling"]
        name = "Matrix" if task["id"] == "matrix" else "RuleTaker"
        ax.set_title(f"{name} ({budget:,}-token evaluation)", fontsize=18, pad=14)
    axes[0].set_ylabel("Exact-match accuracy")
    axes[1].set_ylabel("All-four accuracy")
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("reported-metrics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    if data["schema_version"] != 1:
        raise ValueError("Unsupported reported-metrics schema")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 18,
        "xtick.top": False,
        "ytick.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1,
        "svg.fonttype": "none", "svg.hashsalt": "tandem-reported-metrics-v1",
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_qwen(data, args.out_dir / "qwen3-4b-summary")
    plot_pilots(data, args.out_dir / "pilot-summary")
    print(f"Wrote qwen3-4b-summary and pilot-summary PNG/SVG files to {args.out_dir}")


if __name__ == "__main__":
    main()
