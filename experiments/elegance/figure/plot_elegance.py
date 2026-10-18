"""Plot the elegance results: judge sanity check, training dynamics, final evaluation.

Inputs beside this script (exported from the elegance dashboard):
- elegance-conclusions.csv: untrained 9B vs 2B elegant share of correct on the
  problems both solve (chart == "senior_vs_junior", group == "all").
- elegance-rollouts.csv: per arm and step, rollouts, correct and elegant counts
  (Tandem steps 11-65 were not saved).
- elegance-final-eval.csv: step-200 Tandem/GRPO elegant share of correct on all
  149 unseen problems, solo and handoff, with bootstrap SEs over problems.
Output: elegance.png beside this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ARMS = {"grpo": ("GRPO", "#ef7c20"), "tandem": ("Tandem", "#7b4fbf")}
WINDOW = 10  # steps pooled for the solid training curve


def main() -> None:
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=200, gridspec_kw={"width_ratios": [1, 2.2, 1.4]})
    senior_vs_junior(axes[0])
    training(axes[1])
    final_eval(axes[2])
    for ax, tag in zip(axes, "abc"):
        ax.set_title(f"({tag})", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Elegant share of correct (%)")
    fig.tight_layout()
    fig.savefig(HERE / "elegance.png")


def senior_vs_junior(ax) -> None:
    df = pd.read_csv(HERE / "elegance-conclusions.csv")
    df = df[(df.chart == "senior_vs_junior") & (df.group == "all")].set_index("series").loc[["base", "junior"]]
    ax.bar([0, 1], 100 * df.value, 0.6, yerr=100 * df.se, capsize=4, color=["#7f7f7f", "#4c9a8a"])
    ax.set_xticks([0, 1], ["9B", "2B"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Untrained Qwen3.5")


def training(ax) -> None:
    df = pd.read_csv(HERE / "elegance-rollouts.csv").sort_values("step")
    for arm, (label, color) in ARMS.items():
        rows = df[df.arm == arm].set_index("step")
        steps = np.arange(1, rows.index.max() + 1)
        rows = rows.reindex(steps)  # unsaved steps become gaps
        pooled = rows.elegant.rolling(WINDOW, min_periods=1).sum() / rows.correct.rolling(WINDOW, min_periods=1).sum()
        pooled[rows.correct.isna()] = np.nan
        ax.plot(steps, 100 * rows.elegant / rows.correct, color=color, alpha=0.2, lw=0.8)
        ax.plot(steps, 100 * pooled, color=color, lw=2, label=label)
    ax.set_xlabel("Training step")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")


def final_eval(ax) -> None:
    df = pd.read_csv(HERE / "elegance-final-eval.csv")
    df = df[df.group == "all"].set_index(["arm", "mode"])
    width = 0.36
    for j, (arm, (label, color)) in enumerate(ARMS.items()):
        rows = df.loc[[(arm, "solo"), (arm, "handoff")]]
        xs = [i + (j - 0.5) * width for i in range(2)]
        ax.bar(xs, 100 * rows.elegant_of_correct, width, yerr=100 * rows.elegant_of_correct_se, capsize=3, color=color, label=label)
    ax.set_xticks([0, 1], ["Solo", "Handoff"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Trained Qwen3.5")


if __name__ == "__main__":
    main()
