"""Plot a timestamped training-log snapshot; distinguish monitoring from evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from figure2_checkpoint import VALIDATION, merge_training_records

COLORS = {"grpo": "#ee771c", "tandem": "#7b4cb8"}
LABELS = {"grpo": "Solo GRPO", "tandem": "Tandem RLVR"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--root", type=Path, help="live campaign containing grpo-full and tandem-full")
    parser.add_argument("--out", type=Path, required=True, help="PNG/SVG/JSON filename stem")
    args = parser.parse_args()
    if args.root is not None:
        args.snapshot = args.out.with_suffix(".snapshot.json")
        capture_training_snapshot(args.root, args.snapshot)
    print(json.dumps(plot_training_progress(args.snapshot, args.out), indent=2))


def plot_training_progress(snapshot: Path, out: Path) -> dict:
    """Write reward, validation, length and truncation curves plus health statistics.

    Inputs contain chronological per-attempt rows and capture time. Missing
    metrics remain NaN, with no interpolation; a 10-step mean requires all ten
    observations. Output is a monitoring snapshot, not final benchmark results.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    payload = json.loads(snapshot.read_text())
    arms = {arm: merge_training_records([a["rows"] for a in attempts])
            for arm, attempts in payload["arms"].items()}
    captured = datetime.fromisoformat(payload["captured_at"]).astimezone(ZoneInfo("America/Los_Angeles"))
    summary = {"captured_at": payload["captured_at"], "arms": {}}
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    fig.subplots_adjust(top=0.84, bottom=0.21, hspace=0.46, wspace=0.28)
    fig.suptitle("Figure 2 reproduction — training progress", fontsize=17, x=0.075, ha="left")
    fig.text(0.075, 0.915, f"Snapshot: {captured:%b %d, %Y · %I:%M %p %Z}  |  Target: 200 optimizer steps", color="#555555")
    panels = ((axes[0, 0], "critic/score/mean", "Training reward", "Correct rollouts (%)", 100),
              (axes[1, 0], "response_length/mean", "Response length", "Generated tokens", 1),
              (axes[1, 1], "response_length/clip_ratio", "Responses reaching the token limit", "Share of rollouts (%)", 100))
    for arm, records in arms.items():
        last = max(records)
        x = np.arange(1, last + 1)
        color = COLORS[arm]
        for ax, metric, title, ylabel, scale in panels:
            y = np.array([records.get(int(step), {}).get(metric, np.nan) for step in x], dtype=float) * scale
            smooth = np.full(len(y), np.nan)
            for end in range(9, len(y)):
                smooth[end] = np.mean(y[end - 9:end + 1])
            ax.plot(x, y, color=color, alpha=0.25, lw=1)
            ax.plot(x, smooth, color=color, lw=2.3, label=LABELS[arm])
            ax.set_title(title, loc="left", fontsize=12)
            ax.set_ylabel(ylabel)
        validation = [(step, row[VALIDATION] * 100) for step, row in sorted(records.items()) if VALIDATION in row]
        if validation:
            vx, vy = zip(*validation)
            axes[0, 1].plot(vx, vy, marker="o", ms=5, lw=1.6, color=color, label=LABELS[arm])
            axes[0, 1].annotate(f"{vy[-1]:.1f}%", (vx[-1], vy[-1]), xytext=(7, 5 if arm == "grpo" else -14),
                                textcoords="offset points", color=color)
        steps = sorted(records)
        def mean_metric(selected, metric):
            values = [records[s][metric] for s in selected if metric in records[s]]
            return float(np.mean(values)) if values else None
        gradients = [r["actor/grad_norm"] for r in records.values() if "actor/grad_norm" in r]
        summary["arms"][arm] = {"last_logged_step": last, "logged_steps": len(records),
            "missing_steps": sorted(set(range(1, last + 1)) - records.keys()),
            "first10_reward": mean_metric(steps[:10], "critic/score/mean"),
            "last10_reward": mean_metric(steps[-10:], "critic/score/mean"),
            "last10_mean_length": mean_metric(steps[-10:], "response_length/mean"),
            "last10_at_limit": mean_metric(steps[-10:], "response_length/clip_ratio"),
            "gradient_range": [min(gradients), max(gradients)] if gradients else None,
            "nonfinite_observed_metrics": [(s, k) for s, row in records.items() for k, v in row.items()
                                             if isinstance(v, (int, float)) and not np.isfinite(v)],
            "validation": validation,
            "senior_token_fraction": mean_metric(steps[-10:], "actor/tandem_senior_token_frac")}
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=10)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 1].set_title("Held-out DeepScaleR accuracy", loc="left", fontsize=12)
    axes[0, 1].set_ylabel("Pass@4 (%)")
    axes[0, 1].set_ylim(50, 75)
    axes[1, 0].axhline(3000, color="#999999", lw=1, ls=":")
    axes[1, 0].set_ylim(0, 3200)
    axes[1, 1].set_ylim(0, 100)
    xmax = max(max(records) for records in arms.values()) + 7
    for ax in axes.flat:
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Optimizer step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    fig.text(0.075, 0.075,
             "Faint lines: per-step values. Thick lines: trailing 10-step means. Missing metrics remain gaps, without interpolation.\n"
             "Reward = binary correctness on sampled training batches. Validation uses solo GRPO / tandem team rollouts;\n"
             "these are monitoring curves, not a matched solo-capability comparison or the final Figure 2 benchmark.",
             fontsize=10, color="#555555", va="center", linespacing=1.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def capture_training_snapshot(root: Path, out: Path) -> None:
    """Atomically save complete JSONL records from live runs; omit an unfinished last line.

    This is an explicitly timestamped monitoring snapshot, never a replacement
    for source logs or a claim that the training campaign has finished.
    """
    payload = {"captured_at": datetime.now(timezone.utc).isoformat(), "root": str(root), "arms": {}}
    for arm in COLORS:
        attempts = []
        for path in sorted((root / f"{arm}-full").glob("metrics-*.jsonl"),
                           key=lambda p: (p.stat().st_mtime_ns, tuple(map(int, re.findall(r"\d+", p.stem))))):
            raw = path.read_bytes()
            complete = raw if raw.endswith(b"\n") else raw.rsplit(b"\n", 1)[0] if b"\n" in raw else b""
            attempts.append({"file": path.name,
                             "rows": [json.loads(line) for line in complete.splitlines() if line.strip()],
                             "omitted_incomplete_line": bool(raw and not raw.endswith(b"\n"))})
        if not attempts:
            raise ValueError(f"No logs for {arm}")
        payload["arms"][arm] = attempts
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".partial")
    temporary.write_text(json.dumps(payload) + "\n")
    temporary.replace(out)


if __name__ == "__main__":
    main()
