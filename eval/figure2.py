"""Plot Figure 2 from five or six evaluation JSONs, with bootstrap standard errors."""
import argparse
from pathlib import Path

import numpy as np

import common
import config

BENCHMARKS = ("amc23_25", "aime24_26", "minerva", "olympiad")
TITLES = ("AMC", "AIME", "Minerva", "Olympiad", "Macro avg")
COLORS = {"base": "#8d8d8d", "grpo": "#ee771c", "tandem": "#7b4cb8"}
LABELS = {"base": "Base", "grpo": "GRPO", "tandem": "Tandem"}
EXPECTED_SIZES = {"amc23_25": 121, "aime24_26": 90, "minerva": 272, "olympiad": 581}


EXPECTED_IDS = None

def configure_panel(panel: dict) -> None:
    """Set explicit ordered benchmark sizes/titles and complete expected task IDs."""
    global BENCHMARKS, TITLES, EXPECTED_SIZES, EXPECTED_IDS
    BENCHMARKS = tuple(panel["benchmarks"])
    TITLES = tuple(panel["benchmarks"][n]["title"] for n in BENCHMARKS) + ("Macro avg",)
    EXPECTED_SIZES = {n: len(panel["benchmarks"][n]["task_ids"]) for n in BENCHMARKS}
    EXPECTED_IDS = {(n, task) for n in BENCHMARKS for task in panel["benchmarks"][n]["task_ids"]}
    if not BENCHMARKS or len(EXPECTED_IDS) != sum(EXPECTED_SIZES.values()):
        raise ValueError("Empty panel or duplicate task IDs")


def main() -> None:
    """Read --results/{base,grpo,tandem}/phase.json; write PNG, SVG and score JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-baselines", action="store_true", help="Include distinct senior/junior solo baselines")
    parser.add_argument("--panel-manifest", type=Path, help="JSON with benchmark sizes, titles and expected task IDs")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="output filename stem")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    if args.panel_manifest:
        configure_panel(common.load_json(args.panel_manifest))
    if args.paired_baselines:
        COLORS["junior"] = "#2798a5"
        LABELS.update(base="Initial senior", junior="Frozen junior")
    curves = {}
    protocol_identity = None
    for phase, arms in (("solo", COLORS), ("handoff", ("base", "grpo", "tandem") if args.paired_baselines else ("grpo", "tandem"))):
        curves[phase] = {}
        for arm in arms:
            result = common.load_json(args.results / arm / f"{phase}.json")
            if args.panel_manifest:
                identity = {key: result[key] for key in ("dataset_sha256", "grader", "context", "max_prompt")}
                if protocol_identity is not None and identity != protocol_identity:
                    raise ValueError("Coding comparison uses different datasets, graders or budgets")
                protocol_identity = identity
            if result["phase"] != phase or result["sampling"] != config.record():
                raise ValueError(f"{arm}/{phase}: phase or decoding differs from the protocol")
            curves[phase][arm] = summarize(result, args.bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    common.save_json(args.out.with_suffix(".json"), {
        "protocol": "independently trained validation-selected checkpoints; problem bootstrap within benchmark; bands = 1 SE",
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed,
        "curves": curves,
    })
    plot(curves, args.out)


def summarize(result: dict, bootstrap: int = 2000, seed: int = 20261004) -> dict:
    """Return unbiased pass@k means and bootstrap SEs, in percent; reject partial runs."""
    n = result["n"]
    ks = (1, 2, 4, 8, 16, 32) if result["phase"] == "solo" else (1, 2, 4, 8)
    if n < max(ks) or result.get("limit", 0) or result.get("unfinished_chains", 0):
        raise ValueError("Figure 2 requires complete runs, n>=32 solo, n>=8 handoff")
    per_set = {}
    actual_ids = set()
    seen = set()
    for row in result["gens"]:
        key = (row["set"], row["idx"])
        if key in seen or len(row["correct"]) != n:
            raise ValueError(f"Duplicate problem or wrong sample count: {key}")
        if any(value not in (0, 1) for value in row["correct"]):
            raise ValueError(f"Nonbinary grade: {key}")
        seen.add(key)
        actual_ids.add((row["set"], row.get("task_id")))
        per_set.setdefault(row["set"], []).append(sum(row["correct"]))
    grouped = common.group_by_benchmark(per_set)
    if {name: len(cs) for name, cs in grouped.items()} != EXPECTED_SIZES:
        raise ValueError(f"Expected complete benchmark panel: {EXPECTED_SIZES}")
    if EXPECTED_IDS is not None and actual_ids != EXPECTED_IDS:
        raise ValueError("Evaluation task IDs differ from the panel manifest")
    rng = np.random.default_rng(seed)
    output, samples = {}, []
    for name in BENCHMARKS:
        values = 100 * np.array([[common.pass_at_k(n, int(c), k) for k in ks]
                                 for c in grouped[name]])
        draws = rng.integers(0, len(values), size=(bootstrap, len(values)))
        boot = values[draws].mean(axis=1)
        samples.append(boot)
        output[name] = {"k": list(ks), "mean": values.mean(axis=0).tolist(),
                        "se": boot.std(axis=0, ddof=1).tolist()}
    output["macro"] = {
        "k": list(ks),
        "mean": np.mean([output[name]["mean"] for name in BENCHMARKS], axis=0).tolist(),
        "se": np.mean(samples, axis=0).std(axis=0, ddof=1).tolist(),
    }
    return output


def plot(curves: dict, out: Path, *, note: str | None = None) -> None:
    """Write PNG/SVG curves: handoff k=1..8 and solo k=1..32, powers of two.

    Solo curves appear only in the bottom row. Label partial inputs explicitly
    with note. All means/SEs are percentages; no scores are imputed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, len(BENCHMARKS)+1, figsize=(2.4*(len(BENCHMARKS)+1), 5.4), layout="constrained", squeeze=False)
    for col, (name, title) in enumerate(zip((*BENCHMARKS, "macro"), TITLES)):
        axes[0, col].set_title(title, weight="bold")
        for row, phase in enumerate(("handoff", "solo")):
            ax = axes[row, col]
            for arm, panels in curves[phase].items():
                curve = panels[name]
                x, mean, se = (np.asarray(curve[key]) for key in ("k", "mean", "se"))
                ax.plot(x, mean, color=COLORS[arm], marker="o", lw=2, ms=4)
                ax.fill_between(x, np.maximum(0, mean-se), np.minimum(100, mean+se),
                                color=COLORS[arm], alpha=0.12, linewidth=0)
            ax.set_xscale("log", base=2)
            ticks = [1, 2, 4, 8] if row == 0 else [1, 2, 4, 8, 16, 32]
            ax.set_xticks(ticks, labels=[str(k) for k in ticks])
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{phase.capitalize()} pass@k (%)")
            ax.set_xlabel("k")
    handles = [Line2D([], [], color=c, marker="o", label=LABELS[arm])
               for arm, c in COLORS.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(COLORS), frameon=False)
    if note:
        fig.supxlabel(note, fontsize=9)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
