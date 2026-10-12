"""Summarize MedCalc solo, handoff, and junior-legibility evaluations."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

ARMS = ("base", "grpo", "tandem")
TRAINED = ("grpo", "tandem")


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _macro_pass(result: dict, k: int):
    metrics = result.get("metrics", {}).get("macro", {})
    key = f"pass@{k}"
    if key not in metrics:
        return None
    return float(metrics[key])


def _retention(solo: dict | None, handoff: dict | None, k: int):
    if solo is None or handoff is None:
        return None
    solo_score = _macro_pass(solo, k)
    handoff_score = _macro_pass(handoff, k)
    if solo_score is None or handoff_score is None or solo_score == 0:
        return None
    return handoff_score / solo_score


def collect(results: Path) -> dict:
    summary = {"arms": {}}
    for arm in ARMS:
        directory = results / arm
        solo = _load(directory / "solo.json")
        handoff = _load(directory / "handoff.json")
        legibility = _load(directory / "legibility.json")
        if solo is None and handoff is None and legibility is None:
            continue
        entry = {}
        if solo is not None:
            entry["solo"] = {
                "n": solo.get("n"),
                "limit": solo.get("limit", 0),
                "macro": solo.get("metrics", {}).get("macro", {}),
                "by_benchmark": solo.get("metrics", {}).get("by_benchmark", {}),
                "coverage": solo.get("coverage"),
            }
        if handoff is not None:
            entry["handoff"] = {
                "n": handoff.get("n"),
                "limit": handoff.get("limit", 0),
                "macro": handoff.get("metrics", {}).get("macro", {}),
                "by_benchmark": handoff.get("metrics", {}).get("by_benchmark", {}),
                "coverage": handoff.get("coverage"),
                "retention_of_solo_pass@4": _retention(solo, handoff, 4),
            }
        if legibility is not None:
            entry["legibility"] = {
                "macro_ce": legibility.get("macro_ce"),
                "mean_ce": legibility.get("mean_ce"),
                "n_scored": legibility.get("n_scored"),
                "n_dropped": legibility.get("n_dropped"),
                "by_benchmark": legibility.get("by_benchmark"),
            }
        summary["arms"][arm] = entry
    return summary


def _format_pass(entry: dict, phase: str, k: int) -> str:
    phase_entry = entry.get(phase)
    if not phase_entry:
        return "   —  "
    value = phase_entry["macro"].get(f"pass@{k}")
    if value is None:
        return "   —  "
    return f"{100 * value:6.2f}"


def render_table(summary: dict) -> str:
    lines = [
        "Macro pass@k is the unweighted mean across MedCalc categories.",
        "Retention is handoff pass@4 divided by the same arm's solo pass@4.",
        "",
        f"{'arm':8} {'solo@1':>8} {'solo@4':>8} {'hand@4':>8} {'retain':>8} {'junior CE':>10}",
    ]
    for arm, entry in summary["arms"].items():
        retention = (entry.get("handoff") or {}).get("retention_of_solo_pass@4")
        ce = (entry.get("legibility") or {}).get("macro_ce")
        lines.append(
            f"{arm:8} {_format_pass(entry, 'solo', 1):>8} {_format_pass(entry, 'solo', 4):>8} "
            f"{_format_pass(entry, 'handoff', 4):>8} "
            f"{(f'{100 * retention:6.1f}%' if retention is not None else '   —  '):>8} "
            f"{(f'{ce:10.4f}' if ce is not None else '         —'):>10}"
        )
    if not summary["arms"]:
        lines.append("(no evaluation JSON found)")
    return "\n".join(lines)


def try_plot(results: Path, panel: Path, out: Path) -> str | None:
    """Write the Figure 2 style curves when every required file is a complete run.

    figure2 pulls NumPy and Matplotlib. Keep that import here so the numeric
    table still prints in an environment that only has the JSON results.
    """
    try:
        import figure2
    except ImportError as error:
        return f"plot skipped: {error}"
    if not panel.exists():
        return "plot skipped: panel.json is missing"
    needed = [results / "base" / "solo.json"]
    needed += [results / arm / phase for arm in TRAINED for phase in ("solo.json", "handoff.json")]
    if any(not path.exists() for path in needed):
        return "plot skipped: need base/grpo/tandem solo JSON and grpo/tandem handoff JSON"
    figure2.configure_panel(json.loads(panel.read_text()))
    try:
        curves = {}
        for phase, arms in (("solo", figure2.COLORS), ("handoff", TRAINED)):
            curves[phase] = {}
            for arm in arms:
                result = json.loads((results / arm / f"{phase}.json").read_text())
                curves[phase][arm] = figure2.summarize(result)
    except ValueError as error:
        return f"plot skipped: {error}"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure2.plot(curves, out)
    return f"wrote {out.with_suffix('.png')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "medcalc")
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "medcalc" / "panel.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "medcalc" / "summary")
    args = parser.parse_args()
    summary = collect(args.results)
    text = render_table(summary)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"table": text, **summary}
    note = try_plot(args.results, args.panel, args.out)
    if note:
        payload["plot"] = note
        print(note)
    temporary = args.out.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out.with_suffix(".json"))


if __name__ == "__main__":
    main()
