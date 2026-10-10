"""Submit one dependency-gated APPS campaign with explicit site resource routes."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main() -> None:
    """Submit once; successful calibration and smokes are required for full runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-args", required=True)
    parser.add_argument("--eval-args", required=True)
    parser.add_argument("--train-args", required=True, help="One GPU added per arm; include explicit CPUs/RAM/time/route")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["RUN_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    record = root / "jobs.json"
    # Exclusive creation prevents accidental duplicate campaigns, including partial submission.
    with record.open("x") as f:
        json.dump({}, f)
    jobs = {}
    def submit(name, phase, resources, deps=(), **environment):
        command = ["sbatch", "--parsable", "--job-name=apps-"+name,
                   "--output="+str(root / (name+"-%j.out")), "--kill-on-invalid-dep=yes",
                   *shlex.split(resources)]
        if deps:
            command += ["--dependency=afterok:"+":".join(deps)]
        env = {**os.environ, **environment}
        command += [str(Path(os.environ["REPO"]) / "experiments/coding/job.sbatch"), phase]
        job = subprocess.check_output(command, env=env, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Invalid Slurm response: {job}")
        jobs[name] = job
        tmp = record.with_suffix(".partial")
        tmp.write_text(json.dumps(jobs, indent=2)+"\n")
        tmp.replace(record)
        return job
    setup = submit("setup", "setup", args.cpu_args)
    calibration = [submit("cal-"+phase, "calibration", args.eval_args, [setup], EVAL_PHASE=phase)
                   for phase in ("solo", "word-handoff")]
    gate = submit("gate", "gate", args.cpu_args, calibration)
    if args.calibration_only:
        print(json.dumps(jobs, indent=2))
        return
    full = []
    for arm in ("grpo", "tandem"):
        resources = args.train_args + " --gpus=1"
        smoke = submit(arm+"-smoke", "train", resources, [gate], ARM=arm, MODE="smoke")
        full.append(submit(arm+"-full", "train", resources, [smoke], ARM=arm, MODE="full"))
    selected = submit("select", "select", args.cpu_args, full)
    plot_deps = []
    for arm in ("base", "grpo", "tandem"):
        solo = submit(arm+"-solo", "evaluate", args.eval_args, [selected], ARM=arm, EVAL_PHASE="solo")
        plot_deps.append(solo)
        submit(arm+"-ce", "evaluate", args.eval_args, [solo], ARM=arm, EVAL_PHASE="legibility")
        if arm != "base":
            plot_deps.append(submit(arm+"-handoff", "evaluate", args.eval_args, [selected], ARM=arm, EVAL_PHASE="handoff"))
            submit(arm+"-word", "evaluate", args.eval_args, [selected], ARM=arm, EVAL_PHASE="word-handoff")
    submit("plot", "plot", args.cpu_args, plot_deps)
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
