#!/usr/bin/env bash
# Validate and calibrate in an existing allocation; never submits another job.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an existing interactive GPU allocation}"
cd "${REPO:?}"
export UV_BIN_DIR=${UV_BIN_DIR:-$(dirname "$(command -v uv)")}
export APPS_GRADER_RUNTIME=${APPS_GRADER_RUNTIME:-${RUN_ROOT:?}/grader}
export APPS_BWRAP="$APPS_GRADER_RUNTIME/bwrap-libs/ld-linux-x86-64.so.2 --library-path $APPS_GRADER_RUNTIME/bwrap-libs $APPS_GRADER_RUNTIME/bwrap"
export PYTHONPATH="$REPO/tests:$REPO${PYTHONPATH:+:$PYTHONPATH}"
run_python() { uv run --no-project --offline --python "${TANDEM_ENV:?}/bin/python" python "$@"; }
mkdir -p "$RUN_ROOT/calibration" "$RUN_ROOT/checks"
run_python -m unittest test_coding test_figure2 test_benchmark_aggregation \
    test_training_checkpoint test_validation_mode test_senior_gate \
    test_word_state_machine test_eval_resume test_word_handoff > "$RUN_ROOT/final-node-tests.log" 2>&1
uv run --no-project --offline --python "${GRADING_ENV:?}/bin/python" python \
    -m unittest test_math_grading > "$RUN_ROOT/math-regression.log" 2>&1
run_python -m experiments.coding.campaign preflight
BASE_MODEL=$(run_python -c 'import json,os; print(json.load(open(os.environ["RUN_ROOT"]+"/base.json"))["base"]["path"])')
run_python -m experiments.coding.campaign protocol --data "${DATA_ROOT:?}" --base "$RUN_ROOT/base.json" --out "$RUN_ROOT/protocol.json"
if [ ! -s "$RUN_ROOT/calibration/solo.json" ]; then
    run_python -m experiments.coding.campaign evaluate --phase solo --model "$BASE_MODEL" --junior "$BASE_MODEL" \
        --data "$DATA_ROOT" --split validation --out "$RUN_ROOT/calibration/solo.json" --decoding training --n 8 \
        > "$RUN_ROOT/calibration/solo-resumed.log" 2>&1
fi
run_python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
baseline = root / 'checks/solo-restart-baseline.json'
if baseline.exists():
    old = json.loads(baseline.read_text())['gens']
    new = json.loads((root / 'calibration/solo.json').read_text())['gens']
    assert len(new) == 200 and new[:len(old)] == old
    (root / 'checks/restart.json').write_text(json.dumps({
        'passed': True, 'preserved_tasks': len(old), 'completed_tasks': len(new)}) + '\n')
PY
run_python -m experiments.coding.campaign evaluate --phase word-handoff --model "$BASE_MODEL" --junior "$BASE_MODEL" \
    --data "$DATA_ROOT" --split validation --out "$RUN_ROOT/calibration/word-handoff.json" --decoding training --n 8 \
    --native-concurrency "${NATIVE_CONCURRENCY:-2}" > "$RUN_ROOT/calibration/word-handoff.log" 2>&1
run_python -m experiments.coding.campaign calibration --data "$DATA_ROOT" --solo "$RUN_ROOT/calibration/solo.json" \
    --word "$RUN_ROOT/calibration/word-handoff.json" --out "$RUN_ROOT/calibration/report.json"
# A failed gate exits above, before either smoke or full training.
for arm in grpo tandem; do
    if [ -s "$RUN_ROOT/$arm-smoke/selected.json" ]; then
        run_python train/figure2_checkpoint.py --root "$RUN_ROOT/$arm-smoke" --arm "$arm" --steps 3 \
            --mode smoke --validation-metric apps/validation/pass@4
    else
        ARM="$arm" MODE=smoke bash experiments/coding/job.sbatch train > "$RUN_ROOT/$arm-smoke.log" 2>&1
    fi
done
echo 'Calibration and both interactive training smokes passed; production resources can now be selected from measured fit and current queues.'
