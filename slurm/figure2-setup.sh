#!/usr/bin/env bash
# Run on a CPU allocation; REPO, TANDEM_ENV, HF_HOME and UV_CACHE_DIR are site paths.
set -euo pipefail
cd "${REPO:?set REPO to the git checkout}"
export PATH="$HOME/.local/bin:/data/scratch/$USER/.local/bin:$PATH"
export UV_LINK_MODE=copy
TANDEM_ENV=${TANDEM_ENV:-$REPO/.venv}
uv venv --python 3.11 "$TANDEM_ENV"
uv pip install --python "$TANDEM_ENV/bin/python" -r env/figure2-requirements.txt
uv pip freeze --python "$TANDEM_ENV/bin/python" > logs/figure2-environment.txt
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline eval/download_figure2.py \
    --out results/figure2/models.json
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline \
    -m unittest discover -s tests -p test_benchmark_aggregation.py
