#!/usr/bin/env bash
# Run on a CPU allocation; REPO, TANDEM_ENV, HF_HOME and UV_CACHE_DIR are site paths.
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?set REPO to the git checkout}"
export PATH="${HOME:-/tmp}/.local/bin:/data/scratch/${SLURM_JOB_USER:-${USER:-}}/.local/bin:${PATH:-/usr/bin:/bin}"
# uv's default links cache entries when the environment shares its filesystem.
# For an environment on another filesystem, pass UV_LINK_MODE=copy explicitly.
TANDEM_ENV=${TANDEM_ENV:-$REPO/.venv}
[ -x "$TANDEM_ENV/bin/python" ] || uv venv --python 3.11 "$TANDEM_ENV"
# The shared cache's old ANTLR sdist is missing bin/pygrun. Fetch this tiny
# pinned package afresh without modifying a cache used by other projects.
uv pip install --python "$TANDEM_ENV/bin/python" --no-cache antlr4-python3-runtime==4.9.3
uv pip install --python "$TANDEM_ENV/bin/python" -r env/figure2-requirements.txt
uv pip freeze --python "$TANDEM_ENV/bin/python" > logs/figure2-environment.txt
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python -c \
    'import torch, vllm; from vllm import LLM; from transformers import AutoTokenizer; print("torch", torch.__version__, "vllm", vllm.__version__)'
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline eval/download_figure2.py \
    --out results/figure2/models.json
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline \
    -m unittest discover -s tests -p test_benchmark_aggregation.py
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline \
    -m unittest discover -s tests -p test_figure2.py
