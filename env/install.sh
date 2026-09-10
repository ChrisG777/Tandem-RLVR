#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-}"
if [ -z "$PREFIX" ]; then
    echo "usage: bash env/install.sh <conda-env-prefix>" >&2
    echo "  e.g. bash env/install.sh \$HOME/envs/tandem-rlvr" >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" >&2
    exit 1
fi

conda create -p "$PREFIX" python=3.11 -y

PY="$PREFIX/bin/python"

"$PY" -m pip install --upgrade pip setuptools wheel

"$PY" -m pip install "vllm==0.19.1"

"$PY" -m pip install "transformers==5.5.4"

"$PY" - <<'EOF'
import torch
import transformers
import vllm

print("torch", torch.__version__,
      "| vllm", vllm.__version__,
      "| transformers", transformers.__version__)
EOF

cat <<EOF

Core stack installed at $PREFIX.
Next: docs/INSTALL.md, which builds both forks into this environment.
EOF
