#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_HOME:-}" ]]; then
  echo "Set HF_HOME to the local Hugging Face cache; this command never downloads model files." >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
EDITH_PYTHON="${EDITH_PYTHON:-python3.11}"

"${EDITH_PYTHON}" -m pytest tests/edith -q
"${EDITH_PYTHON}" -m tests.edith.validation
