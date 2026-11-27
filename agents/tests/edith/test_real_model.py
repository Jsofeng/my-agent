"""Optional offline coverage for the locally cached production embedding model."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_minilm_groups_a_paraphrase_more_than_unrelated_text():
    cache = os.environ.get("HF_HOME")
    if not cache:
        pytest.skip("Set HF_HOME to a local MiniLM model cache; tests never download it.")
    model_cache = Path(cache) / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not model_cache.exists():
        pytest.skip("MiniLM is not present in the configured local cache.")
    env = os.environ.copy()
    env.update({"HF_HOME": cache, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    script = (
        "import numpy as np; "
        "from edith.embeddings.sentence_transformer import SentenceTransformerEmbedder; "
        "v=SentenceTransformerEmbedder().embed(['I enjoy quiet workspaces.', "
        "'I prefer a calm place to focus.', 'The fictional train departed after lunch.']); "
        "print(float(np.dot(v[0],v[1])) > float(np.dot(v[0],v[2])))"
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, text=True,
                            capture_output=True, check=True)
    assert result.stdout.strip() == "True"
