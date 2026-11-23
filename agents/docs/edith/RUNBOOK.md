# Edith runbook

## Local data

The memory database is `data/edith.db`; the derived vector index will be `data/edith.faiss`. Keep both paths local and protect them as personal data. Do not include them in logs or issue reports.

## Hard deletion

Edith enables SQLite secure deletion and checkpoints the WAL after a hard delete. Old filesystem backups, copied databases, and snapshots may still contain deleted memory. Generate exports and new backups from the current database only. Encryption at rest is future work.

## Dependency setup

`requirements-edith.txt` pins Edith's new dependencies. It is not a complete dependency list for Jarvis because no project-wide manifest existed at the time this file was added.

Set `EDITH_ENABLED=true` to enable memory integration. Background extraction uses the configured Groq-backed smolagents model after a conversation is idle, reaches the configured batch size, ends, or during shutdown flush. Tests use only fake model clients and do not call Groq. If retries are exhausted, unextracted turns remain in SQLite and startup catch-up retries them on the next launch.

## Enable and launch

Use Python 3.10 or newer. `requirements-edith.txt` contains only the pinned Edith dependencies; install it into the same environment as Jarvis and its existing app dependencies. Set `EDITH_ENABLED=true` in the runtime environment. The database and index default to `data/edith.db` and `data/edith.faiss`, and can be moved with `EDITH_DATABASE_PATH` and `EDITH_INDEX_PATH`. Edith loads the MiniLM model on the first embedding call, not while constructing the adapter. Keep the database private because it contains conversation and memory content.

When Edith initializes, the app starts the extraction worker and its graceful shutdown flush. The Memory Inspector tab starts the nightly 03:00 local consolidation scheduler and runs catch-up if no consolidation completed in the last day. With Edith disabled or initialization failed, Jarvis uses its original chat layout and answers normally.

## Inspect and control memories

Open **Memory Inspector** to browse by layer, kind, entity or status; view a superseded history chain; correct or hard-delete an item; export current SQLite state as JSON; check or rebuild FAISS; run consolidation; or run optional `VACUUM`. Correcting a fact creates a superseding row, preserving its history. Sensitive edits remain pending until the explicit chat confirmation. Hard delete removes the SQLite row, embedding, FAISS ID, and entity links, then truncates the WAL. Exports omit embedding BLOBs and are generated from the current database.

## Backups, restore, and deletion

SQLite is authoritative; `data/edith.faiss` is disposable and can be rebuilt from stored SQLite embeddings using **Rebuild vector index**. Back up the SQLite database only after a checkpoint and while Jarvis is stopped, or use SQLite's online backup API from a maintenance script. The FAISS file may be copied as a convenience but must never be treated as authoritative. After restoring a backup, point Edith at that restored database and rebuild/check the index before use.

Hard deletion cannot erase copies already captured by old backups, filesystem snapshots, or exported JSON files. Keep backup retention in mind when honoring a deletion request. Encryption at rest remains future work.

## Offline validation

The complete Edith command runs its tests, the restart-style acceptance flow, a held-out MiniLM retrieval evaluation, and a 10,000-memory warm context benchmark. It requires the MiniLM model already cached locally and makes no network calls:

```sh
EDITH_PYTHON=/path/to/python3.11 HF_HOME=/path/to/local/huggingface/cache ./scripts/test_edith.sh
```

Set `EDITH_PYTHON` to the Python executable in the environment with `requirements-edith.txt` and Jarvis's test dependencies installed. The slow real-model smoke test is marked `slow`; the script sets Hugging Face offline mode. To run only the fast/unit tests, use `python -m pytest tests/edith -q -m 'not slow'`.

Measured on the synthetic fixtures with the locally cached `all-MiniLM-L6-v2` model (87.34 MiB): lazy embedder construction took less than 0.001 s and left the model unloaded; first model load and embedding took 3.88 s. Retrieval quality on 30 synthetic memories and 30 paraphrased queries (10 development, 20 held out; no threshold tuning) was dev recall@3 1.00, dev recall@5 1.00, held-out recall@3 0.70, held-out recall@5 0.70. The held-out result is the baseline; improve it only with a separate development set and evaluate changes against a newly frozen holdout. A warm 10,000-memory context build measured 1.90 ms median and 2.63 ms p95 over 30 runs, below the 200 ms target. The timing fixture uses synthetic equal vectors and is a local engineering measurement, not a production latency guarantee.

The full repository pytest invocation also collects unrelated calendar tests. In the Phase 8 environment, four calendar tests failed: Google API imports were unavailable in the Edith-only environment, and two event fixtures use dates that are now in the past. Edith's own offline suite passed; these repository failures are outside the Edith package and were not changed here.
