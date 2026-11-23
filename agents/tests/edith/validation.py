"""Offline retrieval quality and warm-cache scale measurements for Project Edith.

Run with HF_HOME pointing at a locally cached MiniLM model and HF_HUB_OFFLINE=1.
All facts and queries below are synthetic evaluation fixtures.
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path

# Keep local CPU inference bounded on small developer machines.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from edith.config import EdithConfig
from edith.context.context_manager import ContextManager
from edith.db import MemoryRepository
from edith.embeddings.base import Embedder
from edith.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edith.models import MemoryInput
from edith.vector import FaissVectorStore, VectorSynchronizer


GOLDEN = [
    ("Prefers working in a quiet room with minimal background noise", "prefers silence while concentrating"),
    ("Likes collaborating with a small team instead of a large group", "works best with a handful of colleagues"),
    ("Prefers written project updates over unscheduled calls", "would rather get progress notes in writing"),
    ("Enjoys starting focused work early in the morning", "is most productive at the beginning of the day"),
    ("Prefers concise explanations with practical examples", "likes short answers that show a real example"),
    ("Can build data pipelines using Apache Airflow", "has experience orchestrating ETL jobs"),
    ("Uses Python for automation and data analysis", "writes scripts to analyze datasets"),
    ("Has experience designing REST APIs with FastAPI", "builds web services in Python"),
    ("Can query relational databases with SQL", "is comfortable retrieving records from database tables"),
    ("Uses Docker to package applications", "can containerize a software service"),
    ("Project Cedar is a personal recipe organizer", "what is the purpose of the recipe app"),
    ("Project Harbor tracks volunteer shift availability", "how does the scheduling tool work"),
    ("Project Lantern is a mobile reading list", "what is the book tracking application called"),
    ("Project Meadow visualizes neighborhood tree data", "which project maps local trees"),
    ("Project Orbit is a budget planning prototype", "what does the expense planning project do"),
    ("Attended a community data meetup in March 2025", "when did the community analytics gathering happen"),
    ("Completed a prototype review on 2025-04-12", "what date was the initial design walkthrough"),
    ("Presented a demo to the volunteer group on 2025-06-03", "when was the volunteer software shown"),
    ("Finished a database migration on 2025-08-19", "when did the data schema transition finish"),
    ("Joined a product planning session on 2025-10-02", "when was the roadmap discussion"),
    ("Goal is to complete a cloud engineering certificate", "what credential is being pursued"),
    ("Wants to improve public speaking confidence", "which communication ability is a growth goal"),
    ("Plans to publish a small open source library", "what kind of software release is intended"),
    ("Aims to mentor a new developer this year", "what contribution goal involves coaching"),
    ("Would like to run a half marathon next spring", "what athletic milestone is planned"),
    ("The fictional Acme example uses a blue interface", "what color is the sample Acme screen"),
    ("A synthetic demo stores notes in a local SQLite file", "where does the toy application keep its notes"),
    ("The sample sensor reports temperature every minute", "how frequently does the demo thermometer send data"),
    ("A mock storefront groups items by category", "how are products arranged in the sample shop"),
    ("The imaginary garden planner tracks watering dates", "what does the fictional gardening tool record"),
]

# The first ten are used as the development split; the held-out queries below
# are fixed in advance and are never used to change thresholds or weights.
DEV = [
    (0, "needs a low-distraction environment for deep work"),
    (1, "prefers a close-knit working group"),
    (2, "how should project status be communicated"),
    (5, "which workflow tool can coordinate data movement"),
    (6, "what language is used for analysis scripts"),
    (10, "tell me about the recipe management app"),
    (11, "which tool organizes volunteer shifts"),
    (15, "when was the local data networking event"),
    (20, "what professional qualification is planned"),
    (21, "what speaking skill is the goal"),
]
HELD_OUT = [
    (3, "when during the day is focused work easiest"),
    (4, "what response style is most useful"),
    (7, "which framework has been used for web endpoints"),
    (8, "what database query language is familiar"),
    (9, "how are apps packaged into containers"),
    (12, "what is the name of the reading tracker"),
    (13, "which prototype displays urban greenery"),
    (16, "when did the prototype evaluation happen"),
    (18, "what date was the database conversion completed"),
    (22, "what open source work is planned"),
    (23, "what distance race is planned for next season"),
    (24, "what example uses blue colors"),
    (25, "which database backs the small demo notes"),
    (26, "how often does the sample device send readings"),
    (27, "how are goods sorted in the mock online shop"),
    (28, "what dates does the garden example keep"),
    (14, "describe the prototype related to planning expenses"),
    (17, "when did the group see the volunteer demo"),
    (19, "what endurance event is on the plan"),
    (2, "prefers receiving progress without an impromptu call"),
]


class BenchmarkEmbedder(Embedder):
    model_name = "benchmark-identical-v1"
    dimension = 384

    def embed(self, texts):
        return np.tile(np.eye(1, self.dimension, 0, dtype=np.float32), (len(texts), 1))


def _recall_at(results: list[dict], expected: int, cutoff: int) -> float:
    return float(any(int(row["id"]) == expected for row in results[:cutoff]))


def run_quality() -> dict:
    cache_root = os.environ.get("HF_HOME")
    if not cache_root:
        raise RuntimeError("Set HF_HOME to the local MiniLM cache; the evaluation is offline-only.")
    init_started = time.perf_counter()
    embedder = SentenceTransformerEmbedder()
    embedder_init_seconds = time.perf_counter() - init_started
    remains_unloaded = embedder.__class__._model is None
    started = time.perf_counter()
    embedder.embed(["Warm up the local sentence transformer."])
    first_load_seconds = time.perf_counter() - started
    model_cache = Path(cache_root) / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    resolved_files = {path.resolve() for path in model_cache.rglob("*") if path.is_file()}
    model_bytes = sum(path.stat().st_size for path in resolved_files)
    with tempfile.TemporaryDirectory(prefix="edith-quality-") as temp_dir:
        root = Path(temp_dir)
        settings = EdithConfig(database_path=root / "edith.db", index_path=root / "edith.faiss")
        repo = MemoryRepository(settings.database_path)
        try:
            sync = VectorSynchronizer(repo, embedder, FaissVectorStore(embedder.dimension), settings.index_path)
            ids = []
            for content, _ in GOLDEN:
                ids.append(repo.create_memory(MemoryInput(
                    layer="semantic", kind="profile_fact", content="Synthetic fixture: " + content,
                    subject="synthetic-eval", importance=0.7, confidence=0.9)))
            sync.reconcile()
            dev_scores = []
            held_scores = []
            for expected_idx, query in DEV:
                results = sync.search(query, top_k=5, min_similarity=0.0)
                dev_scores.append((_recall_at(results, ids[expected_idx], 3),
                                   _recall_at(results, ids[expected_idx], 5)))
            for expected_idx, query in HELD_OUT:
                results = sync.search(query, top_k=5, min_similarity=0.0)
                held_scores.append((_recall_at(results, ids[expected_idx], 3),
                                    _recall_at(results, ids[expected_idx], 5)))
            return {
                "fixture_memories": len(GOLDEN), "dev_queries": len(DEV),
                "held_out_queries": len(HELD_OUT),
                "dev_recall_at_3": round(statistics.mean(x[0] for x in dev_scores), 3),
                "dev_recall_at_5": round(statistics.mean(x[1] for x in dev_scores), 3),
                "held_out_recall_at_3": round(statistics.mean(x[0] for x in held_scores), 3),
                "held_out_recall_at_5": round(statistics.mean(x[1] for x in held_scores), 3),
                "model_cache_mib": round(model_bytes / (1024 * 1024), 2),
                "lazy_embedder_init_seconds": round(embedder_init_seconds, 6),
                "model_still_unloaded_after_init": remains_unloaded,
                "first_load_seconds": round(first_load_seconds, 3),
            }
        finally:
            repo.close()


def run_performance(memory_count: int = 10000, repetitions: int = 30) -> dict:
    with tempfile.TemporaryDirectory(prefix="edith-perf-") as temp_dir:
        root = Path(temp_dir)
        settings = EdithConfig(database_path=root / "edith.db", index_path=root / "edith.faiss",
                               context_token_budget=600, top_k=8)
        repo = MemoryRepository(settings.database_path)
        try:
            embedder = BenchmarkEmbedder()
            sync = VectorSynchronizer(repo, embedder, FaissVectorStore(embedder.dimension), settings.index_path)
            vector = embedder.embed(["synthetic benchmark query"])[0]
            for index in range(memory_count):
                layer = "semantic" if index % 2 == 0 else "episodic"
                kind = "profile_fact" if layer == "semantic" else "event"
                repo.create_memory(MemoryInput(
                    layer=layer, kind=kind,
                    content="Synthetic performance fixture memory number %05d" % index,
                    subject="benchmark", importance=0.7, confidence=0.9,
                    occurred_at="2026-01-01T00:00:00Z" if layer == "episodic" else None,
                    embedding=vector.tobytes(), embedding_model=embedder.model_name,
                ))
            sync.reconcile()
            manager = ContextManager(repo, sync, settings.context_token_budget, settings.top_k, settings)
            timings = []
            for _ in range(repetitions):
                started = time.perf_counter()
                manager.build("synthetic benchmark query")
                timings.append((time.perf_counter() - started) * 1000)
            timings.sort()
            median_ms = statistics.median(timings)
            return {"memory_count": memory_count, "repetitions": repetitions,
                    "warm_context_median_ms": round(median_ms, 3),
                    "warm_context_p95_ms": round(timings[int((len(timings) - 1) * 0.95)], 3),
                    "target_under_200ms": median_ms < 200}
        finally:
            repo.close()


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("retrieval_quality:", run_quality())
    print("warm_performance:", run_performance())
