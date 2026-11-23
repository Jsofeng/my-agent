# Project Edith architecture

Edith is a local, single-user memory package integrated through an adapter. SQLite at `data/edith.db` is the source of truth. A future FAISS index at `data/edith.faiss` is derived from SQLite embeddings. The package does not import Jarvis or smolagents; only `edith/integration/jarvis_adapter.py` will connect application code to it.

Phase 1 provides configuration, typed input records, ordered SQLite migrations, a write-coordinating repository, episodic and semantic persistence services, conversation/message and rolling-summary persistence primitives, pending confirmation/question tables, and a common-secret scanner. Short-term messages and summaries remain separate from episodic and semantic memories.

Each repository instance uses a SQLite connection local to each thread. SQLite runs in WAL mode with `busy_timeout` and `secure_delete` enabled. Repository writes are serialized by an in-process reentrant lock. Hard deletion removes entity links and the memory row in SQLite, writes only a content-free event, commits, and requests a truncating WAL checkpoint. FAISS synchronization and vector locking are introduced in Phase 2.

The Gradio chat currently stores display messages in `gr.State` but sends each new prompt as an individual task. The adapter will use a delimited task prefix for retrieved context. Memory errors must fail open at that boundary.

The extraction pre-filter will skip empty/very short acknowledgements, generic requests with no user-specific details, and command-like turns. It will retain batches containing first-person facts, preferences, plans, decisions, events, or explicit corrections. Extraction runs asynchronously and does not block the response.

Encryption at rest is intentionally out of scope and remains future work.
