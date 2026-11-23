# Project Edith architecture

Edith is a local, single-user memory package integrated through an adapter. SQLite at `data/edith.db` is the source of truth. A future FAISS index at `data/edith.faiss` is derived from SQLite embeddings. The package does not import Jarvis or smolagents; only `edith/integration/jarvis_adapter.py` will connect application code to it.

Phase 1 provides configuration, typed input records, ordered SQLite migrations, a write-coordinating repository, episodic and semantic persistence services, conversation/message and rolling-summary persistence primitives, pending confirmation/question tables, and a common-secret scanner. Short-term messages and summaries remain separate from episodic and semantic memories. Phase 2 adds a lazy `sentence-transformers/all-MiniLM-L6-v2` embedder and exact `IndexIDMap2(IndexFlatIP(384))` search behind `Embedder` and `VectorStore` interfaces.

Each repository instance uses a SQLite connection local to each thread. SQLite runs in WAL mode with `busy_timeout` and `secure_delete` enabled. Repository writes are serialized by an in-process reentrant lock. Hard deletion removes entity links and the memory row in SQLite, writes only a content-free event, commits, and requests a truncating WAL checkpoint. FAISS writes and saves use a store lock, and reads use the same lock for snapshot safety. Vector BLOBs and `index_status='pending'` are committed before FAISS changes; the row is marked indexed only after an atomic index save. Reconciliation removes orphan IDs and rebuilds from stored vectors, re-embedding only when missing or when the model changes.

The Gradio chat currently stores display messages in `gr.State` but sends each new prompt as an individual task. The adapter will use a delimited task prefix for retrieved context. Memory errors must fail open at that boundary.

When `EDITH_ENABLED` is true, Jarvis creates an app-launch conversation, appends retrieved context to the internal task while leaving the visible user message unchanged, and installs the `save_memory` and `search_memory` tools. The system prompt then marks memory as untrusted data and prohibits invented history. Memory retrieval failures log only the exception type and leave the original task intact. The explicit save tool rejects secret-like strings and holds detected sensitive-category candidates in `pending_confirmations` instead of inserting them into `memories`.

The extraction pre-filter will skip empty/very short acknowledgements, generic requests with no user-specific details, and command-like turns. It will retain batches containing first-person facts, preferences, plans, decisions, events, or explicit corrections. Extraction runs asynchronously and does not block the response.

Encryption at rest is intentionally out of scope and remains future work.
