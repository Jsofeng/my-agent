# Edith decisions

- Use a new conversation per Gradio browser session/app launch, and start a new conversation after six idle hours, because the current UI has session state but no clear-chat action.
- Prefix each user task with a delimited memory reference block, because the existing integration invokes `agent.run(task=...)` and does not pass display history or per-turn system prompts.
- Add memory safety guidance to the system prompt only when `EDITH_ENABLED` is true, because the disabled path must preserve the original prompt and behavior.
- Use Python 3.11 for integration validation; the machine's system `python3` is 3.9.6, while the app source uses Python 3.10 union syntax and smolagents 1.26.0 requires Python 3.10 or newer.
- Keep FAISS behind a `VectorStore` interface and store embeddings in SQLite, because SQLite remains authoritative and FAISS can be rebuilt; `sqlite-vec` was rejected due to extension-loading requirements and project maturity.
- Use `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional L2-normalized vectors in `IndexIDMap2(IndexFlatIP(384))`, because exact cosine search and SQLite IDs give straightforward deletion and recovery at personal scale.
- Use Python's `sqlite3` module with WAL, a busy timeout, one thread-local connection per worker thread, and a repository write lock, because UI, extraction, and scheduler code share one database.
- Pin Edith's new dependencies in `requirements-edith.txt`, because the repository contains no existing dependency manifest and a partial `requirements.txt` would misrepresent the full Jarvis environment.
- Keep all timestamp fields as UTC ISO-8601 text and default the user timezone to `America/Toronto`, because temporal parsing and storage need a stable common representation.
