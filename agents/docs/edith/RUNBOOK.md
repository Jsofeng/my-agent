# Edith runbook

## Local data

The memory database is `data/edith.db`; the derived vector index will be `data/edith.faiss`. Keep both paths local and protect them as personal data. Do not include them in logs or issue reports.

## Hard deletion

Edith enables SQLite secure deletion and checkpoints the WAL after a hard delete. Old filesystem backups, copied databases, and snapshots may still contain deleted memory. Generate exports and new backups from the current database only. Encryption at rest is future work.

## Dependency setup

`requirements-edith.txt` pins Edith's new dependencies. It is not a complete dependency list for Jarvis because no project-wide manifest existed at the time this file was added.
