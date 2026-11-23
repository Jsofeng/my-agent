"""Ordered, idempotent database migrations."""

from __future__ import annotations

import sqlite3


MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'default',
        started_at TEXT NOT NULL, ended_at TEXT, title TEXT,
        status TEXT NOT NULL DEFAULT 'active'
    );
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
        role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
        content TEXT NOT NULL, created_at TEXT NOT NULL, extracted INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
        summary TEXT NOT NULL, covers_up_to_message_id INTEGER,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY, uuid TEXT NOT NULL UNIQUE,
        user_id TEXT NOT NULL DEFAULT 'default',
        layer TEXT NOT NULL CHECK(layer IN ('episodic','semantic')),
        kind TEXT NOT NULL CHECK(kind IN ('event','decision','achievement','experience',
            'preference','skill','interest','goal','relationship','project','profile_fact')),
        content TEXT NOT NULL, subject TEXT, predicate TEXT,
        importance REAL NOT NULL, confidence REAL NOT NULL,
        provenance TEXT NOT NULL CHECK(provenance IN ('user_stated','inferred','user_confirmed')),
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','archived','deleted')),
        index_status TEXT NOT NULL DEFAULT 'pending' CHECK(index_status IN ('indexed','pending')),
        occurred_at TEXT, valid_from TEXT, valid_to TEXT, superseded_by INTEGER,
        consolidated_into INTEGER, source_conversation_id INTEGER,
        source_message_ids TEXT NOT NULL DEFAULT '[]', content_hash TEXT NOT NULL,
        embedding BLOB, embedding_model TEXT, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, last_accessed_at TEXT, access_count INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'default',
        canonical_name TEXT NOT NULL, type TEXT NOT NULL CHECK(type IN
          ('person','company','project','technology','event','place','other')),
        aliases_json TEXT NOT NULL DEFAULT '[]', first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL, mention_count INTEGER NOT NULL DEFAULT 1,
        UNIQUE(user_id, canonical_name)
    );
    CREATE TABLE IF NOT EXISTS memory_entities (
        memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        role TEXT NOT NULL DEFAULT 'mentioned', PRIMARY KEY(memory_id, entity_id, role)
    );
    CREATE TABLE IF NOT EXISTS memory_events (
        id INTEGER PRIMARY KEY, memory_id INTEGER NOT NULL, action TEXT NOT NULL,
        at TEXT NOT NULL, detail_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS pending_confirmations (
        id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
        candidate_json TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
        expires_after_turn INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending'
    );
    CREATE TABLE IF NOT EXISTS pending_questions (
        id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
        kind TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS consolidation_runs (
        id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
        status TEXT NOT NULL, stats_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS index_state (
        embedding_model TEXT PRIMARY KEY, dim INTEGER NOT NULL,
        vector_count INTEGER NOT NULL DEFAULT 0, last_synced_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_memories_user_layer_status ON memories(user_id,layer,status);
    CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
    CREATE INDEX IF NOT EXISTS idx_memories_subject_predicate_status ON memories(subject,predicate,status);
    CREATE INDEX IF NOT EXISTS idx_memories_occurred_at ON memories(occurred_at);
    CREATE INDEX IF NOT EXISTS idx_memories_index_status ON memories(index_status);
    CREATE INDEX IF NOT EXISTS idx_messages_extracted ON messages(extracted);
    CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id,memory_id);
    """,
)


def apply_migrations(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = connection.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    version = int(row[0]) if row else 0
    if row is None:
        connection.execute("INSERT INTO schema_version(version) VALUES (0)")
    for target, sql in enumerate(MIGRATIONS, start=1):
        if target <= version:
            continue
        connection.executescript("BEGIN IMMEDIATE;\n" + sql + f"\nUPDATE schema_version SET version={target};\nCOMMIT;")
