"""Durable LangGraph Checkpointer Factory (Postgres -> SqliteSaver -> Persistent File Saver)."""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT_DB = ROOT_DIR / "data" / "processed" / "langgraph_checkpoints.db"


class LocalSqliteCheckpointSaver(InMemorySaver):
    """Cross-platform SQLite-persisted checkpointer wrapping `InMemorySaver`.

    Guarantees durable restart survival on both Windows and Linux even when
    optional `langgraph-checkpoint-sqlite` / `aiosqlite` wheels are absent.
    """

    def __init__(self, db_path: str | Path = DEFAULT_CHECKPOINT_DB) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._hydrate_from_disk()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lg_kv_snapshots (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    storage_blob BLOB NOT NULL,
                    writes_blob BLOB NOT NULL,
                    blobs_blob BLOB NOT NULL
                );
                """
            )

    def _hydrate_from_disk(self) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT storage_blob, writes_blob, blobs_blob FROM lg_kv_snapshots WHERE id = 1"
            ).fetchone()
        if row:
            try:
                loaded_storage = pickle.loads(row[0])
                loaded_writes = pickle.loads(row[1])
                loaded_blobs = pickle.loads(row[2])
                self.storage.update(loaded_storage)
                self.writes.update(loaded_writes)
                self.blobs.update(loaded_blobs)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not hydrate SQLite checkpoints: %s", exc)

    def _flush_to_disk(self) -> None:
        storage_dict = {k: {ns: dict(cp) for ns, cp in v.items()} for k, v in self.storage.items()}
        writes_dict = dict(self.writes)
        blobs_dict = dict(self.blobs)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lg_kv_snapshots (id, storage_blob, writes_blob, blobs_blob)
                VALUES (1, ?, ?, ?)
                """,
                (
                    pickle.dumps(storage_dict),
                    pickle.dumps(writes_dict),
                    pickle.dumps(blobs_dict),
                ),
            )

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict[str, Any]:
        res = super().put(config, checkpoint, metadata, new_versions)
        self._flush_to_disk()
        return res

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path=task_path)
        self._flush_to_disk()

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict[str, Any]:
        res = await super().aput(config, checkpoint, metadata, new_versions)
        self._flush_to_disk()
        return res

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await super().aput_writes(config, writes, task_id, task_path=task_path)
        self._flush_to_disk()


def get_sync_or_local_checkpointer(
    db_path: str | Path | None = None,
) -> BaseCheckpointSaver[Any]:
    """Return a durable local SQLite checkpointer (safe in single-process & tests)."""
    resolved = Path(db_path or os.getenv("GIGBID_CHECKPOINT_DB", str(DEFAULT_CHECKPOINT_DB)))
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]

        conn = sqlite3.connect(str(resolved), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    except Exception:
        return LocalSqliteCheckpointSaver(db_path=resolved)


@asynccontextmanager
async def open_checkpointer(
    *,
    postgres_url: str | None = None,
    sqlite_path: str | Path | None = None,
) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """Open an async Postgres checkpointer when reachable, falling back to SQLite."""
    db_url = (postgres_url or os.getenv("DATABASE_URL", "")).strip()
    if db_url.startswith("postgresql") and os.getenv(
        "GIGBID_PREFER_POSTGRES_CHECKPOINTER", "false"
    ).lower() in {"1", "true"}:
        clean_url = db_url.replace("postgresql+psycopg://", "postgresql://")
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(clean_url) as pg_saver:
                await pg_saver.setup()
                yield pg_saver
                return
        except Exception as exc:
            logger.warning("Postgres checkpointer unavailable (%s); falling back to SQLite.", exc)

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore[import-not-found]

        resolved = Path(
            sqlite_path or os.getenv("GIGBID_CHECKPOINT_DB", str(DEFAULT_CHECKPOINT_DB))
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(resolved)) as sq_saver:
            yield sq_saver
            return
    except Exception:
        yield LocalSqliteCheckpointSaver(
            db_path=sqlite_path or os.getenv("GIGBID_CHECKPOINT_DB", str(DEFAULT_CHECKPOINT_DB))
        )


__all__ = [
    "LocalSqliteCheckpointSaver",
    "get_sync_or_local_checkpointer",
    "open_checkpointer",
]
