"""SQLite storage and controls for explicit, local VOCA memories."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from uuid import uuid4


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    category: str
    created_at: str


class MemoryStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def is_enabled(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM app_state WHERE key = 'memory_enabled'").fetchone()
        return row is None or row["value"] == "true"

    def set_enabled(self, enabled: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO app_state (key, value) VALUES ('memory_enabled', ?)",
                ("true" if enabled else "false",),
            )

    def add(self, content: str, *, category: str = "general") -> Memory:
        cleaned_content = content.strip()
        if not cleaned_content:
            raise ValueError("Memory content cannot be empty.")
        cleaned_category = category.strip().lower() or "general"
        existing = self._find_exact(cleaned_content)
        if existing is not None:
            return existing

        memory = Memory(
            id=str(uuid4()),
            content=cleaned_content,
            category=cleaned_category,
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO memories (id, content, category, created_at) VALUES (?, ?, ?, ?)",
                (memory.id, memory.content, memory.category, memory.created_at),
            )
        return memory

    def list(self, limit: int = 50) -> list[Memory]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return result.rowcount > 0

    def _find_exact(self, content: str) -> Memory | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE content = ?", (content,)).fetchone()
        return _memory_from_row(row) if row else None


def _memory_from_row(row: sqlite3.Row) -> Memory:
    return Memory(id=row["id"], content=row["content"], category=row["category"], created_at=row["created_at"])
