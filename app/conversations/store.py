"""SQLite-backed conversation history with no network or cloud dependency."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from uuid import uuid4


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoredMessage:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    audio_path: str | None


class ConversationStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    audio_path TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def create_conversation(self, title: str = "New conversation", *, make_active: bool = True) -> Conversation:
        now = _timestamp()
        conversation = Conversation(id=str(uuid4()), title=title, created_at=now, updated_at=now)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation.id, conversation.title, conversation.created_at, conversation.updated_at),
            )
            if make_active:
                connection.execute(
                    "INSERT OR REPLACE INTO app_state (key, value) VALUES ('active_conversation_id', ?)",
                    (conversation.id,),
                )
        return conversation

    def get_active_conversation(self) -> Conversation:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.* FROM conversations c
                JOIN app_state s ON s.value = c.id
                WHERE s.key = 'active_conversation_id'
                """
            ).fetchone()
        return _conversation_from_row(row) if row else self.create_conversation()

    def list_conversations(self, limit: int = 20) -> list[Conversation]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_conversation_from_row(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return _conversation_from_row(row) if row else None

    def update_title(self, conversation_id: str, title: str) -> None:
        cleaned_title = title.strip()
        if not cleaned_title:
            raise ValueError("Conversation title cannot be empty.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (cleaned_title, _timestamp(), conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its saved messages from the local database."""
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not exists:
                return False
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            connection.execute(
                "DELETE FROM app_state WHERE key = 'active_conversation_id' AND value = ?",
                (conversation_id,),
            )
        return True

    def add_message(self, conversation_id: str, role: str, content: str, *, audio_path: str | None = None) -> StoredMessage:
        if role not in {"user", "assistant"}:
            raise ValueError("Message role must be 'user' or 'assistant'.")
        cleaned_content = content.strip()
        if not cleaned_content:
            raise ValueError("Cannot store an empty message.")

        message = StoredMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=cleaned_content,
            created_at=_timestamp(),
            audio_path=audio_path,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, audio_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.audio_path,
                    message.created_at,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at, conversation_id),
            )
        return message

    def messages(self, conversation_id: str, *, limit: int | None = None) -> list[StoredMessage]:
        query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
        parameters: tuple[object, ...] = (conversation_id,)
        if limit is not None:
            query = """
                SELECT * FROM (
                    SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?
                ) ORDER BY created_at ASC
            """
            parameters = (conversation_id, limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_message_from_row(row) for row in rows]

    def get_message(self, message_id: str) -> StoredMessage | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _message_from_row(row) if row else None

    def update_message_audio(self, message_id: str, audio_path: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE messages SET audio_path = ? WHERE id = ?", (audio_path, message_id))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation(id=row["id"], title=row["title"], created_at=row["created_at"], updated_at=row["updated_at"])


def _message_from_row(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        audio_path=row["audio_path"],
    )
