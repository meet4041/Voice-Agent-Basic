"""SQLite-backed semantic search for local VOCA content."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from app.conversations.store import ConversationStore
from app.memory.store import MemoryStore


@dataclass(frozen=True)
class SearchResult:
    source_type: str
    source_id: str
    title: str
    content: str
    score: float


class SemanticSearchService:
    def __init__(self, database_path: Path, embedding_provider) -> None:
        self.database_path = database_path
        self.embedding_provider = embedding_provider
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_index (
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    PRIMARY KEY (source_type, source_id)
                )
                """
            )

    def refresh(self, conversation_store: ConversationStore, memory_store: MemoryStore) -> int:
        documents: list[tuple[str, str, str, str]] = []
        for conversation in conversation_store.list_conversations(limit=1000):
            for message in conversation_store.messages(conversation.id):
                documents.append(("message", message.id, conversation.title, message.content))
        for memory in memory_store.list(limit=1000):
            documents.append(("memory", memory.id, f"Memory: {memory.category}", memory.content))

        changed = [document for document in documents if self._needs_embedding(*document)]
        if changed:
            embeddings = self.embedding_provider.embed([document[3] for document in changed])
            with self._connect() as connection:
                for document, embedding in zip(changed, embeddings, strict=True):
                    source_type, source_id, title, content = document
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO semantic_index
                        (source_type, source_id, title, content, content_hash, embedding_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (source_type, source_id, title, content, _content_hash(content), json.dumps(embedding)),
                    )
        return len(changed)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")
        query_embedding = np.asarray(self.embedding_provider.embed([cleaned_query])[0], dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM semantic_index").fetchall()
        results: list[SearchResult] = []
        for row in rows:
            document_embedding = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
            denominator = query_norm * np.linalg.norm(document_embedding)
            score = float(np.dot(query_embedding, document_embedding) / denominator) if denominator else 0.0
            results.append(
                SearchResult(
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    title=row["title"],
                    content=row["content"],
                    score=score,
                )
            )
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _needs_embedding(self, source_type: str, source_id: str, _title: str, content: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_hash FROM semantic_index WHERE source_type = ? AND source_id = ?",
                (source_type, source_id),
            ).fetchone()
        return row is None or row["content_hash"] != _content_hash(content)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
