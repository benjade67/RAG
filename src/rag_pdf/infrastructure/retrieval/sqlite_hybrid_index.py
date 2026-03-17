from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from rag_pdf.config import get_project_root
from rag_pdf.domain.models import BoundingBox, DocumentChunk, RetrievedPassage
from rag_pdf.domain.ports import ChunkIndex, LexicalSearcher


class SqliteHybridChunkIndex(ChunkIndex, LexicalSearcher):
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path or (get_project_root() / "data" / "index" / "hybrid_index.db")
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        connection = self._connect()
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            normalized = self._normalize(embedding)
            connection.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,))
            connection.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,))
            connection.execute(
                """
                INSERT INTO chunks (
                    chunk_id,
                    document_id,
                    page_number,
                    region_id,
                    text,
                    bbox_json,
                    metadata_json,
                    embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.page_number,
                    chunk.region_id,
                    chunk.text,
                    json.dumps(
                        {
                            "x0": chunk.bbox.x0,
                            "y0": chunk.bbox.y0,
                            "x1": chunk.bbox.x1,
                            "y1": chunk.bbox.y1,
                        },
                        ensure_ascii=True,
                    ),
                    json.dumps(chunk.metadata, ensure_ascii=True),
                    json.dumps(normalized, ensure_ascii=True),
                ),
            )
            connection.execute(
                "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                (chunk.chunk_id, chunk.text),
            )
        connection.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        normalized_query = self._normalize(query_embedding)
        rows = self._fetch_rows(document_ids=document_ids)
        scored: list[RetrievedPassage] = []

        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = self._cosine_similarity(normalized_query, embedding)
            scored.append(RetrievedPassage(chunk=self._row_to_chunk(row), score=score))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def search_lexical(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        connection = self._connect()
        sql = """
            SELECT
                chunks.*,
                bm25(chunks_fts) AS lexical_score
            FROM chunks_fts
            JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
        """
        parameters: list[object] = [query]

        if document_ids:
            placeholders = ", ".join("?" for _ in document_ids)
            sql += f" AND chunks.document_id IN ({placeholders})"
            parameters.extend(document_ids)

        sql += " ORDER BY lexical_score LIMIT ?"
        parameters.append(top_k)

        try:
            rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            RetrievedPassage(chunk=self._row_to_chunk(row), score=float(-row["lexical_score"]))
            for row in rows
        ]

    def delete(self, document_ids: list[str]) -> None:
        if not document_ids:
            return
        connection = self._connect()
        placeholders = ", ".join("?" for _ in document_ids)
        chunk_rows = connection.execute(
            f"SELECT chunk_id FROM chunks WHERE document_id IN ({placeholders})",
            document_ids,
        ).fetchall()
        chunk_ids = [row["chunk_id"] for row in chunk_rows]

        if chunk_ids:
            chunk_placeholders = ", ".join("?" for _ in chunk_ids)
            connection.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({chunk_placeholders})",
                chunk_ids,
            )

        connection.execute(
            f"DELETE FROM chunks WHERE document_id IN ({placeholders})",
            document_ids,
        )
        connection.commit()

    def _initialize(self) -> None:
        connection = self._connect()
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                region_id TEXT NOT NULL,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                text
            )
            """
        )
        connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return self._connection

    def _fetch_rows(self, document_ids: list[str] | None = None) -> list[sqlite3.Row]:
        connection = self._connect()
        if not document_ids:
            return connection.execute("SELECT * FROM chunks").fetchall()

        placeholders = ", ".join("?" for _ in document_ids)
        sql = f"SELECT * FROM chunks WHERE document_id IN ({placeholders})"
        return connection.execute(sql, document_ids).fetchall()

    def _row_to_chunk(self, row: sqlite3.Row) -> DocumentChunk:
        bbox_payload = json.loads(row["bbox_json"])
        metadata_payload = json.loads(row["metadata_json"])
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            page_number=row["page_number"],
            region_id=row["region_id"],
            text=row["text"],
            bbox=BoundingBox(
                x0=bbox_payload["x0"],
                y0=bbox_payload["y0"],
                x1=bbox_payload["x1"],
                y1=bbox_payload["y1"],
            ),
            metadata=metadata_payload,
        )

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return [0.0 for _ in vector]
        return [value / norm for value in vector]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        size = min(len(left), len(right))
        if size == 0:
            return 0.0
        return sum(left[index] * right[index] for index in range(size))
