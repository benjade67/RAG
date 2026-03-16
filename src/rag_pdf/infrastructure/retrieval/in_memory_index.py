from __future__ import annotations

from dataclasses import dataclass, field

from rag_pdf.domain.models import DocumentChunk, RetrievedPassage
from rag_pdf.domain.ports import ChunkIndex, LexicalSearcher


@dataclass
class InMemoryChunkIndex(ChunkIndex, LexicalSearcher):
    _chunks: list[DocumentChunk] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list)

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self._chunks.extend(chunks)
        self._vectors.extend(embeddings)

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        query_terms = {term.lower() for term in query.split()}
        scored: list[RetrievedPassage] = []

        for chunk in self._chunks:
            text_terms = set(chunk.text.lower().split())
            overlap = len(query_terms & text_terms)
            if overlap == 0 and chunk.text:
                continue

            scored.append(RetrievedPassage(chunk=chunk, score=float(overlap)))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
