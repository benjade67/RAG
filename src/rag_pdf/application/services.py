from __future__ import annotations

from dataclasses import dataclass

from rag_pdf.domain.models import Answer, RetrievedPassage, SourceDocument
from rag_pdf.domain.ports import (
    AnswerGenerator,
    Chunker,
    ChunkIndex,
    DocumentParser,
    EmbeddingProvider,
    LexicalSearcher,
    PassageReranker,
)


@dataclass(slots=True)
class IngestionService:
    parser: DocumentParser
    chunker: Chunker
    embeddings: EmbeddingProvider
    index: ChunkIndex

    def ingest(self, document: SourceDocument) -> int:
        regions = self.parser.parse(document)
        chunks = self.chunker.chunk(document, regions)
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        self.index.upsert(chunks, vectors)
        return len(chunks)


@dataclass(slots=True)
class RetrievalService:
    vector_index: ChunkIndex
    lexical_searcher: LexicalSearcher | None = None
    reranker: PassageReranker | None = None

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedPassage]:
        candidates = self.vector_index.search(query, top_k=top_k)

        if self.lexical_searcher is not None:
            lexical = self.lexical_searcher.search(query, top_k=top_k)
            candidates = self._merge(candidates, lexical)

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)

        return candidates[:top_k]

    def _merge(
        self,
        vector_results: list[RetrievedPassage],
        lexical_results: list[RetrievedPassage],
    ) -> list[RetrievedPassage]:
        best_by_chunk = {
            item.chunk.chunk_id: item
            for item in vector_results
        }

        for item in lexical_results:
            current = best_by_chunk.get(item.chunk.chunk_id)
            if current is None or item.score > current.score:
                best_by_chunk[item.chunk.chunk_id] = item

        return sorted(best_by_chunk.values(), key=lambda item: item.score, reverse=True)


@dataclass(slots=True)
class QuestionAnsweringService:
    retrieval: RetrievalService
    generator: AnswerGenerator

    def answer(self, question: str, top_k: int = 8) -> Answer:
        passages = self.retrieval.retrieve(question, top_k=top_k)
        return self.generator.generate(question, passages)
