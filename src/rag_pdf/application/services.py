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
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            return 0
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        self.index.upsert(chunks, vectors)
        return len(chunks)

    def reingest(self, document: SourceDocument) -> int:
        self.index.delete([document.document_id])
        return self.ingest(document)


@dataclass(slots=True)
class RetrievalService:
    vector_index: ChunkIndex
    embeddings: EmbeddingProvider
    lexical_searcher: LexicalSearcher | None = None
    reranker: PassageReranker | None = None

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        query_vector = self.embeddings.embed([query])[0]
        candidates = self.vector_index.search(query_vector, top_k=top_k, document_ids=document_ids)

        if self.lexical_searcher is not None:
            lexical = self.lexical_searcher.search_lexical(query, top_k=top_k, document_ids=document_ids)
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

    def answer(
        self,
        question: str,
        top_k: int = 8,
        document_ids: list[str] | None = None,
    ) -> Answer:
        passages = self.retrieval.retrieve(question, top_k=top_k, document_ids=document_ids)
        return self.generator.generate(question, passages)
