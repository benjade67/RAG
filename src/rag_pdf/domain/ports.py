from __future__ import annotations

from abc import ABC, abstractmethod

from rag_pdf.domain.models import (
    Answer,
    DocumentChunk,
    RetrievedPassage,
    SourceDocument,
    VisualRegion,
)


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, document: SourceDocument) -> list[VisualRegion]:
        """Extract page regions, text spans and layout information."""


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: SourceDocument, regions: list[VisualRegion]) -> list[DocumentChunk]:
        """Build layout-aware chunks from parsed regions."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for text chunks or queries."""


class ChunkIndex(ABC):
    @abstractmethod
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks and vectors in the retrieval backend."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve candidate chunks from the index."""


class LexicalSearcher(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve keyword-oriented results."""


class PassageReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
        """Reorder retrieved passages using richer relevance signals."""


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, passages: list[RetrievedPassage]) -> Answer:
        """Produce an answer grounded in cited evidence."""
