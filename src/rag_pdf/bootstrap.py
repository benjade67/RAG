from __future__ import annotations

import os

from rag_pdf.application.services import IngestionService, QuestionAnsweringService, RetrievalService
from rag_pdf.infrastructure.chunking.layout_chunker import LayoutAwareChunker
from rag_pdf.infrastructure.embeddings.stub_embeddings import StubEmbeddingProvider
from rag_pdf.infrastructure.generation.grounded_generator import (
    GroundedAnswerGenerator,
    MistralGroundedAnswerGenerator,
    OllamaGroundedAnswerGenerator,
)
from rag_pdf.infrastructure.parsers.pymupdf_parser import PyMuPdfParser
from rag_pdf.infrastructure.retrieval.in_memory_index import InMemoryChunkIndex


def build_app() -> tuple[IngestionService, QuestionAnsweringService]:
    index = InMemoryChunkIndex()
    ingestion = IngestionService(
        parser=PyMuPdfParser(),
        chunker=LayoutAwareChunker(),
        embeddings=StubEmbeddingProvider(),
        index=index,
    )
    retrieval = RetrievalService(
        vector_index=index,
        lexical_searcher=index,
        reranker=None,
    )
    qa = QuestionAnsweringService(
        retrieval=retrieval,
        generator=_build_answer_generator(),
    )
    return ingestion, qa


def _build_answer_generator() -> (
    GroundedAnswerGenerator | MistralGroundedAnswerGenerator | OllamaGroundedAnswerGenerator
):
    if os.getenv("MISTRAL_API_KEY"):
        return MistralGroundedAnswerGenerator(
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        )
    if os.getenv("OLLAMA_MODEL"):
        return OllamaGroundedAnswerGenerator(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
        )
    return GroundedAnswerGenerator()
