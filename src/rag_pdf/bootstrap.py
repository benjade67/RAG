from __future__ import annotations

import os

from rag_pdf.application.services import IngestionService, QuestionAnsweringService, RetrievalService
from rag_pdf.config import load_environment
from rag_pdf.infrastructure.chunking.layout_chunker import LayoutAwareChunker
from rag_pdf.infrastructure.embeddings.stub_embeddings import (
    MistralEmbeddingProvider,
    OllamaEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    StubEmbeddingProvider,
)
from rag_pdf.infrastructure.generation.grounded_generator import (
    GroundedAnswerGenerator,
    MistralGroundedAnswerGenerator,
    OllamaGroundedAnswerGenerator,
)
from rag_pdf.infrastructure.parsers.pymupdf_parser import PyMuPdfParser
from rag_pdf.infrastructure.prompts.prompt_registry import PromptRegistry
from rag_pdf.infrastructure.retrieval.sqlite_hybrid_index import SqliteHybridChunkIndex


def build_app() -> tuple[IngestionService, QuestionAnsweringService]:
    load_environment()
    index = SqliteHybridChunkIndex()
    embeddings = _build_embedding_provider()
    ingestion = IngestionService(
        parser=PyMuPdfParser(),
        chunker=LayoutAwareChunker(),
        embeddings=embeddings,
        index=index,
    )
    retrieval = RetrievalService(
        vector_index=index,
        embeddings=embeddings,
        lexical_searcher=index,
        reranker=None,
    )
    qa = QuestionAnsweringService(
        retrieval=retrieval,
        generator=_build_answer_generator(),
    )
    return ingestion, qa


def _build_embedding_provider() -> (
    StubEmbeddingProvider
    | SentenceTransformerEmbeddingProvider
    | OllamaEmbeddingProvider
    | MistralEmbeddingProvider
):
    backend = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").lower()

    if backend == "stub":
        return StubEmbeddingProvider()
    if backend == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=os.getenv(
                "SENTENCE_TRANSFORMERS_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        )
    if backend == "ollama":
        return OllamaEmbeddingProvider(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
        )
    if backend == "mistral":
        return MistralEmbeddingProvider(
            model=os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        )
    return StubEmbeddingProvider()


def _build_answer_generator() -> (
    GroundedAnswerGenerator | MistralGroundedAnswerGenerator | OllamaGroundedAnswerGenerator
):
    backend = os.getenv("LLM_BACKEND", "auto").lower()
    prompt = PromptRegistry().get_active_prompt().content
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    if backend == "fallback":
        return GroundedAnswerGenerator()

    if backend in {"auto", "mistral"} and os.getenv("MISTRAL_API_KEY"):
        return MistralGroundedAnswerGenerator(
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            temperature=temperature,
            system_prompt=prompt,
        )

    if backend in {"auto", "ollama"} and os.getenv("OLLAMA_MODEL"):
        return OllamaGroundedAnswerGenerator(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            temperature=temperature,
            system_prompt=prompt,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
        )
    return GroundedAnswerGenerator()
