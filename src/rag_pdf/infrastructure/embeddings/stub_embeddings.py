from __future__ import annotations

from rag_pdf.domain.ports import EmbeddingProvider


class StubEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]
