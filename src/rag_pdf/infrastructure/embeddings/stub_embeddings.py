from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from rag_pdf.domain.ports import EmbeddingProvider

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover - depends on local environment
    Mistral = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - depends on local environment
    SentenceTransformer = None


class StubEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


@dataclass(slots=True)
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model: object | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def _get_model(self):
        if self.model is not None:
            return self.model
        if SentenceTransformer is None:
            raise RuntimeError(
                "Le package `sentence-transformers` n'est pas installe. "
                "Installez les dependances avec `pip install -e .`."
            )
        self.model = SentenceTransformer(self.model_name)
        return self.model


@dataclass(slots=True)
class OllamaEmbeddingProvider(EmbeddingProvider):
    model: str = "nomic-embed-text"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 180

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": text,
        }
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "Ollama embeddings a retourne une erreur HTTP "
                f"{exc.code}. Details: {details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Impossible de contacter Ollama pour les embeddings. "
                f"Verifiez le serveur sur {self.base_url}."
            ) from exc

        embeddings = parsed.get("embeddings", [])
        if not embeddings:
            raise RuntimeError("Ollama n'a retourne aucun embedding.")
        return embeddings[0]


@dataclass(slots=True)
class MistralEmbeddingProvider(EmbeddingProvider):
    model: str = "mistral-embed"
    client: object | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(model=self.model, inputs=texts)
        return [item.embedding for item in response.data]

    def _get_client(self):
        if self.client is not None:
            return self.client
        if Mistral is None:
            raise RuntimeError(
                "Le package `mistralai` n'est pas installe. Installez les dependances avec `pip install -e .`."
            )

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "La variable d'environnement MISTRAL_API_KEY est requise pour les embeddings Mistral."
            )
        self.client = Mistral(api_key=api_key)
        return self.client
