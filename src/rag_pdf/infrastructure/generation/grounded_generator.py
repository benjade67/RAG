from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from rag_pdf.domain.models import Answer, Citation, RetrievedPassage
from rag_pdf.domain.ports import AnswerGenerator

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover - depends on local environment
    Mistral = None


class GroundedAnswerGenerator(AnswerGenerator):
    """Simple deterministic generator used as a placeholder.

    A production implementation should call an LLM with strict citation rules.
    """

    def generate(self, question: str, passages: list[RetrievedPassage]) -> Answer:
        citations = tuple(
            Citation(
                document_id=passage.chunk.document_id,
                page_number=passage.chunk.page_number,
                chunk_id=passage.chunk.chunk_id,
                bbox=passage.chunk.bbox,
                excerpt=passage.chunk.text[:240],
            )
            for passage in passages[:3]
        )

        if not citations:
            text = "Aucune preuve suffisante n'a ete retrouvee dans le corpus."
        else:
            refs = ", ".join(
                f"{citation.document_id} p.{citation.page_number}"
                for citation in citations
            )
            text = f"Reponse provisoire fondee sur: {refs}."

        return Answer(question=question, text=text, citations=citations)


class BaseLlmGroundedAnswerGenerator(AnswerGenerator):
    def _build_citations(self, passages: list[RetrievedPassage], limit: int = 5) -> tuple[Citation, ...]:
        return tuple(
            Citation(
                document_id=passage.chunk.document_id,
                page_number=passage.chunk.page_number,
                chunk_id=passage.chunk.chunk_id,
                bbox=passage.chunk.bbox,
                excerpt=passage.chunk.text[:240],
            )
            for passage in passages[:limit]
        )

    def _build_system_prompt(self) -> str:
        return (
            "Tu es un assistant RAG pour des plans PDF. "
            "Tu reponds uniquement a partir des extraits fournis. "
            "Si l'information n'est pas prouvee, dis-le clairement. "
            "Donne une reponse concise en francais et ajoute des references inline du type "
            "[DOC p.X chunk=ID] quand tu affirmes quelque chose."
        )

    def _build_user_prompt(self, question: str, passages: list[RetrievedPassage]) -> str:
        serialized_passages = []
        for index, passage in enumerate(passages[:5], start=1):
            chunk = passage.chunk
            serialized_passages.append(
                {
                    "rank": index,
                    "score": passage.score,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "region_id": chunk.region_id,
                    "region_kind": chunk.metadata.get("region_kind"),
                    "bbox": {
                        "x0": chunk.bbox.x0,
                        "y0": chunk.bbox.y0,
                        "x1": chunk.bbox.x1,
                        "y1": chunk.bbox.y1,
                    },
                    "text": chunk.text,
                }
            )

        return (
            f"Question: {question}\n\n"
            "Passages recuperes:\n"
            f"{json.dumps(serialized_passages, ensure_ascii=True, indent=2)}\n\n"
            "Instruction: reponds uniquement a partir de ces passages. "
            "Si plusieurs passages se contredisent ou sont incomplets, signale-le. "
            "N'invente pas."
        )

    def _extract_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content)


@dataclass(slots=True)
class MistralGroundedAnswerGenerator(BaseLlmGroundedAnswerGenerator):
    model: str = "mistral-small-latest"
    temperature: float = 0.1
    system_prompt: str | None = None
    client: object | None = None

    def generate(self, question: str, passages: list[RetrievedPassage]) -> Answer:
        citations = self._build_citations(passages)

        if not citations:
            return Answer(
                question=question,
                text="Aucune preuve suffisante n'a ete retrouvee dans le corpus.",
                citations=(),
                metadata={"generator": "mistral", "grounded": False},
            )

        client = self._get_client()
        prompt = self._build_user_prompt(question, passages)
        response = client.chat.complete(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt or self._build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        text = self._extract_text(content).strip()

        return Answer(
            question=question,
            text=text,
            citations=citations,
            metadata={
                "generator": "mistral",
                "model": self.model,
                "grounded": True,
                "temperature": self.temperature,
            },
        )

    def _get_client(self) -> object:
        if self.client is not None:
            return self.client
        if Mistral is None:
            raise RuntimeError(
                "Le package `mistralai` n'est pas installe. Installez les dependances avec `pip install -e .`."
            )

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "La variable d'environnement MISTRAL_API_KEY est requise pour la generation via Mistral."
            )
        return Mistral(api_key=api_key)



@dataclass(slots=True)
class OllamaGroundedAnswerGenerator(BaseLlmGroundedAnswerGenerator):
    model: str = "llama3.1"
    temperature: float = 0.1
    system_prompt: str | None = None
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 180

    def generate(self, question: str, passages: list[RetrievedPassage]) -> Answer:
        citations = self._build_citations(passages)

        if not citations:
            return Answer(
                question=question,
                text="Aucune preuve suffisante n'a ete retrouvee dans le corpus.",
                citations=(),
                metadata={"generator": "ollama", "grounded": False},
            )

        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
            "keep_alive": "10m",
            "messages": [
                {"role": "system", "content": self.system_prompt or self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(question, passages)},
            ],
        }
        content = self._call_ollama(payload)
        text = self._extract_ollama_text(content).strip()

        return Answer(
            question=question,
            text=text,
            citations=citations,
            metadata={
                "generator": "ollama",
                "model": self.model,
                "grounded": True,
                "temperature": self.temperature,
            },
        )

    def _call_ollama(self, payload: dict) -> dict:
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "Ollama a retourne une erreur HTTP "
                f"{exc.code}. Details: {details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Impossible de contacter Ollama. Verifiez que le serveur local est lance "
                f"sur {self.base_url}."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                "Ollama a depasse le delai d'attente. "
                "Le modele est peut-etre en cours de chargement ou la machine manque de ressources."
            ) from exc

    def _extract_ollama_text(self, payload: dict) -> str:
        message = payload.get("message", {})
        return self._extract_text(message.get("content", ""))
