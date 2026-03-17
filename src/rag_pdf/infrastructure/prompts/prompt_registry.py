from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_pdf.config import get_project_root


DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant RAG pour des plans PDF. "
    "Tu reponds uniquement a partir des extraits fournis. "
    "Si l'information n'est pas prouvee, dis-le clairement. "
    "Donne une reponse concise en francais et ajoute des references inline du type "
    "[DOC p.X chunk=ID] quand tu affirmes quelque chose."
)


@dataclass(frozen=True)
class PromptVersion:
    version_id: str
    name: str
    content: str
    created_at: str
    is_active: bool = False


class PromptRegistry:
    def __init__(self, registry_path: Path | None = None) -> None:
        self._registry_path = registry_path or get_project_root() / "data" / "prompts" / "registry.json"
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._registry_path.exists():
            self._save(
                [
                    PromptVersion(
                        version_id="v1",
                        name="Prompt par defaut",
                        content=DEFAULT_SYSTEM_PROMPT,
                        created_at=datetime.now(UTC).isoformat(),
                        is_active=True,
                    )
                ]
            )

    def list_prompts(self) -> list[PromptVersion]:
        payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        return [PromptVersion(**item) for item in payload]

    def get_active_prompt(self) -> PromptVersion:
        prompts = self.list_prompts()
        for prompt in prompts:
            if prompt.is_active:
                return prompt
        raise RuntimeError("Aucun prompt actif n'est configure.")

    def create_prompt_version(self, name: str, content: str, activate: bool = True) -> PromptVersion:
        prompts = self.list_prompts()
        next_number = len(prompts) + 1
        new_prompt = PromptVersion(
            version_id=f"v{next_number}",
            name=name.strip() or f"Prompt v{next_number}",
            content=content.strip(),
            created_at=datetime.now(UTC).isoformat(),
            is_active=activate,
        )
        if activate:
            prompts = [PromptVersion(**{**asdict(prompt), "is_active": False}) for prompt in prompts]
        prompts.append(new_prompt)
        self._save(prompts)
        return new_prompt

    def activate_prompt(self, version_id: str) -> PromptVersion:
        prompts = self.list_prompts()
        updated: list[PromptVersion] = []
        active_prompt: PromptVersion | None = None

        for prompt in prompts:
            is_active = prompt.version_id == version_id
            updated_prompt = PromptVersion(**{**asdict(prompt), "is_active": is_active})
            updated.append(updated_prompt)
            if is_active:
                active_prompt = updated_prompt

        if active_prompt is None:
            raise ValueError(f"Prompt inconnu: {version_id}")

        self._save(updated)
        return active_prompt

    def _save(self, prompts: list[PromptVersion]) -> None:
        payload = [asdict(prompt) for prompt in prompts]
        self._registry_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
