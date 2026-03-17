from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rag_pdf.config import get_project_root


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_terms: tuple[str, ...] = ()
    expected_document_id: str | None = None
    expected_page_number: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    question: str
    answer_text: str
    citations_count: int
    answer_contains_expected_terms: bool
    citation_document_hit: bool
    citation_page_hit: bool
    passed: bool


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    created_at: str
    total_cases: int
    answered_cases: int
    pass_rate: float
    answer_term_hit_rate: float
    citation_document_hit_rate: float
    citation_page_hit_rate: float
    results: tuple[EvaluationCaseResult, ...] = field(default_factory=tuple)


class EvaluationRegistry:
    def __init__(self, base_dir: Path | None = None) -> None:
        root = base_dir or get_project_root() / "data" / "evaluation"
        self._cases_path = root / "cases.json"
        self._runs_path = root / "runs.json"
        root.mkdir(parents=True, exist_ok=True)
        if not self._cases_path.exists():
            self._save_cases([])
        if not self._runs_path.exists():
            self._save_runs([])

    def list_cases(self) -> list[EvaluationCase]:
        payload = json.loads(self._cases_path.read_text(encoding="utf-8"))
        return [self._deserialize_case(item) for item in payload]

    def add_case(
        self,
        question: str,
        expected_terms: list[str] | tuple[str, ...] | None = None,
        expected_document_id: str | None = None,
        expected_page_number: int | None = None,
        notes: str = "",
    ) -> EvaluationCase:
        cases = self.list_cases()
        case = EvaluationCase(
            case_id=f"case-{len(cases) + 1}",
            question=question.strip(),
            expected_terms=tuple(term.strip() for term in (expected_terms or []) if term.strip()),
            expected_document_id=(expected_document_id or "").strip() or None,
            expected_page_number=expected_page_number,
            notes=notes.strip(),
        )
        cases.append(case)
        self._save_cases(cases)
        return case

    def list_runs(self) -> list[EvaluationRun]:
        payload = json.loads(self._runs_path.read_text(encoding="utf-8"))
        return [self._deserialize_run(item) for item in payload]

    def save_run(self, run: EvaluationRun) -> None:
        runs = self.list_runs()
        runs.append(run)
        self._save_runs(runs)

    def build_run(self, results: list[EvaluationCaseResult]) -> EvaluationRun:
        total_cases = len(results)
        answered_cases = sum(1 for item in results if item.answer_text.strip())
        pass_count = sum(1 for item in results if item.passed)
        term_hits = sum(1 for item in results if item.answer_contains_expected_terms)
        document_hits = sum(1 for item in results if item.citation_document_hit)
        page_hits = sum(1 for item in results if item.citation_page_hit)

        return EvaluationRun(
            run_id=f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            created_at=datetime.now(UTC).isoformat(),
            total_cases=total_cases,
            answered_cases=answered_cases,
            pass_rate=(pass_count / total_cases) if total_cases else 0.0,
            answer_term_hit_rate=(term_hits / total_cases) if total_cases else 0.0,
            citation_document_hit_rate=(document_hits / total_cases) if total_cases else 0.0,
            citation_page_hit_rate=(page_hits / total_cases) if total_cases else 0.0,
            results=tuple(results),
        )

    def _save_cases(self, cases: list[EvaluationCase]) -> None:
        payload = [self._serialize_case(case) for case in cases]
        self._cases_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _save_runs(self, runs: list[EvaluationRun]) -> None:
        payload = [self._serialize_run(run) for run in runs]
        self._runs_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _serialize_case(self, case: EvaluationCase) -> dict:
        payload = asdict(case)
        payload["expected_terms"] = list(case.expected_terms)
        return payload

    def _deserialize_case(self, payload: dict) -> EvaluationCase:
        return EvaluationCase(
            case_id=payload["case_id"],
            question=payload["question"],
            expected_terms=tuple(payload.get("expected_terms", [])),
            expected_document_id=payload.get("expected_document_id"),
            expected_page_number=payload.get("expected_page_number"),
            notes=payload.get("notes", ""),
        )

    def _serialize_run(self, run: EvaluationRun) -> dict:
        payload = asdict(run)
        payload["results"] = [asdict(result) for result in run.results]
        return payload

    def _deserialize_run(self, payload: dict) -> EvaluationRun:
        return EvaluationRun(
            run_id=payload["run_id"],
            created_at=payload["created_at"],
            total_cases=payload["total_cases"],
            answered_cases=payload["answered_cases"],
            pass_rate=payload["pass_rate"],
            answer_term_hit_rate=payload["answer_term_hit_rate"],
            citation_document_hit_rate=payload["citation_document_hit_rate"],
            citation_page_hit_rate=payload["citation_page_hit_rate"],
            results=tuple(EvaluationCaseResult(**item) for item in payload.get("results", [])),
        )
