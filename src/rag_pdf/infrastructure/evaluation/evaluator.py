from __future__ import annotations

from rag_pdf.domain.ports import AnswerGenerator
from rag_pdf.infrastructure.evaluation.evaluation_registry import EvaluationCase, EvaluationCaseResult


class RagEvaluator:
    def evaluate_case(self, case: EvaluationCase, answer_text: str, citations: tuple) -> EvaluationCaseResult:
        normalized_answer = answer_text.lower()
        expected_terms_hit = all(term.lower() in normalized_answer for term in case.expected_terms)

        citation_document_hit = True
        if case.expected_document_id:
            citation_document_hit = any(
                citation.document_id == case.expected_document_id
                for citation in citations
            )

        citation_page_hit = True
        if case.expected_page_number is not None:
            citation_page_hit = any(
                citation.page_number == case.expected_page_number
                for citation in citations
            )

        passed = expected_terms_hit and citation_document_hit and citation_page_hit

        return EvaluationCaseResult(
            case_id=case.case_id,
            question=case.question,
            answer_text=answer_text,
            citations_count=len(citations),
            answer_contains_expected_terms=expected_terms_hit,
            citation_document_hit=citation_document_hit,
            citation_page_hit=citation_page_hit,
            passed=passed,
        )
