from __future__ import annotations

import re
from dataclasses import dataclass

from rag_pdf.domain.models import BoundingBox, Citation, DocumentChunk, SourceDocument
from rag_pdf.domain.ports import Chunker, DocumentParser


@dataclass(frozen=True)
class ExtractedField:
    field_name: str
    value: str
    citation: Citation
    confidence: float


@dataclass(frozen=True)
class ExtractionResult:
    document_id: str
    fields: tuple[ExtractedField, ...]


@dataclass(frozen=True)
class VersionChange:
    change_type: str
    citation: Citation


@dataclass(frozen=True)
class VersionComparisonResult:
    base_document_id: str
    target_document_id: str
    added_changes: tuple[VersionChange, ...]
    removed_changes: tuple[VersionChange, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.added_changes or self.removed_changes)


class PlanAnalysisService:
    def __init__(self, parser: DocumentParser, chunker: Chunker) -> None:
        self._parser = parser
        self._chunker = chunker

    def extract_fields(self, document: SourceDocument) -> ExtractionResult:
        chunks = self._load_chunks(document)
        extractors = [
            ("revision", self._extract_revision, 0.95),
            ("scale", self._extract_scale, 0.9),
            ("drawing_title", self._extract_title, 0.75),
            ("project", self._extract_project, 0.7),
        ]

        fields: list[ExtractedField] = []
        for field_name, extractor, confidence in extractors:
            match = extractor(chunks)
            if match is None:
                continue
            chunk, value = match
            fields.append(
                ExtractedField(
                    field_name=field_name,
                    value=value,
                    citation=self._to_citation(chunk),
                    confidence=confidence,
                )
            )

        return ExtractionResult(document_id=document.document_id, fields=tuple(fields))

    def compare_versions(
        self,
        base_document: SourceDocument,
        target_document: SourceDocument,
        limit: int = 10,
    ) -> VersionComparisonResult:
        base_chunks = self._load_chunks(base_document)
        target_chunks = self._load_chunks(target_document)

        base_by_text = {self._normalize_text(chunk.text): chunk for chunk in base_chunks if self._normalize_text(chunk.text)}
        target_by_text = {
            self._normalize_text(chunk.text): chunk for chunk in target_chunks if self._normalize_text(chunk.text)
        }

        added_keys = [key for key in target_by_text if key not in base_by_text]
        removed_keys = [key for key in base_by_text if key not in target_by_text]

        added_changes = tuple(
            VersionChange(change_type="added", citation=self._to_citation(target_by_text[key]))
            for key in added_keys[:limit]
        )
        removed_changes = tuple(
            VersionChange(change_type="removed", citation=self._to_citation(base_by_text[key]))
            for key in removed_keys[:limit]
        )

        return VersionComparisonResult(
            base_document_id=base_document.document_id,
            target_document_id=target_document.document_id,
            added_changes=added_changes,
            removed_changes=removed_changes,
        )

    def _load_chunks(self, document: SourceDocument) -> list[DocumentChunk]:
        regions = self._parser.parse(document)
        chunks = self._chunker.chunk(document, regions)
        return [chunk for chunk in chunks if chunk.text.strip()]

    def _extract_revision(self, chunks: list[DocumentChunk]) -> tuple[DocumentChunk, str] | None:
        patterns = [
            re.compile(r"\bREV(?:ISION)?[:\s._-]*([A-Z0-9]+)\b", re.IGNORECASE),
            re.compile(r"\bINDICE[:\s._-]*([A-Z0-9]+)\b", re.IGNORECASE),
        ]
        return self._match_first(chunks, patterns, region_kinds={"title_block", "table", "note"})

    def _extract_scale(self, chunks: list[DocumentChunk]) -> tuple[DocumentChunk, str] | None:
        patterns = [
            re.compile(r"\bSCALE[:\s._-]*([A-Z0-9:/]+)\b", re.IGNORECASE),
            re.compile(r"\bECHELLE[:\s._-]*([A-Z0-9:/]+)\b", re.IGNORECASE),
        ]
        return self._match_first(chunks, patterns, region_kinds={"title_block", "table", "note"})

    def _extract_title(self, chunks: list[DocumentChunk]) -> tuple[DocumentChunk, str] | None:
        patterns = [
            re.compile(r"\bDRAWING\s+TITLE[:\s._-]*(.+)$", re.IGNORECASE),
            re.compile(r"\bTITLE[:\s._-]*(.+)$", re.IGNORECASE),
            re.compile(r"\bTITRE[:\s._-]*(.+)$", re.IGNORECASE),
        ]
        match = self._match_first(chunks, patterns, region_kinds={"title_block", "note"})
        if match is not None:
            return match

        title_candidates = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("region_kind") == "title_block" and len(chunk.text.strip()) > 8
        ]
        if not title_candidates:
            return None
        best = max(title_candidates, key=lambda chunk: len(chunk.text))
        return best, " ".join(best.text.split())

    def _extract_project(self, chunks: list[DocumentChunk]) -> tuple[DocumentChunk, str] | None:
        patterns = [
            re.compile(r"\bPROJECT[:\s._-]*(.+)$", re.IGNORECASE),
            re.compile(r"\bPROJET[:\s._-]*(.+)$", re.IGNORECASE),
            re.compile(r"\bCLIENT[:\s._-]*(.+)$", re.IGNORECASE),
        ]
        return self._match_first(chunks, patterns, region_kinds={"title_block", "note"})

    def _match_first(
        self,
        chunks: list[DocumentChunk],
        patterns: list[re.Pattern[str]],
        region_kinds: set[str],
    ) -> tuple[DocumentChunk, str] | None:
        prioritized = sorted(
            chunks,
            key=lambda chunk: (
                chunk.metadata.get("region_kind") not in region_kinds,
                chunk.page_number,
                chunk.chunk_id,
            ),
        )
        for chunk in prioritized:
            text = " ".join(chunk.text.split())
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    value = match.group(1).strip(" :;-")
                    if value:
                        return chunk, value
        return None

    def _to_citation(self, chunk: DocumentChunk) -> Citation:
        return Citation(
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            bbox=chunk.bbox if isinstance(chunk.bbox, BoundingBox) else chunk.bbox,
            excerpt=chunk.text[:240],
        )

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return normalized
