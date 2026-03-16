from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_pdf.domain.models import BoundingBox, RegionKind, SourceDocument, TextSpan, VisualRegion
from rag_pdf.domain.ports import DocumentParser

try:
    import fitz
except ImportError:  # pragma: no cover - depends on local environment
    fitz = None


@dataclass(frozen=True)
class RegionClassifier:
    """Heuristics for mapping PDF text blocks to business-relevant regions."""

    title_block_keywords: tuple[str, ...] = (
        "title",
        "drawing",
        "plan",
        "sheet",
        "revision",
        "rev",
        "scale",
        "project",
        "client",
        "dwg",
    )
    note_keywords: tuple[str, ...] = (
        "note",
        "notes",
        "remark",
        "remarks",
        "instruction",
        "warning",
    )

    def classify(self, text: str, bbox: BoundingBox, page_bbox: BoundingBox) -> RegionKind:
        raw_text = text.strip()
        normalized = " ".join(raw_text.lower().split())
        if not normalized:
            return RegionKind.UNKNOWN

        if self._looks_like_table(raw_text):
            return RegionKind.TABLE
        if self._looks_like_note(normalized):
            return RegionKind.NOTE
        if self._looks_like_title_block(normalized, bbox, page_bbox):
            return RegionKind.TITLE_BLOCK
        if self._looks_like_callout(normalized):
            return RegionKind.CALLOUT
        return RegionKind.FREE_TEXT

    def _looks_like_title_block(
        self,
        text: str,
        bbox: BoundingBox,
        page_bbox: BoundingBox,
    ) -> bool:
        page_width = max(page_bbox.x1 - page_bbox.x0, 1.0)
        page_height = max(page_bbox.y1 - page_bbox.y0, 1.0)
        center_x = (bbox.x0 + bbox.x1) / 2
        center_y = (bbox.y0 + bbox.y1) / 2
        in_bottom_band = center_y >= page_bbox.y0 + page_height * 0.7
        in_right_band = center_x >= page_bbox.x0 + page_width * 0.65
        keyword_hits = sum(1 for keyword in self.title_block_keywords if keyword in text)
        return (in_bottom_band and in_right_band) or keyword_hits >= 2

    def _looks_like_table(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        lowered = text.lower()
        many_short_lines = sum(1 for line in lines if len(line.split()) <= 4) >= max(2, len(lines) // 2)
        repeated_numeric_markers = sum(1 for line in lines if any(char.isdigit() for char in line)) >= max(2, len(lines) // 2)
        has_common_headers = any(
            token in lowered
            for token in ("qty", "item", "description", "designation", "repere")
        )
        return (many_short_lines and repeated_numeric_markers) or has_common_headers

    def _looks_like_note(self, text: str) -> bool:
        return any(keyword in text for keyword in self.note_keywords)

    def _looks_like_callout(self, text: str) -> bool:
        tokens = text.split()
        if len(tokens) > 6:
            return False
        uppercase_ratio = sum(1 for char in text if char.isupper()) / max(len(text), 1)
        has_marker = any(char.isdigit() for char in text) or "-" in text
        return uppercase_ratio >= 0.3 and has_marker


class PyMuPdfParser(DocumentParser):
    """Parse digital PDF plans into text regions with coordinates."""

    def __init__(self, classifier: RegionClassifier | None = None) -> None:
        self._classifier = classifier or RegionClassifier()

    def parse(self, document: SourceDocument) -> list[VisualRegion]:
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF n'est pas installe. Installez les dependances avec `pip install -e .`."
            )

        pdf_path = Path(document.file_path)
        with fitz.open(pdf_path) as pdf:
            regions: list[VisualRegion] = []
            for page_index, page in enumerate(pdf, start=1):
                page_bbox = BoundingBox(0.0, 0.0, float(page.rect.width), float(page.rect.height))
                page_regions = self._parse_page(
                    document=document,
                    page=page,
                    page_number=page_index,
                    page_bbox=page_bbox,
                )
                regions.extend(page_regions)
            return regions

    def _parse_page(
        self,
        document: SourceDocument,
        page: Any,
        page_number: int,
        page_bbox: BoundingBox,
    ) -> list[VisualRegion]:
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        regions: list[VisualRegion] = []

        for block_index, block in enumerate(blocks):
            if block.get("type") != 0:
                continue

            text_spans = self._extract_text_spans(block)
            text = "\n".join(span.text for span in text_spans if span.text).strip()
            if not text:
                continue

            bbox = self._to_bbox(block["bbox"])
            region_kind = self._classifier.classify(text=text, bbox=bbox, page_bbox=page_bbox)

            regions.append(
                VisualRegion(
                    region_id=f"{document.document_id}:p{page_number}:b{block_index}",
                    page_number=page_number,
                    kind=region_kind,
                    bbox=bbox,
                    text_spans=tuple(text_spans),
                    metadata={
                        "source_path": document.file_path,
                        "page_width": page_bbox.x1,
                        "page_height": page_bbox.y1,
                        "block_index": block_index,
                        "line_count": len(text_spans),
                        "drawing_count": self._safe_drawing_count(page),
                    },
                )
            )

        if not regions:
            regions.append(
                VisualRegion(
                    region_id=f"{document.document_id}:p{page_number}:empty",
                    page_number=page_number,
                    kind=RegionKind.PAGE,
                    bbox=page_bbox,
                    metadata={
                        "source_path": document.file_path,
                        "fallback_text": "",
                        "drawing_count": self._safe_drawing_count(page),
                    },
                )
            )

        return regions

    def _extract_text_spans(self, block: dict[str, Any]) -> list[TextSpan]:
        extracted: list[TextSpan] = []
        reading_order = 0

        for line in block.get("lines", []):
            line_text_parts: list[str] = []
            line_bbox = None

            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    line_text_parts.append(text)
                span_bbox = span.get("bbox")
                if span_bbox is not None and line_bbox is None:
                    line_bbox = span_bbox

            merged_line = "".join(line_text_parts).strip()
            if not merged_line or line_bbox is None:
                continue

            extracted.append(
                TextSpan(
                    text=merged_line,
                    bbox=self._to_bbox(line_bbox),
                    reading_order=reading_order,
                )
            )
            reading_order += 1

        return extracted

    def _safe_drawing_count(self, page: Any) -> int:
        try:
            drawings = page.get_drawings()
        except Exception:  # pragma: no cover - defensive around backend specifics
            return 0
        return len(drawings)

    def _to_bbox(self, raw_bbox: tuple[float, float, float, float] | list[float]) -> BoundingBox:
        x0, y0, x1, y1 = raw_bbox
        return BoundingBox(float(x0), float(y0), float(x1), float(y1))
