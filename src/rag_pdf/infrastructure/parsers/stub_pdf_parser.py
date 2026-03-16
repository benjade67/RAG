from __future__ import annotations

from rag_pdf.domain.models import RegionKind, SourceDocument, VisualRegion
from rag_pdf.domain.ports import DocumentParser


class StubPdfParser(DocumentParser):
    """Placeholder parser.

    This is where a PDF parser with text coordinates and layout extraction
    will later be connected.
    """

    def parse(self, document: SourceDocument) -> list[VisualRegion]:
        return [
            VisualRegion(
                region_id=f"{document.document_id}:page:1",
                page_number=1,
                kind=RegionKind.PAGE,
                bbox=document.metadata["default_bbox"],
                metadata={"source_path": document.file_path},
            )
        ]
