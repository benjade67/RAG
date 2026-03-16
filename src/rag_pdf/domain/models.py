from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RegionKind(str, Enum):
    PAGE = "page"
    TITLE_BLOCK = "title_block"
    TABLE = "table"
    CALLOUT = "callout"
    NOTE = "note"
    LEGEND = "legend"
    FREE_TEXT = "free_text"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


@dataclass(frozen=True)
class TextSpan:
    text: str
    bbox: BoundingBox
    reading_order: int


@dataclass(frozen=True)
class VisualRegion:
    region_id: str
    page_number: int
    kind: RegionKind
    bbox: BoundingBox
    text_spans: tuple[TextSpan, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    file_path: str
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    page_number: int
    region_id: str
    text: str
    bbox: BoundingBox
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    document_id: str
    page_number: int
    chunk_id: str
    bbox: BoundingBox
    excerpt: str


@dataclass(frozen=True)
class RetrievedPassage:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: tuple[Citation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
