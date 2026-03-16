from __future__ import annotations

from dataclasses import dataclass

from rag_pdf.domain.models import BoundingBox, DocumentChunk, RegionKind, SourceDocument, TextSpan, VisualRegion
from rag_pdf.domain.ports import Chunker


@dataclass(frozen=True)
class RegionGroup:
    group_id: str
    group_kind: RegionKind
    page_number: int
    bbox: BoundingBox
    regions: tuple[VisualRegion, ...]


class LayoutAwareChunker(Chunker):
    """Groups visually-related regions before chunking.

    This favors chunks that mirror how plans are read:
    title blocks together, tables together, annotations grouped by proximity.
    """

    def chunk(self, document: SourceDocument, regions: list[VisualRegion]) -> list[DocumentChunk]:
        region_groups = self._build_groups(document, regions)
        chunks: list[DocumentChunk] = []

        for group in region_groups:
            text = self._build_group_text(group)

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}:{group.group_id}",
                    document_id=document.document_id,
                    page_number=group.page_number,
                    region_id=group.group_id,
                    text=text,
                    bbox=group.bbox,
                    metadata={
                        "region_kind": group.group_kind.value,
                        "source_region_ids": [region.region_id for region in group.regions],
                        "source_region_kinds": [region.kind.value for region in group.regions],
                        "group_size": len(group.regions),
                        **self._merge_metadata(group.regions),
                    },
                )
            )

        return chunks

    def _build_groups(self, document: SourceDocument, regions: list[VisualRegion]) -> list[RegionGroup]:
        groups: list[RegionGroup] = []
        grouped_region_ids: set[str] = set()

        for page_number in sorted({region.page_number for region in regions}):
            page_regions = [region for region in regions if region.page_number == page_number]
            sorted_regions = sorted(page_regions, key=self._sort_key)

            title_blocks = [region for region in sorted_regions if region.kind == RegionKind.TITLE_BLOCK]
            if title_blocks:
                groups.append(self._make_group(document, page_number, RegionKind.TITLE_BLOCK, title_blocks, "title-block"))
                grouped_region_ids.update(region.region_id for region in title_blocks)

            tables = [region for region in sorted_regions if region.kind == RegionKind.TABLE]
            for index, cluster in enumerate(self._cluster_regions(tables, max_vertical_gap=40.0, max_horizontal_gap=80.0), start=1):
                groups.append(self._make_group(document, page_number, RegionKind.TABLE, cluster, f"table-{index}"))
                grouped_region_ids.update(region.region_id for region in cluster)

            callouts = [region for region in sorted_regions if region.kind == RegionKind.CALLOUT]
            for index, cluster in enumerate(self._cluster_regions(callouts, max_vertical_gap=60.0, max_horizontal_gap=120.0), start=1):
                groups.append(self._make_group(document, page_number, RegionKind.CALLOUT, cluster, f"callout-{index}"))
                grouped_region_ids.update(region.region_id for region in cluster)

            notes = [region for region in sorted_regions if region.kind == RegionKind.NOTE]
            for index, cluster in enumerate(self._cluster_regions(notes, max_vertical_gap=70.0, max_horizontal_gap=180.0), start=1):
                groups.append(self._make_group(document, page_number, RegionKind.NOTE, cluster, f"note-{index}"))
                grouped_region_ids.update(region.region_id for region in cluster)

            for index, region in enumerate(sorted_regions, start=1):
                if region.region_id in grouped_region_ids:
                    continue
                groups.append(self._make_group(document, page_number, region.kind, [region], f"region-{index}"))

        return groups

    def _cluster_regions(
        self,
        regions: list[VisualRegion],
        max_vertical_gap: float,
        max_horizontal_gap: float,
    ) -> list[list[VisualRegion]]:
        if not regions:
            return []

        clusters: list[list[VisualRegion]] = []
        for region in sorted(regions, key=self._sort_key):
            attached = False
            for cluster in clusters:
                if any(self._are_related(region, candidate, max_vertical_gap, max_horizontal_gap) for candidate in cluster):
                    cluster.append(region)
                    attached = True
                    break
            if not attached:
                clusters.append([region])
        return clusters

    def _are_related(
        self,
        left: VisualRegion,
        right: VisualRegion,
        max_vertical_gap: float,
        max_horizontal_gap: float,
    ) -> bool:
        vertical_gap = self._axis_gap(left.bbox.y0, left.bbox.y1, right.bbox.y0, right.bbox.y1)
        horizontal_gap = self._axis_gap(left.bbox.x0, left.bbox.x1, right.bbox.x0, right.bbox.x1)
        vertical_stack = horizontal_gap == 0.0 and vertical_gap <= max_vertical_gap
        horizontal_row = vertical_gap == 0.0 and horizontal_gap <= max_horizontal_gap
        close_block = vertical_gap <= max_vertical_gap and horizontal_gap <= max_horizontal_gap
        return vertical_stack or horizontal_row or close_block

    def _axis_gap(self, a0: float, a1: float, b0: float, b1: float) -> float:
        if a1 < b0:
            return b0 - a1
        if b1 < a0:
            return a0 - b1
        return 0.0

    def _make_group(
        self,
        document: SourceDocument,
        page_number: int,
        group_kind: RegionKind,
        regions: list[VisualRegion],
        suffix: str,
    ) -> RegionGroup:
        bbox = regions[0].bbox
        for region in regions[1:]:
            bbox = bbox.union(region.bbox)

        return RegionGroup(
            group_id=f"{document.document_id}:p{page_number}:{suffix}",
            group_kind=group_kind,
            page_number=page_number,
            bbox=bbox,
            regions=tuple(sorted(regions, key=self._sort_key)),
        )

    def _build_group_text(self, group: RegionGroup) -> str:
        lines: list[str] = []
        for region in group.regions:
            region_lines = self._collect_region_lines(region)
            if region_lines:
                lines.extend(region_lines)
            else:
                fallback = region.metadata.get("fallback_text", "")
                if fallback:
                    lines.append(fallback)

        return "\n".join(lines).strip()

    def _collect_region_lines(self, region: VisualRegion) -> list[str]:
        sorted_spans = sorted(region.text_spans, key=lambda span: (span.reading_order, span.bbox.y0, span.bbox.x0))
        return [span.text.strip() for span in sorted_spans if span.text.strip()]

    def _merge_metadata(self, regions: tuple[VisualRegion, ...]) -> dict:
        merged: dict = {}
        drawing_count = max((int(region.metadata.get("drawing_count", 0)) for region in regions), default=0)
        merged["drawing_count"] = drawing_count
        merged["line_count"] = sum(int(region.metadata.get("line_count", len(region.text_spans))) for region in regions)
        return merged

    def _sort_key(self, region: VisualRegion) -> tuple[float, float]:
        return (region.bbox.y0, region.bbox.x0)
