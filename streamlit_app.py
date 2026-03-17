from __future__ import annotations

import hashlib
import io

import streamlit as st
from PIL import Image, ImageDraw

from rag_pdf.application.analysis import ExtractionResult, PlanAnalysisService
from rag_pdf.bootstrap import build_app
from rag_pdf.config import load_environment
from rag_pdf.domain.models import Citation, SourceDocument
from rag_pdf.infrastructure.catalog.plan_catalog import PlanCatalog, PlanRecord
from rag_pdf.infrastructure.generation.grounded_generator import GroundedAnswerGenerator

try:
    import fitz
except ImportError:  # pragma: no cover - depends on local environment
    fitz = None


load_environment()

st.set_page_config(
    page_title="RAG Plans PDF",
    layout="wide",
)


def _build_document(plan: PlanRecord) -> SourceDocument:
    return SourceDocument(
        document_id=plan.plan_id,
        file_path=plan.file_path,
        checksum=plan.checksum,
        metadata={"original_name": plan.filename},
    )


def _catalog() -> PlanCatalog:
    return PlanCatalog()


def _analysis_service() -> PlanAnalysisService:
    ingestion, _ = build_app()
    return PlanAnalysisService(parser=ingestion.parser, chunker=ingestion.chunker)


def _scope_signature(plans: list[PlanRecord]) -> str:
    raw = "|".join(f"{plan.plan_id}:{plan.checksum}" for plan in plans)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _active_document_ids(plans: list[PlanRecord]) -> list[str]:
    return [plan.plan_id for plan in plans]


def _render_sidebar() -> tuple[int, bool, float]:
    catalog = _catalog()
    with st.sidebar:
        st.title("RAG Plans PDF")
        st.caption("Interrogez une base de plans AutoCAD PDF avec citations.")
        top_k = st.slider("Nombre de passages", min_value=1, max_value=10, value=5)
        allow_fallback = st.toggle("Fallback local si LLM KO", value=True)
        preview_zoom = st.slider("Zoom apercu", min_value=1.0, max_value=3.0, value=1.6, step=0.1)
        st.markdown("Le backend utilise automatiquement `Mistral`, `Ollama` ou le fallback local selon votre `.env`.")
        st.caption(f"Plans enregistres: {len(catalog.list_plans())}")
        st.page_link("streamlit_app.py", label="Assistant", icon=":material/chat:")
        st.page_link("pages/1_Admin.py", label="Administration", icon=":material/admin_panel_settings:")
    return top_k, allow_fallback, preview_zoom


def _render_citations(citations: tuple[Citation, ...]) -> None:
    st.subheader("Citations")
    if not citations:
        st.info("Aucune citation disponible.")
        return

    for index, citation in enumerate(citations, start=1):
        st.markdown(
            "\n".join(
                [
                    f"**[{index}] {citation.document_id}** page **{citation.page_number}**",
                    f"`chunk={citation.chunk_id}`",
                    f"`bbox=({citation.bbox.x0:.1f}, {citation.bbox.y0:.1f}, {citation.bbox.x1:.1f}, {citation.bbox.y1:.1f})`",
                    citation.excerpt or "_Extrait vide_",
                ]
            )
        )
        st.divider()


def _render_passages(passages) -> None:
    st.subheader("Passages recuperes")
    if not passages:
        st.info("Aucun passage retrouve.")
        return

    for rank, passage in enumerate(passages, start=1):
        chunk = passage.chunk
        with st.expander(
            f"#{rank} score={passage.score:.2f} page={chunk.page_number} kind={chunk.metadata.get('region_kind', 'unknown')}"
        ):
            st.code(chunk.text or "", language="text")
            st.caption(
                f"chunk={chunk.chunk_id} | bbox=({chunk.bbox.x0:.1f}, {chunk.bbox.y0:.1f}, {chunk.bbox.x1:.1f}, {chunk.bbox.y1:.1f})"
            )


def _render_extraction_results(results: list[tuple[PlanRecord, ExtractionResult]], selected_fields: list[str]) -> None:
    st.subheader("Extraction de valeurs")
    if not results:
        st.info("Aucun champ n a pu etre extrait sur la base courante.")
        return

    displayed = 0
    for plan, result in results:
        fields = [field for field in result.fields if field.field_name in selected_fields]
        if not fields:
            continue
        displayed += 1
        with st.expander(plan.filename, expanded=False):
            for field in fields:
                st.markdown(
                    "\n".join(
                        [
                            f"**{field.field_name}**: `{field.value}`",
                            f"confiance: `{field.confidence:.2f}`",
                            f"citation: `{field.citation.document_id} p.{field.citation.page_number} {field.citation.chunk_id}`",
                        ]
                    )
                )
                st.caption(field.citation.excerpt)
                st.divider()

    if displayed == 0:
        st.info("Aucun des champs selectionnes n a ete trouve.")


@st.cache_data(show_spinner=False)
def _render_page_image(pdf_path: str, page_number: int, zoom: float) -> Image.Image:
    if fitz is None:
        raise RuntimeError("PyMuPDF est requis pour afficher l apercu du PDF.")

    with fitz.open(pdf_path) as pdf:
        page = pdf.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


def _draw_citation_boxes(
    base_image: Image.Image,
    citations: list[Citation],
    page_number: int,
    zoom: float,
) -> Image.Image:
    image = base_image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    palette = [
        (220, 38, 38, 90),
        (37, 99, 235, 90),
        (22, 163, 74, 90),
        (202, 138, 4, 90),
        (147, 51, 234, 90),
    ]

    page_citations = [citation for citation in citations if citation.page_number == page_number]
    for index, citation in enumerate(page_citations, start=1):
        fill = palette[(index - 1) % len(palette)]
        outline = fill[:3] + (255,)
        bbox = citation.bbox
        rectangle = [
            bbox.x0 * zoom,
            bbox.y0 * zoom,
            bbox.x1 * zoom,
            bbox.y1 * zoom,
        ]
        draw.rectangle(rectangle, outline=outline, width=3, fill=fill)
        label_box = [rectangle[0], max(0, rectangle[1] - 24), rectangle[0] + 28, rectangle[1]]
        draw.rectangle(label_box, fill=outline)
        draw.text((label_box[0] + 8, label_box[1] + 4), str(index), fill=(255, 255, 255, 255))

    return image


def _render_pdf_preview(plan_by_id: dict[str, PlanRecord], citations: tuple[Citation, ...], zoom: float) -> None:
    st.subheader("Apercu du PDF")
    if not citations:
        st.info("Lancez une question ou une extraction pour afficher les zones citees.")
        return

    cited_documents = sorted({citation.document_id for citation in citations})
    selected_document_id = st.selectbox("Document a afficher", options=cited_documents, index=0)
    selected_plan = plan_by_id.get(selected_document_id)

    if selected_plan is None:
        st.warning("Le document cite n est plus disponible dans la base.")
        return

    document_citations = tuple(citation for citation in citations if citation.document_id == selected_document_id)
    cited_pages = sorted({citation.page_number for citation in document_citations})
    selected_page = st.selectbox("Page a afficher", options=cited_pages, index=0)

    preview_image = _render_page_image(selected_plan.file_path, selected_page, zoom)
    highlighted_image = _draw_citation_boxes(preview_image, list(document_citations), selected_page, zoom)
    st.image(highlighted_image, caption=f"{selected_plan.filename} - page {selected_page}", use_container_width=True)
    st.caption("Chaque encadre numerote correspond a une citation listee ci-dessous.")


def _looks_like_scanned_pdf(passages) -> bool:
    return not any((passage.chunk.text or "").strip() for passage in passages)


def _ensure_scope_indexed(plans: list[PlanRecord]) -> tuple[object, object]:
    signature = _scope_signature(plans)
    if st.session_state.get("ingested_scope_signature") == signature:
        return st.session_state["services"]

    st.session_state["services"] = build_app()
    ingestion, qa = st.session_state["services"]

    for plan in plans:
        ingestion.ingest(_build_document(plan))

    st.session_state["ingested_scope_signature"] = signature
    return ingestion, qa


def _extract_all_fields(active_plans: list[PlanRecord], selected_fields: list[str]) -> tuple[list[tuple[PlanRecord, ExtractionResult]], tuple[Citation, ...]]:
    analysis = _analysis_service()
    results: list[tuple[PlanRecord, ExtractionResult]] = []
    citations: list[Citation] = []

    for plan in active_plans:
        result = analysis.extract_fields(_build_document(plan))
        filtered_fields = tuple(field for field in result.fields if field.field_name in selected_fields)
        if not filtered_fields:
            continue
        filtered_result = ExtractionResult(document_id=result.document_id, fields=filtered_fields)
        results.append((plan, filtered_result))
        citations.extend(field.citation for field in filtered_fields)

    return results, tuple(citations)


def main() -> None:
    top_k, allow_fallback, preview_zoom = _render_sidebar()
    catalog = _catalog()
    plans = catalog.list_plans()

    if "services" not in st.session_state:
        st.session_state["services"] = build_app()
        st.session_state["ingested_scope_signature"] = None

    st.title("Assistant RAG pour base de plans PDF")
    st.write("Interrogez toute la base ou extrayez des valeurs structurees avec citations.")

    if not plans:
        st.info(
            "Aucun plan n est enregistre. Utilisez la page Administration "
            "pour synchroniser un dossier de PDF."
        )
        return

    active_plans = plans
    mode = st.radio(
        "Mode",
        options=["Question / Recherche", "Extraction de valeurs"],
        horizontal=True,
    )

    left, right = st.columns([1.35, 1.05], gap="large")

    with left:
        if mode == "Question / Recherche":
            question = st.text_area(
                "Question",
                value="",
                height=120,
                placeholder="Ex: Quelle est la revision du plan ?",
            )

            if st.button("Interroger la base", type="primary", use_container_width=True):
                if not question.strip():
                    st.warning("Saisissez une question.")
                    st.stop()

                try:
                    with st.spinner("Indexation de la base et interrogation en cours..."):
                        _, qa = _ensure_scope_indexed(active_plans)
                        passages = qa.retrieval.retrieve(
                            question,
                            top_k=top_k,
                            document_ids=_active_document_ids(active_plans),
                        )
                        answer = qa.generator.generate(question, passages)
                except RuntimeError as exc:
                    st.error(str(exc))
                    if "CUDA error" in str(exc):
                        st.info(
                            "Ollama a plante cote GPU. Vous pouvez relancer Ollama, utiliser un modele plus petit, "
                            "ou desactiver le GPU dans Ollama si besoin."
                        )

                    if allow_fallback:
                        with st.spinner("Generation de secours en cours..."):
                            _, qa = _ensure_scope_indexed(active_plans)
                            passages = qa.retrieval.retrieve(
                                question,
                                top_k=top_k,
                                document_ids=_active_document_ids(active_plans),
                            )
                            answer = GroundedAnswerGenerator().generate(question, passages)
                        st.warning("Reponse produite avec le fallback local, sans LLM.")
                    else:
                        return

                st.subheader("Reponse")
                st.write(answer.text)
                if _looks_like_scanned_pdf(passages):
                    st.warning(
                        "Ce PDF semble contenir peu ou pas de texte exploitable. "
                        "Pour un PDF issu d une image ou d un screenshot, il faudra probablement ajouter de l OCR."
                    )
                _render_citations(answer.citations)

                st.session_state["latest_passages"] = passages
                st.session_state["latest_citations"] = answer.citations
                st.session_state["latest_plans"] = {plan.plan_id: plan for plan in active_plans}
                st.session_state["latest_generator"] = answer.metadata.get("generator", "fallback")
                st.session_state["latest_model"] = answer.metadata.get("model", "local")

        else:
            selected_fields = st.multiselect(
                "Champs a extraire",
                options=["revision", "scale", "drawing_title", "project"],
                default=["revision", "scale", "drawing_title"],
            )

            if st.button("Extraire sur toute la base", type="primary", use_container_width=True):
                if not selected_fields:
                    st.warning("Selectionnez au moins un champ.")
                    st.stop()

                with st.spinner("Extraction structuree en cours..."):
                    results, citations = _extract_all_fields(active_plans, selected_fields)

                _render_extraction_results(results, selected_fields)
                st.session_state["latest_passages"] = []
                st.session_state["latest_citations"] = citations
                st.session_state["latest_plans"] = {plan.plan_id: plan for plan in active_plans}
                st.session_state["latest_generator"] = "analysis_service"
                st.session_state["latest_model"] = "regex-layout"

    with right:
        st.subheader("Base courante")
        st.write(f"Plans dans la base active: {len(active_plans)}")
        for plan in active_plans[:8]:
            status = plan.index_status.state
            chunk_count = plan.index_status.chunk_count
            st.caption(f"{plan.filename} | {status} | chunks={chunk_count}")
        if len(active_plans) > 8:
            st.caption(f"... et {len(active_plans) - 8} autre(s)")

        with st.expander("Statut d indexation", expanded=False):
            for plan in active_plans:
                st.markdown(
                    "\n".join(
                        [
                            f"**{plan.filename}**",
                            f"etat: `{plan.index_status.state}`",
                            f"chunks: `{plan.index_status.chunk_count}`",
                            f"backend: `{plan.index_status.embedding_backend or 'n/a'}`",
                            f"modele: `{plan.index_status.embedding_model or 'n/a'}`",
                            f"date: `{plan.index_status.last_indexed_at or 'n/a'}`",
                            f"erreur: `{plan.index_status.error_message or 'aucune'}`",
                        ]
                    )
                )
                st.divider()

        if "latest_citations" in st.session_state:
            st.metric("Generateur", st.session_state.get("latest_generator", "fallback"))
            st.caption(f"Modele: {st.session_state.get('latest_model', 'local')}")
            _render_pdf_preview(
                st.session_state["latest_plans"],
                st.session_state["latest_citations"],
                preview_zoom,
            )

    if st.session_state.get("latest_passages"):
        _render_passages(st.session_state["latest_passages"])


if __name__ == "__main__":
    main()
