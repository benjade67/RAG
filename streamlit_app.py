from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from rag_pdf.bootstrap import build_app
from rag_pdf.config import load_environment
from rag_pdf.domain.models import Citation, SourceDocument
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


def _persist_uploaded_pdf(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def _build_document(pdf_path: Path, original_name: str) -> SourceDocument:
    checksum = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    document_id = Path(original_name).stem.upper().replace(" ", "_")
    return SourceDocument(
        document_id=document_id,
        file_path=str(pdf_path),
        checksum=checksum,
        metadata={"original_name": original_name},
    )


def _render_sidebar() -> tuple[int, str, bool, float]:
    with st.sidebar:
        st.title("RAG Plans PDF")
        st.caption("Interrogez un plan AutoCAD PDF avec citations.")
        top_k = st.slider("Nombre de passages", min_value=1, max_value=10, value=5)
        allow_fallback = st.toggle("Fallback local si LLM KO", value=True)
        preview_zoom = st.slider("Zoom apercu", min_value=1.0, max_value=3.0, value=1.6, step=0.1)
        default_question = st.selectbox(
            "Question rapide",
            options=[
                "Quelle est la revision du plan ?",
                "Quelle est la designation de l'item 3 ?",
                "Quelles notes apparaissent sur le plan ?",
                "Quelle information cle contient le cartouche ?",
            ],
            index=0,
        )
        st.markdown("Le backend utilise automatiquement `Mistral`, `Ollama` ou le fallback local selon votre `.env`.")
    return top_k, default_question, allow_fallback, preview_zoom


def _render_citations(answer) -> None:
    st.subheader("Citations")
    if not answer.citations:
        st.info("Aucune citation disponible.")
        return

    for citation in answer.citations:
        st.markdown(
            "\n".join(
                [
                    f"**{citation.document_id}** page **{citation.page_number}**",
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


@st.cache_data(show_spinner=False)
def _render_page_image(pdf_path: str, page_number: int, zoom: float) -> Image.Image:
    if fitz is None:
        raise RuntimeError("PyMuPDF est requis pour afficher l'aperçu du PDF.")

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


def _render_pdf_preview(pdf_path: str, citations: tuple[Citation, ...], zoom: float) -> None:
    st.subheader("Aperçu du PDF")
    if not citations:
        st.info("Posez une question pour afficher les zones citées.")
        return

    cited_pages = sorted({citation.page_number for citation in citations})
    selected_page = st.selectbox("Page à afficher", options=cited_pages, index=0)
    preview_image = _render_page_image(pdf_path, selected_page, zoom)
    highlighted_image = _draw_citation_boxes(preview_image, list(citations), selected_page, zoom)
    st.image(highlighted_image, caption=f"Page {selected_page} avec zones citées", use_container_width=True)

    st.caption("Chaque encadré numéroté correspond à une citation listée ci-dessous.")


def _looks_like_scanned_pdf(passages) -> bool:
    return not any((passage.chunk.text or "").strip() for passage in passages)


def main() -> None:
    top_k, default_question, allow_fallback, preview_zoom = _render_sidebar()

    if "services" not in st.session_state:
        st.session_state["services"] = build_app()
        st.session_state["ingested_checksum"] = None

    ingestion, qa = st.session_state["services"]

    st.title("Assistant RAG pour plans PDF")
    st.write("Chargez un plan PDF, posez une question, puis inspectez la reponse citee et les passages recuperes.")

    uploaded_file = st.file_uploader("Plan PDF", type=["pdf"])
    question = st.text_area("Question", value=default_question, height=100)

    if uploaded_file is None:
        st.info("Ajoutez un PDF pour commencer.")
        return

    left, right = st.columns([1.4, 1.0], gap="large")

    with left:
        if st.button("Interroger le plan", type="primary", use_container_width=True):
            pdf_path = _persist_uploaded_pdf(uploaded_file)
            document = _build_document(pdf_path, uploaded_file.name)

            try:
                with st.spinner("Ingestion du plan et interrogation en cours..."):
                    if st.session_state.get("ingested_checksum") != document.checksum:
                        st.session_state["services"] = build_app()
                        ingestion, qa = st.session_state["services"]
                        ingestion.ingest(document)
                        st.session_state["ingested_checksum"] = document.checksum
                    passages = qa.retrieval.retrieve(question, top_k=top_k)
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
                        passages = qa.retrieval.retrieve(question, top_k=top_k)
                        answer = GroundedAnswerGenerator().generate(question, passages)
                    st.warning("Reponse produite avec le fallback local, sans LLM.")
                else:
                    return

            st.subheader("Reponse")
            st.write(answer.text)
            if _looks_like_scanned_pdf(passages):
                st.warning(
                    "Ce PDF semble contenir peu ou pas de texte exploitable. "
                    "Pour un PDF issu d'une image ou d'un screenshot, il faudra probablement ajouter de l'OCR."
                )
            _render_citations(answer)

            st.session_state["latest_passages"] = passages
            st.session_state["latest_answer"] = answer
            st.session_state["latest_filename"] = uploaded_file.name
            st.session_state["latest_pdf_path"] = str(pdf_path)

    with right:
        st.subheader("Document courant")
        st.write(uploaded_file.name)
        st.caption(f"{uploaded_file.size / 1024:.1f} KB")

        if "latest_answer" in st.session_state and st.session_state.get("latest_filename") == uploaded_file.name:
            generator = st.session_state["latest_answer"].metadata.get("generator", "fallback")
            model = st.session_state["latest_answer"].metadata.get("model", "local")
            st.metric("Generateur", generator)
            st.caption(f"Modele: {model}")
            _render_pdf_preview(
                st.session_state["latest_pdf_path"],
                st.session_state["latest_answer"].citations,
                preview_zoom,
            )

    if "latest_passages" in st.session_state and st.session_state.get("latest_filename") == uploaded_file.name:
        _render_passages(st.session_state["latest_passages"])


if __name__ == "__main__":
    main()
