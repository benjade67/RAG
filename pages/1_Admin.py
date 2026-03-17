from __future__ import annotations

import os

import streamlit as st

from rag_pdf.application.analysis import PlanAnalysisService
from rag_pdf.bootstrap import build_app
from rag_pdf.config import get_project_root, load_environment, update_environment
from rag_pdf.domain.models import SourceDocument
from rag_pdf.infrastructure.catalog.plan_catalog import PlanCatalog, PlanRecord
from rag_pdf.infrastructure.evaluation.evaluation_registry import EvaluationRegistry
from rag_pdf.infrastructure.evaluation.evaluator import RagEvaluator
from rag_pdf.infrastructure.prompts.prompt_registry import PromptRegistry


load_environment()

st.set_page_config(
    page_title="Administration RAG Plans PDF",
    layout="wide",
)


def _catalog() -> PlanCatalog:
    return PlanCatalog()


def _prompt_registry() -> PromptRegistry:
    return PromptRegistry()


def _evaluation_registry() -> EvaluationRegistry:
    return EvaluationRegistry()


def _build_document(plan: PlanRecord) -> SourceDocument:
    return SourceDocument(
        document_id=plan.plan_id,
        file_path=plan.file_path,
        checksum=plan.checksum,
        metadata={"original_name": plan.filename},
    )


def _authorized_for_admin() -> bool:
    required_token = os.getenv("RAG_ADMIN_TOKEN", "").strip()
    if not required_token:
        return True
    entered_token = st.session_state.get("admin_token", "")
    return entered_token == required_token


def _embedding_runtime_config() -> tuple[str, str]:
    backend = os.getenv("EMBEDDING_BACKEND", "stub")
    if backend == "ollama":
        return backend, os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    if backend == "mistral":
        return backend, os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    if backend == "sentence_transformers":
        return backend, os.getenv(
            "SENTENCE_TRANSFORMERS_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    return backend, "stub"


def _reset_runtime_state() -> None:
    st.session_state["services"] = build_app()
    st.session_state["ingested_scope_signature"] = None


def _analysis_service() -> PlanAnalysisService:
    ingestion, _ = build_app()
    return PlanAnalysisService(parser=ingestion.parser, chunker=ingestion.chunker)


def _reindex_plans(catalog: PlanCatalog, plans: list[PlanRecord]) -> None:
    if not plans:
        return

    ingestion, _ = build_app()
    backend_name, model_name = _embedding_runtime_config()

    for plan in plans:
        try:
            chunk_count = ingestion.reingest(_build_document(plan))
            catalog.mark_indexed(
                plan_id=plan.plan_id,
                chunk_count=chunk_count,
                embedding_backend=backend_name,
                embedding_model=model_name,
            )
        except RuntimeError as exc:
            catalog.mark_index_error(
                plan_id=plan.plan_id,
                embedding_backend=backend_name,
                embedding_model=model_name,
                error_message=str(exc),
            )
            raise


def _delete_from_index(plan_ids: list[str]) -> None:
    if not plan_ids:
        return
    ingestion, _ = build_app()
    ingestion.index.delete(plan_ids)


def _render_sidebar_links() -> None:
    with st.sidebar:
        st.title("Navigation")
        st.page_link("streamlit_app.py", label="Assistant", icon=":material/chat:")
        st.page_link("pages/1_Admin.py", label="Administration", icon=":material/admin_panel_settings:")


def _run_evaluation(catalog: PlanCatalog) -> None:
    registry = _evaluation_registry()
    cases = registry.list_cases()
    plans = catalog.list_plans()
    if not cases:
        raise RuntimeError("Aucun cas d evaluation n est configure.")
    if not plans:
        raise RuntimeError("Aucun plan n est disponible dans la base.")

    ingestion, qa = build_app()
    evaluator = RagEvaluator()

    for plan in plans:
        ingestion.ingest(_build_document(plan))

    document_ids = [plan.plan_id for plan in plans]
    results = []
    for case in cases:
        answer = qa.answer(case.question, top_k=5, document_ids=document_ids)
        results.append(evaluator.evaluate_case(case, answer.text, answer.citations))

    run = registry.build_run(results)
    registry.save_run(run)


def _version_label(version: PlanRecord) -> str:
    revision = version.revision_label or "n/a"
    duplicate_suffix = " | doublon exact" if version.duplicate_of_version_id else ""
    current_suffix = " | active" if version.is_current else ""
    archived_suffix = " | archivee" if version.is_archived else ""
    return f"{version.filename} | rev={revision}{current_suffix}{duplicate_suffix}{archived_suffix}"


def _render_config_section() -> None:
    llm_col, prompt_col = st.columns(2, gap="large")

    with llm_col:
        st.subheader("Configuration LLM")
        backend = st.selectbox(
            "Backend LLM",
            options=["auto", "ollama", "mistral", "fallback"],
            index=["auto", "ollama", "mistral", "fallback"].index(os.getenv("LLM_BACKEND", "auto")),
        )
        embedding_backend = st.selectbox(
            "Backend embeddings",
            options=["stub", "sentence_transformers", "ollama", "mistral"],
            index=["stub", "sentence_transformers", "ollama", "mistral"].index(
                os.getenv("EMBEDDING_BACKEND", "stub")
            ),
        )
        ollama_model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
        ollama_embed_model = st.text_input(
            "Ollama embed model",
            value=os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma"),
        )
        ollama_base_url = st.text_input(
            "Ollama URL",
            value=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        )
        ollama_timeout = st.number_input(
            "Ollama timeout",
            min_value=30,
            max_value=600,
            value=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
            step=10,
        )
        mistral_model = st.text_input(
            "Mistral model",
            value=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        )
        mistral_embed_model = st.text_input(
            "Mistral embed model",
            value=os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        )
        llm_temperature = st.number_input(
            "Temperature LLM",
            min_value=0.0,
            max_value=1.5,
            value=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            step=0.1,
        )
        sentence_transformers_model = st.text_input(
            "Sentence-transformers model",
            value=os.getenv(
                "SENTENCE_TRANSFORMERS_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
        )

        if st.button("Enregistrer la configuration", use_container_width=True):
            update_environment(
                {
                    "LLM_BACKEND": backend,
                    "EMBEDDING_BACKEND": embedding_backend,
                    "OLLAMA_MODEL": ollama_model.strip(),
                    "OLLAMA_EMBED_MODEL": ollama_embed_model.strip(),
                    "OLLAMA_BASE_URL": ollama_base_url.strip(),
                    "OLLAMA_TIMEOUT_SECONDS": str(ollama_timeout),
                    "MISTRAL_MODEL": mistral_model.strip(),
                    "MISTRAL_EMBED_MODEL": mistral_embed_model.strip(),
                    "LLM_TEMPERATURE": str(llm_temperature),
                    "SENTENCE_TRANSFORMERS_MODEL": sentence_transformers_model.strip(),
                }
            )
            _reset_runtime_state()
            st.success("Configuration enregistree dans le .env.")

    with prompt_col:
        st.subheader("Prompts")
        registry = _prompt_registry()
        active_prompt = registry.get_active_prompt()
        st.caption(f"Prompt actif: {active_prompt.version_id} - {active_prompt.name}")

        prompt_versions = registry.list_prompts()
        selected_version_id = st.selectbox(
            "Versions de prompt",
            options=[prompt.version_id for prompt in prompt_versions],
            index=next(
                index for index, prompt in enumerate(prompt_versions) if prompt.version_id == active_prompt.version_id
            ),
        )
        selected_prompt = next(prompt for prompt in prompt_versions if prompt.version_id == selected_version_id)
        st.text_area("Contenu du prompt selectionne", value=selected_prompt.content, height=180, disabled=True)

        if st.button("Activer cette version", use_container_width=True):
            registry.activate_prompt(selected_version_id)
            _reset_runtime_state()
            st.success(f"Prompt {selected_version_id} active.")

        new_prompt_name = st.text_input("Nom de la nouvelle version", value="")
        new_prompt_content = st.text_area("Nouveau prompt systeme", value=active_prompt.content, height=220)
        if st.button("Creer une nouvelle version de prompt", use_container_width=True):
            registry.create_prompt_version(
                name=new_prompt_name or "Nouvelle version",
                content=new_prompt_content,
                activate=True,
            )
            _reset_runtime_state()
            st.success("Nouvelle version de prompt creee et activee.")


def _render_knowledge_base_admin(catalog: PlanCatalog) -> None:
    st.subheader("Base de connaissance")
    managed_plans = catalog.list_managed_plans()

    if not managed_plans:
        st.info("Aucun plan metier n est encore enregistre.")
        return

    st.caption(
        f"Plans metier: {len(managed_plans)} | versions: {sum(len(plan.versions) for plan in managed_plans)}"
    )

    selected_plan_id = st.selectbox(
        "Plan metier",
        options=[plan.logical_plan_id for plan in managed_plans],
        format_func=lambda logical_plan_id: next(
            (
                f"{plan.display_name} ({len(plan.versions)} version(s))"
                for plan in managed_plans
                if plan.logical_plan_id == logical_plan_id
            ),
            logical_plan_id,
        ),
    )
    managed_plan = next(plan for plan in managed_plans if plan.logical_plan_id == selected_plan_id)
    current_version = managed_plan.current_version

    meta_col, version_col = st.columns(2, gap="large")
    with meta_col:
        st.markdown("**Metadonnees du plan**")
        display_name = st.text_input("Nom d affichage", value=managed_plan.display_name, key="kb_display_name")
        project_code = st.text_input("Code projet", value=managed_plan.project_code or "", key="kb_project_code")
        discipline = st.text_input("Discipline", value=managed_plan.discipline or "", key="kb_discipline")
        notes = st.text_area("Notes", value=managed_plan.notes or "", height=120, key="kb_notes")

        if st.button("Enregistrer les metadonnees", use_container_width=True):
            catalog.update_plan_metadata(
                managed_plan.logical_plan_id,
                display_name=display_name,
                project_code=project_code,
                discipline=discipline,
                notes=notes,
            )
            st.success("Metadonnees mises a jour.")
            st.rerun()

        archive_label = "Desarchiver le plan" if managed_plan.is_archived else "Archiver le plan"
        if st.button(archive_label, use_container_width=True):
            catalog.set_plan_archived(managed_plan.logical_plan_id, archived=not managed_plan.is_archived)
            st.success("Statut du plan mis a jour.")
            st.rerun()

        if st.button("Supprimer tout le plan", type="secondary", use_container_width=True):
            removed = catalog.remove_plan(managed_plan.logical_plan_id)
            if removed is not None:
                _delete_from_index([version.plan_id for version in removed.versions])
                _reset_runtime_state()
                st.success("Plan supprime de la base de connaissance.")
                st.rerun()

    with version_col:
        st.markdown("**Versions du plan**")
        if current_version is not None:
            st.caption(f"Version active: {current_version.filename} ({current_version.version_id})")

        selected_version_id = st.selectbox(
            "Version",
            options=[version.version_id for version in managed_plan.versions],
            format_func=lambda version_id: _version_label(
                next(version for version in managed_plan.versions if version.version_id == version_id)
            ),
            key="kb_version_select",
        )
        selected_version = next(version for version in managed_plan.versions if version.version_id == selected_version_id)

        if st.button("Definir comme version active", use_container_width=True):
            catalog.set_active_version(managed_plan.logical_plan_id, selected_version.version_id)
            st.success("Version active mise a jour.")
            st.rerun()

        if st.button("Supprimer cette version", type="secondary", use_container_width=True):
            removed_version = catalog.remove_version(selected_version.version_id)
            if removed_version is not None:
                _delete_from_index([removed_version.plan_id])
                _reset_runtime_state()
                st.success("Version supprimee.")
                st.rerun()

        st.markdown(
            "\n".join(
                [
                    f"fichier: `{selected_version.filename}`",
                    f"version_id: `{selected_version.version_id}`",
                    f"revision: `{selected_version.revision_label or 'n/a'}`",
                    f"checksum: `{selected_version.checksum[:12]}...`",
                    f"doublon_de: `{selected_version.duplicate_of_version_id or 'non'}`",
                    f"indexation: `{selected_version.index_status.state}`",
                    f"chunks: `{selected_version.index_status.chunk_count}`",
                    f"ajoute_le: `{selected_version.added_at}`",
                ]
            )
        )

        analysis = _analysis_service()

        if st.button("Extraire les champs cles", use_container_width=True):
            extraction = analysis.extract_fields(_build_document(selected_version))
            if not extraction.fields:
                st.info("Aucun champ cle n a pu etre extrait sur cette version.")
            else:
                for field in extraction.fields:
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

        comparable_versions = [version for version in managed_plan.versions if version.version_id != selected_version.version_id]
        if comparable_versions:
            compare_with_version_id = st.selectbox(
                "Comparer avec",
                options=[version.version_id for version in comparable_versions],
                format_func=lambda version_id: _version_label(
                    next(version for version in comparable_versions if version.version_id == version_id)
                ),
                key="kb_compare_version",
            )
            if st.button("Comparer les deux versions", use_container_width=True):
                base_version = next(
                    version for version in managed_plan.versions if version.version_id == compare_with_version_id
                )
                comparison = analysis.compare_versions(
                    _build_document(base_version),
                    _build_document(selected_version),
                )
                if not comparison.has_changes:
                    st.info("Aucune difference textuelle nette detectee entre ces deux versions.")
                else:
                    st.markdown(
                        f"Ajouts: `{len(comparison.added_changes)}` | Suppressions: `{len(comparison.removed_changes)}`"
                    )
                    if comparison.added_changes:
                        st.markdown("**Ajouts detectes**")
                        for change in comparison.added_changes:
                            st.caption(
                                f"p.{change.citation.page_number} | {change.citation.chunk_id} | {change.citation.excerpt}"
                            )
                    if comparison.removed_changes:
                        st.markdown("**Suppressions detectees**")
                        for change in comparison.removed_changes:
                            st.caption(
                                f"p.{change.citation.page_number} | {change.citation.chunk_id} | {change.citation.excerpt}"
                            )

    with st.expander("Doublons detectes", expanded=False):
        duplicate_versions = [version for version in catalog.list_all_versions() if version.duplicate_of_version_id]
        if not duplicate_versions:
            st.info("Aucun doublon exact detecte.")
        else:
            for version in duplicate_versions:
                st.caption(f"{version.filename} -> doublon exact de {version.duplicate_of_version_id}")


def _render_sync_and_indexing(catalog: PlanCatalog) -> None:
    sync_col, reindex_col = st.columns(2, gap="large")
    with sync_col:
        st.subheader("Synchronisation de la base")
        default_catalog_dir = str((get_project_root() / "samples").resolve())
        catalog_dir = st.text_input("Dossier a synchroniser", value=default_catalog_dir)
        recursive = st.toggle("Recherche recursive", value=True)

        if st.button("Synchroniser les PDF", use_container_width=True):
            try:
                added = catalog.register_directory(catalog_dir, recursive=recursive)
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.success(f"{len(added)} plan(s) enregistres ou mis a jour.")
                st.rerun()

    with reindex_col:
        st.subheader("Reindexation")
        all_plans = catalog.list_plans()
        backend_name, model_name = _embedding_runtime_config()
        st.caption(f"Embeddings actifs: {backend_name} / {model_name}")

        if all_plans and st.button("Reindexer toute la base", use_container_width=True):
            try:
                _reindex_plans(catalog, all_plans)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                _reset_runtime_state()
                st.success("Reindexation terminee pour toute la base.")

        if all_plans:
            selected_plan_id = st.selectbox(
                "Reindexation ciblee",
                options=[plan.plan_id for plan in all_plans],
                format_func=lambda plan_id: next(plan.filename for plan in all_plans if plan.plan_id == plan_id),
            )
            if st.button("Reindexer cette version", use_container_width=True):
                target_plans = [plan for plan in all_plans if plan.plan_id == selected_plan_id]
                try:
                    _reindex_plans(catalog, target_plans)
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    _reset_runtime_state()
                    st.success("Reindexation terminee pour la version selectionnee.")


def _render_status(catalog: PlanCatalog) -> None:
    st.subheader("Statut de la base")
    managed_plans = catalog.list_managed_plans()
    if not managed_plans:
        st.info("Aucun plan enregistre pour le moment.")
        return

    for managed_plan in managed_plans:
        st.markdown(
            "\n".join(
                [
                    f"**{managed_plan.display_name}**",
                    f"plan_metier: `{managed_plan.logical_plan_id}`",
                    f"versions: `{len(managed_plan.versions)}`",
                    f"version_active: `{managed_plan.active_version_id or 'n/a'}`",
                    f"archive: `{managed_plan.is_archived}`",
                ]
            )
        )
        for version in managed_plan.versions:
            st.caption(
                " | ".join(
                    [
                        version.filename,
                        f"etat={version.index_status.state}",
                        f"chunks={version.index_status.chunk_count}",
                        f"backend={version.index_status.embedding_backend or 'n/a'}",
                        f"modele={version.index_status.embedding_model or 'n/a'}",
                        f"doublon={version.duplicate_of_version_id or 'non'}",
                    ]
                )
            )
        st.divider()


def _render_evaluation(catalog: PlanCatalog) -> None:
    st.subheader("Evaluation")
    evaluation_registry = _evaluation_registry()
    cases = evaluation_registry.list_cases()

    eval_col, run_col = st.columns(2, gap="large")
    with eval_col:
        st.caption(f"Cas d evaluation: {len(cases)}")
        question = st.text_input("Question de test", value="")
        expected_terms_raw = st.text_input("Termes attendus", value="", placeholder="REV B, hydraulique")
        expected_document_id = st.text_input("Document attendu", value="")
        expected_page_number_raw = st.text_input("Page attendue", value="")
        notes = st.text_area("Notes", value="", height=100)

        if st.button("Ajouter un cas d evaluation", use_container_width=True):
            expected_terms = [term.strip() for term in expected_terms_raw.split(",") if term.strip()]
            expected_page_number = int(expected_page_number_raw) if expected_page_number_raw.strip() else None
            evaluation_registry.add_case(
                question=question,
                expected_terms=expected_terms,
                expected_document_id=expected_document_id or None,
                expected_page_number=expected_page_number,
                notes=notes,
            )
            st.success("Cas d evaluation ajoute.")
            st.rerun()

        with st.expander("Jeu de test courant", expanded=False):
            for case in cases:
                st.markdown(
                    "\n".join(
                        [
                            f"**{case.case_id}**",
                            case.question,
                            f"termes: `{', '.join(case.expected_terms) or 'n/a'}`",
                            f"document: `{case.expected_document_id or 'n/a'}`",
                            f"page: `{case.expected_page_number or 'n/a'}`",
                            f"notes: `{case.notes or 'n/a'}`",
                        ]
                    )
                )
                st.divider()

    with run_col:
        if st.button("Lancer l evaluation", use_container_width=True):
            try:
                _run_evaluation(catalog)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.success("Evaluation terminee.")

        runs = evaluation_registry.list_runs()
        if runs:
            latest = runs[-1]
            st.metric("Pass rate", f"{latest.pass_rate:.0%}")
            st.metric("Term hit rate", f"{latest.answer_term_hit_rate:.0%}")
            st.metric("Citation doc hit rate", f"{latest.citation_document_hit_rate:.0%}")
            st.metric("Citation page hit rate", f"{latest.citation_page_hit_rate:.0%}")

            with st.expander("Dernier run", expanded=False):
                st.caption(f"run_id={latest.run_id} | date={latest.created_at}")
                for result in latest.results:
                    st.markdown(
                        "\n".join(
                            [
                                f"**{result.case_id}** - {'PASS' if result.passed else 'FAIL'}",
                                result.question,
                                f"citations: `{result.citations_count}`",
                                f"terms: `{result.answer_contains_expected_terms}`",
                                f"doc hit: `{result.citation_document_hit}`",
                                f"page hit: `{result.citation_page_hit}`",
                                f"answer: `{result.answer_text}`",
                            ]
                        )
                    )
                    st.divider()


def main() -> None:
    _render_sidebar_links()
    catalog = _catalog()

    st.title("Administration")
    st.write("Configurez le LLM, les embeddings, les prompts et la base de connaissance documentaire.")

    required_token = os.getenv("RAG_ADMIN_TOKEN", "").strip()
    if required_token:
        st.text_input("Jeton administrateur", type="password", key="admin_token")

    if not _authorized_for_admin():
        st.warning("Jeton administrateur requis.")
        return

    _render_config_section()
    st.divider()
    _render_knowledge_base_admin(catalog)
    st.divider()
    _render_sync_and_indexing(catalog)
    st.divider()
    _render_status(catalog)
    st.divider()
    _render_evaluation(catalog)


if __name__ == "__main__":
    main()
