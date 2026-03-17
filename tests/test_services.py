import os
from pathlib import Path

from rag_pdf.application.analysis import PlanAnalysisService
from rag_pdf.application.services import IngestionService, QuestionAnsweringService, RetrievalService
from rag_pdf.config import load_environment, update_environment
from rag_pdf.domain.models import BoundingBox, Citation, DocumentChunk, RegionKind, RetrievedPassage, SourceDocument, VisualRegion
from rag_pdf.domain.ports import AnswerGenerator, ChunkIndex, Chunker, DocumentParser, EmbeddingProvider
from rag_pdf.infrastructure.catalog.plan_catalog import PlanCatalog
from rag_pdf.infrastructure.chunking.layout_chunker import LayoutAwareChunker
from rag_pdf.infrastructure.embeddings.stub_embeddings import (
    MistralEmbeddingProvider,
    OllamaEmbeddingProvider,
)
from rag_pdf.infrastructure.evaluation.evaluation_registry import EvaluationCaseResult, EvaluationRegistry
from rag_pdf.infrastructure.evaluation.evaluator import RagEvaluator
from rag_pdf.infrastructure.generation.grounded_generator import (
    MistralGroundedAnswerGenerator,
    OllamaGroundedAnswerGenerator,
)
from rag_pdf.infrastructure.parsers.pymupdf_parser import RegionClassifier
from rag_pdf.infrastructure.prompts.prompt_registry import PromptRegistry
from rag_pdf.infrastructure.retrieval.sqlite_hybrid_index import SqliteHybridChunkIndex


class FakeParser(DocumentParser):
    def parse(self, document: SourceDocument):
        return []


class FakeChunker(Chunker):
    def chunk(self, document: SourceDocument, regions):
        return [
            DocumentChunk(
                chunk_id="chunk-1",
                document_id=document.document_id,
                page_number=1,
                region_id="region-1",
                text="ligne principale distribution",
                bbox=BoundingBox(0, 0, 10, 10),
            )
        ]


class EmptyChunker(Chunker):
    def chunk(self, document: SourceDocument, regions):
        return [
            DocumentChunk(
                chunk_id="empty-1",
                document_id=document.document_id,
                page_number=1,
                region_id="region-empty",
                text="   ",
                bbox=BoundingBox(0, 0, 10, 10),
            )
        ]


class FakeEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class AnalysisChunker(Chunker):
    def __init__(self, chunks):
        self._chunks = chunks

    def chunk(self, document: SourceDocument, regions):
        return self._chunks


class FakeIndex(ChunkIndex):
    def __init__(self):
        self.stored = []

    def upsert(self, chunks, embeddings):
        self.stored.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int, document_ids: list[str] | None = None):
        return [RetrievedPassage(chunk=self.stored[0], score=1.0)] if self.stored else []

    def search_lexical(self, query: str, top_k: int, document_ids: list[str] | None = None):
        return [RetrievedPassage(chunk=self.stored[0], score=1.0)] if self.stored else []

    def delete(self, document_ids: list[str]) -> None:
        self.stored = [chunk for chunk in self.stored if chunk.document_id not in document_ids]


class FakeGenerator(AnswerGenerator):
    def generate(self, question: str, passages):
        citations = tuple(
            Citation(
                document_id=passage.chunk.document_id,
                page_number=passage.chunk.page_number,
                chunk_id=passage.chunk.chunk_id,
                bbox=passage.chunk.bbox,
                excerpt=passage.chunk.text,
            )
            for passage in passages
        )
        return type("AnswerLike", (), {"text": "ok", "citations": citations, "question": question})()


def test_ingestion_service_indexes_generated_chunks():
    document = SourceDocument(
        document_id="DOC-001",
        file_path="demo.pdf",
        checksum="x",
    )
    index = FakeIndex()
    service = IngestionService(
        parser=FakeParser(),
        chunker=FakeChunker(),
        embeddings=FakeEmbeddings(),
        index=index,
    )

    count = service.ingest(document)

    assert count == 1
    assert len(index.stored) == 1


def test_question_answering_returns_grounded_response():
    document = SourceDocument(
        document_id="DOC-001",
        file_path="demo.pdf",
        checksum="x",
    )
    index = FakeIndex()
    ingestion = IngestionService(
        parser=FakeParser(),
        chunker=FakeChunker(),
        embeddings=FakeEmbeddings(),
        index=index,
    )
    retrieval = RetrievalService(vector_index=index, embeddings=FakeEmbeddings(), lexical_searcher=index)
    qa = QuestionAnsweringService(retrieval=retrieval, generator=FakeGenerator())

    ingestion.ingest(document)
    answer = qa.answer("Ou est la ligne principale ?")

    assert answer.text == "ok"
    assert len(answer.citations) == 1


def test_ingestion_service_skips_empty_chunks():
    document = SourceDocument(
        document_id="DOC-001",
        file_path="demo.pdf",
        checksum="x",
    )
    index = FakeIndex()
    service = IngestionService(
        parser=FakeParser(),
        chunker=EmptyChunker(),
        embeddings=FakeEmbeddings(),
        index=index,
    )

    count = service.ingest(document)

    assert count == 0
    assert len(index.stored) == 0


def test_region_classifier_detects_title_block():
    classifier = RegionClassifier()

    kind = classifier.classify(
        text="DRAWING TITLE REV SCALE CLIENT",
        bbox=BoundingBox(700, 820, 980, 980),
        page_bbox=BoundingBox(0, 0, 1000, 1000),
    )

    assert kind.value == "title_block"


def test_region_classifier_detects_table():
    classifier = RegionClassifier()

    text = "\n".join(
        [
            "ITEM 1",
            "QTY 2",
            "DESCRIPTION PIPE",
            "REPERE A-201",
        ]
    )

    kind = classifier.classify(
        text=text,
        bbox=BoundingBox(100, 100, 300, 300),
        page_bbox=BoundingBox(0, 0, 1000, 1000),
    )

    assert kind.value == "table"


def test_layout_chunker_groups_title_block_regions():
    chunker = LayoutAwareChunker()
    document = SourceDocument(document_id="DOC-001", file_path="demo.pdf", checksum="x")
    regions = [
        VisualRegion(
            region_id="r1",
            page_number=1,
            kind=RegionKind.TITLE_BLOCK,
            bbox=BoundingBox(780, 720, 920, 748),
            metadata={"line_count": 1, "drawing_count": 5},
        ),
        VisualRegion(
            region_id="r2",
            page_number=1,
            kind=RegionKind.TITLE_BLOCK,
            bbox=BoundingBox(780, 748, 980, 792),
            metadata={"line_count": 2, "drawing_count": 5},
        ),
    ]

    chunks = chunker.chunk(document, regions)

    assert len(chunks) == 1
    assert chunks[0].metadata["region_kind"] == "title_block"
    assert chunks[0].metadata["group_size"] == 2


def test_layout_chunker_groups_callouts_by_proximity():
    chunker = LayoutAwareChunker()
    document = SourceDocument(document_id="DOC-001", file_path="demo.pdf", checksum="x")
    regions = [
        VisualRegion(region_id="c1", page_number=1, kind=RegionKind.CALLOUT, bbox=BoundingBox(700, 100, 860, 120)),
        VisualRegion(region_id="c2", page_number=1, kind=RegionKind.CALLOUT, bbox=BoundingBox(705, 132, 870, 152)),
        VisualRegion(region_id="c3", page_number=1, kind=RegionKind.CALLOUT, bbox=BoundingBox(710, 300, 870, 320)),
    ]

    chunks = chunker.chunk(document, regions)

    assert len(chunks) == 2
    assert chunks[0].metadata["group_size"] == 2


class FakeMistralMessage:
    def __init__(self, content):
        self.content = content


class FakeMistralChoice:
    def __init__(self, content):
        self.message = FakeMistralMessage(content)


class FakeMistralResponse:
    def __init__(self, content):
        self.choices = [FakeMistralChoice(content)]


class FakeMistralChat:
    def __init__(self, content):
        self._content = content

    def complete(self, **kwargs):
        return FakeMistralResponse(self._content)


class FakeMistralClient:
    def __init__(self, content):
        self.chat = FakeMistralChat(content)


class FakeMistralEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeMistralEmbeddingsResponse:
    def __init__(self, embeddings):
        self.data = [FakeMistralEmbeddingItem(embedding) for embedding in embeddings]


class FakeMistralEmbeddingsApi:
    def create(self, model, inputs):
        return FakeMistralEmbeddingsResponse([[0.1, 0.2] for _ in inputs])


class FakeMistralEmbeddingClient:
    def __init__(self):
        self.embeddings = FakeMistralEmbeddingsApi()


def test_mistral_generator_builds_grounded_answer():
    generator = MistralGroundedAnswerGenerator(client=FakeMistralClient("Revision B [DOC-001 p.1 chunk=chunk-1]"))
    passages = [
        RetrievedPassage(
            chunk=DocumentChunk(
                chunk_id="chunk-1",
                document_id="DOC-001",
                page_number=1,
                region_id="title",
                text="REV B",
                bbox=BoundingBox(0, 0, 10, 10),
                metadata={"region_kind": "title_block"},
            ),
            score=1.0,
        )
    ]

    answer = generator.generate("Quelle est la revision ?", passages)

    assert "Revision B" in answer.text
    assert answer.metadata["generator"] == "mistral"
    assert len(answer.citations) == 1


class FakeOllamaGenerator(OllamaGroundedAnswerGenerator):
    def _call_ollama(self, payload: dict) -> dict:
        return {
            "message": {
                "content": "Revision B [DOC-001 p.1 chunk=chunk-1]",
            }
        }


def test_ollama_generator_builds_grounded_answer():
    generator = FakeOllamaGenerator(model="llama3.1")
    passages = [
        RetrievedPassage(
            chunk=DocumentChunk(
                chunk_id="chunk-1",
                document_id="DOC-001",
                page_number=1,
                region_id="title",
                text="REV B",
                bbox=BoundingBox(0, 0, 10, 10),
                metadata={"region_kind": "title_block"},
            ),
            score=1.0,
        )
    ]

    answer = generator.generate("Quelle est la revision ?", passages)

    assert "Revision B" in answer.text
    assert answer.metadata["generator"] == "ollama"
    assert len(answer.citations) == 1


class FakeOllamaEmbeddingProvider(OllamaEmbeddingProvider):
    def _embed_single(self, text: str) -> list[float]:
        return [0.4, 0.6]


def test_ollama_embedding_provider_returns_vectors():
    provider = FakeOllamaEmbeddingProvider(model="embeddinggemma")

    vectors = provider.embed(["abc", "def"])

    assert vectors == [[0.4, 0.6], [0.4, 0.6]]


def test_mistral_embedding_provider_returns_vectors():
    provider = MistralEmbeddingProvider(client=FakeMistralEmbeddingClient())

    vectors = provider.embed(["abc"])

    assert vectors == [[0.1, 0.2]]


def test_load_environment_reads_dotenv_file():
    os.environ.pop("OLLAMA_MODEL", None)

    load_environment()

    assert os.getenv("OLLAMA_MODEL") == "qwen2.5:7b-instruct"


def test_update_environment_rewrites_values():
    from rag_pdf import config as config_module

    class FakeParent:
        def mkdir(self, parents=False, exist_ok=False):
            return None

    class FakeEnvPath:
        def __init__(self):
            self.parent = FakeParent()
            self.content = "OLLAMA_MODEL=old-model\n"

        def exists(self):
            return True

        def read_text(self, encoding="utf-8"):
            return self.content

        def write_text(self, content, encoding="utf-8"):
            self.content = content

    fake_env_path = FakeEnvPath()
    original = config_module.get_env_path
    config_module.get_env_path = lambda: fake_env_path

    try:
        update_environment({"OLLAMA_MODEL": "new-model", "LLM_BACKEND": "ollama"})
    finally:
        config_module.get_env_path = original

    assert "OLLAMA_MODEL=new-model" in fake_env_path.content
    assert "LLM_BACKEND=ollama" in fake_env_path.content


def test_plan_catalog_registers_pdf():
    base_dir = Path("data/test_plan_catalog_registers_pdf")
    try:
        catalog = PlanCatalog(catalog_path=base_dir / "catalog.json")
        sample_pdf = Path("samples/demo_plan.pdf").resolve()

        record = catalog.register_plan(sample_pdf)

        assert record.logical_plan_id == "DEMO_PLAN"
        assert record.plan_id == record.version_id
        assert len(catalog.list_plans()) == 1
        assert len(catalog.list_managed_plans()) == 1
    finally:
        catalog_path = base_dir / "catalog.json"
        if catalog_path.exists():
            catalog_path.unlink()
        if base_dir.exists():
            base_dir.rmdir()


def test_plan_catalog_marks_indexed():
    base_dir = Path("data/test_plan_catalog_marks_indexed")
    try:
        catalog = PlanCatalog(catalog_path=base_dir / "catalog.json")
        sample_pdf = Path("samples/demo_plan.pdf").resolve()
        record = catalog.register_plan(sample_pdf)

        updated = catalog.mark_indexed(
            plan_id=record.plan_id,
            chunk_count=12,
            embedding_backend="ollama",
            embedding_model="embeddinggemma",
        )

        assert updated.index_status.state == "indexed"
        assert updated.index_status.chunk_count == 12
        assert updated.index_status.embedding_backend == "ollama"
    finally:
        catalog_path = base_dir / "catalog.json"
        if catalog_path.exists():
            catalog_path.unlink()
        if base_dir.exists():
            base_dir.rmdir()


def test_plan_catalog_marks_index_error():
    base_dir = Path("data/test_plan_catalog_marks_index_error")
    try:
        catalog = PlanCatalog(catalog_path=base_dir / "catalog.json")
        sample_pdf = Path("samples/demo_plan.pdf").resolve()
        record = catalog.register_plan(sample_pdf)

        updated = catalog.mark_index_error(
            plan_id=record.plan_id,
            embedding_backend="ollama",
            embedding_model="embeddinggemma",
            error_message="boom",
        )

        assert updated.index_status.state == "error"
        assert updated.index_status.error_message == "boom"
    finally:
        catalog_path = base_dir / "catalog.json"
        if catalog_path.exists():
            catalog_path.unlink()
        if base_dir.exists():
            base_dir.rmdir()


def test_plan_catalog_groups_versions_and_flags_duplicates():
    base_dir = Path("data/test_plan_catalog_grouping")
    try:
        catalog = PlanCatalog(catalog_path=base_dir / "catalog.json")
        original = Path("samples/demo_plan.pdf").resolve()
        rev_copy = base_dir / "demo_plan_rev_b.pdf"
        dup_copy = base_dir / "demo_plan_duplicate.pdf"
        base_dir.mkdir(parents=True, exist_ok=True)
        rev_copy.write_bytes(original.read_bytes())
        dup_copy.write_bytes(original.read_bytes())

        first = catalog.register_plan(original)
        second = catalog.register_plan(rev_copy)
        duplicate = catalog.register_plan(dup_copy)

        managed = catalog.list_managed_plans()

        assert len(managed) == 2
        assert len(next(plan for plan in managed if plan.logical_plan_id == first.logical_plan_id).versions) == 2
        assert second.logical_plan_id == first.logical_plan_id
        assert duplicate.duplicate_of_version_id is not None
    finally:
        for file_path in [base_dir / "catalog.json", base_dir / "demo_plan_rev_b.pdf", base_dir / "demo_plan_duplicate.pdf"]:
            if file_path.exists():
                file_path.unlink()
        if base_dir.exists():
            base_dir.rmdir()


def test_plan_catalog_can_switch_active_version_and_delete():
    base_dir = Path("data/test_plan_catalog_switch")
    try:
        catalog = PlanCatalog(catalog_path=base_dir / "catalog.json")
        original = Path("samples/demo_plan.pdf").resolve()
        rev_copy = base_dir / "demo_plan_rev_b.pdf"
        base_dir.mkdir(parents=True, exist_ok=True)
        rev_copy.write_bytes(original.read_bytes())

        first = catalog.register_plan(original)
        second = catalog.register_plan(rev_copy)

        updated_plan = catalog.set_active_version(first.logical_plan_id, second.version_id)
        active = updated_plan.current_version

        assert active is not None
        assert active.version_id == second.version_id

        removed = catalog.remove_version(first.version_id)

        assert removed is not None
        assert len(catalog.list_managed_plans()[0].versions) == 1
    finally:
        for file_path in [base_dir / "catalog.json", base_dir / "demo_plan_rev_b.pdf"]:
            if file_path.exists():
                file_path.unlink()
        if base_dir.exists():
            base_dir.rmdir()


def test_sqlite_hybrid_index_filters_by_document():
    index = SqliteHybridChunkIndex(db_path=":memory:")
    chunks = [
        DocumentChunk(
            chunk_id="chunk-a",
            document_id="DOC-A",
            page_number=1,
            region_id="r1",
            text="revision A cartouche",
            bbox=BoundingBox(0, 0, 10, 10),
        ),
        DocumentChunk(
            chunk_id="chunk-b",
            document_id="DOC-B",
            page_number=1,
            region_id="r2",
            text="revision B cartouche",
            bbox=BoundingBox(0, 0, 10, 10),
        ),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    index.upsert(chunks, embeddings)
    results = index.search([1.0, 0.0], top_k=5, document_ids=["DOC-A"])

    assert len(results) == 1
    assert results[0].chunk.document_id == "DOC-A"


def test_prompt_registry_creates_and_activates_versions():
    registry_path = Path("data/test_prompts_registry.json")
    try:
        registry = PromptRegistry(registry_path=registry_path)
        new_prompt = registry.create_prompt_version(
            name="Test prompt",
            content="Prompt de test",
            activate=True,
        )

        active_prompt = registry.get_active_prompt()

        assert new_prompt.version_id == active_prompt.version_id
        assert active_prompt.content == "Prompt de test"
    finally:
        if registry_path.exists():
            registry_path.unlink()


def test_evaluation_registry_adds_case_and_builds_run():
    registry_dir = Path("data/test_evaluation_registry")
    try:
        registry = EvaluationRegistry(base_dir=registry_dir)
        case = registry.add_case(
            question="Quelle est la revision ?",
            expected_terms=["rev b"],
            expected_document_id="DOC-001",
            expected_page_number=1,
        )

        run = registry.build_run(
            [
                EvaluationCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    answer_text="La revision est REV B.",
                    citations_count=1,
                    answer_contains_expected_terms=True,
                    citation_document_hit=True,
                    citation_page_hit=True,
                    passed=True,
                )
            ]
        )
        registry.save_run(run)

        assert len(registry.list_cases()) == 1
        assert registry.list_runs()[-1].pass_rate == 1.0
    finally:
        cases_path = registry_dir / "cases.json"
        runs_path = registry_dir / "runs.json"
        if cases_path.exists():
            cases_path.unlink()
        if runs_path.exists():
            runs_path.unlink()
        if registry_dir.exists():
            registry_dir.rmdir()


def test_rag_evaluator_marks_case_as_passed():
    from rag_pdf.infrastructure.evaluation.evaluation_registry import EvaluationCase

    evaluator = RagEvaluator()
    case = EvaluationCase(
        case_id="case-1",
        question="Quelle est la revision ?",
        expected_terms=("rev b",),
        expected_document_id="DOC-001",
        expected_page_number=1,
    )
    citations = (
        Citation(
            document_id="DOC-001",
            page_number=1,
            chunk_id="chunk-1",
            bbox=BoundingBox(0, 0, 10, 10),
            excerpt="REV B",
        ),
    )

    result = evaluator.evaluate_case(case, "La revision est REV B.", citations)

    assert result.passed is True
    assert result.answer_contains_expected_terms is True
    assert result.citation_document_hit is True
    assert result.citation_page_hit is True


def test_plan_analysis_extracts_structured_fields():
    document = SourceDocument(document_id="DOC-001", file_path="demo.pdf", checksum="x")
    chunks = [
        DocumentChunk(
            chunk_id="chunk-title",
            document_id=document.document_id,
            page_number=1,
            region_id="title",
            text="DRAWING TITLE MAIN DISTRIBUTION REV B SCALE 1:50 PROJECT ALPHA",
            bbox=BoundingBox(0, 0, 10, 10),
            metadata={"region_kind": "title_block"},
        )
    ]
    service = PlanAnalysisService(parser=FakeParser(), chunker=AnalysisChunker(chunks))

    result = service.extract_fields(document)
    values = {field.field_name: field.value for field in result.fields}

    assert values["revision"] == "B"
    assert values["scale"] == "1:50"
    assert "MAIN DISTRIBUTION" in values["drawing_title"]


def test_plan_analysis_compares_versions():
    base_document = SourceDocument(document_id="DOC-A", file_path="a.pdf", checksum="a")
    target_document = SourceDocument(document_id="DOC-B", file_path="b.pdf", checksum="b")
    base_chunks = [
        DocumentChunk(
            chunk_id="base-1",
            document_id=base_document.document_id,
            page_number=1,
            region_id="r1",
            text="NOTE EXISTING NETWORK",
            bbox=BoundingBox(0, 0, 10, 10),
        )
    ]
    target_chunks = [
        DocumentChunk(
            chunk_id="target-1",
            document_id=target_document.document_id,
            page_number=1,
            region_id="r1",
            text="NOTE UPDATED NETWORK",
            bbox=BoundingBox(0, 0, 10, 10),
        )
    ]

    class SwitchParser(DocumentParser):
        def parse(self, document: SourceDocument):
            return []

    class SwitchChunker(Chunker):
        def chunk(self, document: SourceDocument, regions):
            if document.document_id == "DOC-A":
                return base_chunks
            return target_chunks

    service = PlanAnalysisService(parser=SwitchParser(), chunker=SwitchChunker())
    comparison = service.compare_versions(base_document, target_document)

    assert len(comparison.added_changes) == 1
    assert len(comparison.removed_changes) == 1
