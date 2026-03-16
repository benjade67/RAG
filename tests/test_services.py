import os

from rag_pdf.application.services import IngestionService, QuestionAnsweringService, RetrievalService
from rag_pdf.domain.models import BoundingBox, Citation, DocumentChunk, RegionKind, RetrievedPassage, SourceDocument, VisualRegion
from rag_pdf.domain.ports import AnswerGenerator, ChunkIndex, Chunker, DocumentParser, EmbeddingProvider
from rag_pdf.config import load_environment
from rag_pdf.infrastructure.chunking.layout_chunker import LayoutAwareChunker
from rag_pdf.infrastructure.generation.grounded_generator import (
    MistralGroundedAnswerGenerator,
    OllamaGroundedAnswerGenerator,
)
from rag_pdf.infrastructure.parsers.pymupdf_parser import RegionClassifier


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


class FakeEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeIndex(ChunkIndex):
    def __init__(self):
        self.stored = []

    def upsert(self, chunks, embeddings):
        self.stored.extend(chunks)

    def search(self, query: str, top_k: int):
        return [RetrievedPassage(chunk=self.stored[0], score=1.0)] if self.stored else []


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
    retrieval = RetrievalService(vector_index=index)
    qa = QuestionAnsweringService(retrieval=retrieval, generator=FakeGenerator())

    ingestion.ingest(document)
    answer = qa.answer("Ou est la ligne principale ?")

    assert answer.text == "ok"
    assert len(answer.citations) == 1


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


def test_load_environment_reads_dotenv_file():
    os.environ.pop("OLLAMA_MODEL", None)

    load_environment()

    assert os.getenv("OLLAMA_MODEL") == "qwen2.5:7b-instruct"
