from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from rag_pdf.bootstrap import build_app
from rag_pdf.config import load_environment
from rag_pdf.domain.models import SourceDocument


def main() -> None:
    load_environment()

    if len(sys.argv) < 2:
        print("Usage: python -m rag_pdf.main <pdf_path> [question]")
        return

    pdf_path = Path(sys.argv[1])
    question = sys.argv[2] if len(sys.argv) > 2 else "Quelle information cle contient ce plan ?"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    ingestion, qa = build_app()

    document = SourceDocument(
        document_id=pdf_path.stem.upper(),
        file_path=str(pdf_path),
        checksum=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    )

    ingestion.ingest(document)
    answer = qa.answer(question)

    print(answer.text)
    for citation in answer.citations:
        print(
            f"- {citation.document_id} p.{citation.page_number} "
            f"chunk={citation.chunk_id} excerpt={citation.excerpt!r}"
        )


if __name__ == "__main__":
    main()
