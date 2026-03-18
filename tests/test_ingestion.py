"""
test_ingestion.py — Unit tests for document loading and chunking logic.

These tests are fully offline: no LLM, no vector store, no network calls.
They verify that the ingestion layer correctly extracts text and produces
well-formed LangChain Document chunks with the expected metadata.
"""

from __future__ import annotations

import io

import pytest

from app.ingestion import (
    _detect_document_type,
    _extract_text_from_txt,
    ingest_file,
    load_and_chunk,
    save_upload,
)
from app.models import DocumentType


# ======================================================================= #
#  _detect_document_type                                                    #
# ======================================================================= #


class TestDetectDocumentType:
    def test_pdf(self):
        assert _detect_document_type("report.pdf") == DocumentType.PDF

    def test_pdf_uppercase(self):
        assert _detect_document_type("REPORT.PDF") == DocumentType.PDF

    def test_txt(self):
        assert _detect_document_type("notes.txt") == DocumentType.TXT

    def test_text_extension(self):
        assert _detect_document_type("notes.text") == DocumentType.TXT

    def test_docx(self):
        assert _detect_document_type("document.docx") == DocumentType.DOCX

    def test_unknown(self):
        assert _detect_document_type("archive.zip") == DocumentType.UNKNOWN

    def test_no_extension(self):
        assert _detect_document_type("README") == DocumentType.UNKNOWN


# ======================================================================= #
#  _extract_text_from_txt                                                   #
# ======================================================================= #


class TestExtractTextFromTxt:
    def test_utf8_content(self):
        content = "Hello, world!\nSecond line."
        pages = _extract_text_from_txt(content.encode("utf-8"))
        assert len(pages) == 1
        text, page_num = pages[0]
        assert "Hello, world!" in text
        assert page_num == 0

    def test_latin1_fallback(self):
        """Bytes that are valid Latin-1 but invalid UTF-8 must not crash."""
        latin1_bytes = "Caf\xe9".encode("latin-1")  # "Café"
        pages = _extract_text_from_txt(latin1_bytes)
        assert len(pages) == 1
        assert "Caf" in pages[0][0]

    def test_empty_content(self):
        pages = _extract_text_from_txt(b"")
        assert pages[0][0] == ""


# ======================================================================= #
#  load_and_chunk                                                           #
# ======================================================================= #


class TestLoadAndChunk:
    """Tests using plain-text content to avoid PDF/DOCX dependencies."""

    _LONG_TEXT = " ".join([f"Sentence number {i}." for i in range(200)])

    def test_returns_list_of_documents(self):
        chunks = load_and_chunk(
            document_id="test-doc-1",
            filename="test.txt",
            file_bytes=self._LONG_TEXT.encode(),
        )
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_metadata_present(self):
        chunks = load_and_chunk(
            document_id="test-doc-1",
            filename="test.txt",
            file_bytes=self._LONG_TEXT.encode(),
        )
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["document_id"] == "test-doc-1"
            assert chunk.metadata["filename"] == "test.txt"
            assert chunk.metadata["chunk_index"] == i
            assert "page_number" in chunk.metadata

    def test_chunk_index_sequential(self):
        chunks = load_and_chunk(
            document_id="doc-x",
            filename="long.txt",
            file_bytes=self._LONG_TEXT.encode(),
        )
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_custom_chunk_size(self):
        text = "A" * 500
        chunks_small = load_and_chunk(
            document_id="d",
            filename="a.txt",
            file_bytes=text.encode(),
            chunk_size=100,
            chunk_overlap=10,
        )
        chunks_large = load_and_chunk(
            document_id="d",
            filename="a.txt",
            file_bytes=text.encode(),
            chunk_size=500,
            chunk_overlap=10,
        )
        # Smaller chunk size → more chunks
        assert len(chunks_small) > len(chunks_large)

    def test_chunk_max_size_respected(self):
        text = "X" * 10_000
        chunks = load_and_chunk(
            document_id="d",
            filename="a.txt",
            file_bytes=text.encode(),
            chunk_size=500,
            chunk_overlap=50,
        )
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # allow small splitter overshoot

    def test_overlap_creates_shared_content(self):
        """Adjacent chunks should share some text when overlap > 0."""
        # 400 chars repeated twice — should produce 2 overlapping chunks
        word = "overlap "
        text = (word * 50) + "\n\n" + (word * 50)
        chunks = load_and_chunk(
            document_id="d",
            filename="a.txt",
            file_bytes=text.encode(),
            chunk_size=300,
            chunk_overlap=100,
        )
        if len(chunks) >= 2:
            # The end of chunk[0] should appear near the start of chunk[1]
            end_of_first = chunks[0].page_content[-80:]
            start_of_second = chunks[1].page_content[:80]
            # At least some words should overlap
            first_words = set(end_of_first.split())
            second_words = set(start_of_second.split())
            assert first_words & second_words, "Expected overlapping words"

    def test_empty_file_raises(self):
        with pytest.raises(ValueError, match="No extractable text"):
            load_and_chunk(
                document_id="d",
                filename="empty.txt",
                file_bytes=b"   \n  \t  ",   # whitespace only
            )

    def test_unsupported_type_falls_back_to_txt(self):
        """Unknown extensions should be treated as plain text."""
        chunks = load_and_chunk(
            document_id="d",
            filename="data.csv",
            file_bytes=b"col1,col2\nval1,val2\n",
        )
        assert len(chunks) >= 1


# ======================================================================= #
#  ingest_file (high-level)                                                 #
# ======================================================================= #


class TestIngestFile:
    def test_returns_document_id_and_chunks(self):
        doc_id, chunks = ingest_file(
            filename="sample.txt",
            file_bytes=b"This is a sample document with enough text to form a chunk.",
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36  # UUID format
        assert len(chunks) >= 1

    def test_custom_document_id_is_preserved(self):
        fixed_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        doc_id, chunks = ingest_file(
            filename="sample.txt",
            file_bytes=b"Hello world",
            document_id=fixed_id,
        )
        assert doc_id == fixed_id
        assert all(c.metadata["document_id"] == fixed_id for c in chunks)

    def test_generates_uuid_when_id_not_given(self):
        doc_id1, _ = ingest_file(filename="a.txt", file_bytes=b"abc def ghi")
        doc_id2, _ = ingest_file(filename="a.txt", file_bytes=b"abc def ghi")
        # Each call without an explicit id should get a unique UUID
        assert doc_id1 != doc_id2


# ======================================================================= #
#  save_upload                                                              #
# ======================================================================= #


class TestSaveUpload:
    def test_file_persisted_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.ingestion.settings.UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr("app.ingestion.settings.MAX_FILE_SIZE_MB", 10)
        content = b"Test file content."
        doc_id, saved_path = save_upload("test.txt", io.BytesIO(content))
        assert saved_path.exists()
        assert saved_path.read_bytes() == content

    def test_rejects_oversized_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.ingestion.settings.UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr("app.ingestion.settings.MAX_FILE_SIZE_MB", 1)
        big_content = b"X" * (2 * 1024 * 1024)  # 2 MB
        with pytest.raises(ValueError, match="exceeds"):
            save_upload("big.txt", io.BytesIO(big_content))
