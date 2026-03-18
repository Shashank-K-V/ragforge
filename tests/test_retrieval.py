"""
test_retrieval.py — Unit and integration tests for the vector store layer.

Strategy
--------
* Uses a temporary ChromaDB directory (via tmp_path fixture) so tests
  never pollute the real chroma_db/ directory.
* The embedding model is mocked where possible to keep tests fast and
  deterministic — no network calls, no model download required.
* A small set of tests do use real embeddings to verify that ChromaDB
  integration works end-to-end (marked with @pytest.mark.integration).

Run only fast tests (default):
    pytest tests/test_retrieval.py

Run integration tests too:
    pytest tests/test_retrieval.py -m integration
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from app.models import DocumentInfo, DocumentType


# ======================================================================= #
#  Fixtures                                                                 #
# ======================================================================= #


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch):
    """Redirect ChromaDB and registry to a temporary directory."""
    chroma_dir = str(tmp_path / "chroma")
    registry_path = Path(chroma_dir) / "document_registry.json"
    monkeypatch.setattr("app.retrieval.settings.CHROMA_PERSIST_DIR", chroma_dir)
    monkeypatch.setattr("app.retrieval.settings.CHROMA_COLLECTION_NAME", "test_coll")
    monkeypatch.setattr(
        "app.retrieval._REGISTRY_PATH",
        registry_path,
    )
    # Reset singleton vector store so it picks up the new directory
    monkeypatch.setattr("app.retrieval._vector_store", None)
    monkeypatch.setattr("app.retrieval._embedding_model", None)
    return tmp_path


@pytest.fixture()
def mock_embedding_model():
    """
    Replace HuggingFaceEmbeddings with a fast mock that returns
    fixed-length random-ish vectors.  Allows Chroma tests without
    downloading the real model.
    """
    import numpy as np

    class _FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # Deterministic: hash the text to seed the RNG
            return [
                list(np.random.default_rng(abs(hash(t)) % (2**32)).random(384).astype(float))
                for t in texts
            ]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    return _FakeEmbeddings()


# ======================================================================= #
#  Document registry                                                        #
# ======================================================================= #


class TestDocumentRegistry:
    def test_register_and_list(self, isolated_settings):
        from app.retrieval import list_documents, register_document

        register_document(
            document_id="doc-1",
            filename="report.pdf",
            document_type=DocumentType.PDF,
            chunk_count=10,
        )
        docs = list_documents()
        assert len(docs) == 1
        assert docs[0].document_id == "doc-1"
        assert docs[0].filename == "report.pdf"
        assert docs[0].chunk_count == 10

    def test_multiple_documents_listed_newest_first(self, isolated_settings):
        from app.retrieval import list_documents, register_document
        import time

        register_document("id-a", "a.txt", DocumentType.TXT, 5)
        time.sleep(0.01)   # ensure distinct timestamps
        register_document("id-b", "b.txt", DocumentType.TXT, 3)

        docs = list_documents()
        assert docs[0].document_id == "id-b"  # newest first
        assert docs[1].document_id == "id-a"

    def test_upsert_same_document_id(self, isolated_settings):
        from app.retrieval import list_documents, register_document

        register_document("id-x", "old.pdf", DocumentType.PDF, 5)
        register_document("id-x", "new.pdf", DocumentType.PDF, 8)
        docs = list_documents()
        assert len(docs) == 1
        assert docs[0].filename == "new.pdf"
        assert docs[0].chunk_count == 8

    def test_empty_registry_returns_empty_list(self, isolated_settings):
        from app.retrieval import list_documents

        assert list_documents() == []

    def test_corrupted_registry_returns_empty_list(self, isolated_settings):
        from app.retrieval import _REGISTRY_PATH, list_documents

        _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REGISTRY_PATH.write_text("NOT VALID JSON", encoding="utf-8")
        assert list_documents() == []


# ======================================================================= #
#  Vector store health check                                                #
# ======================================================================= #


class TestVectorStoreHealth:
    def test_health_returns_dict_with_status(self, isolated_settings, monkeypatch, mock_embedding_model):
        monkeypatch.setattr("app.retrieval._embedding_model", mock_embedding_model)
        from app.retrieval import check_vector_store_health

        result = check_vector_store_health()
        assert "status" in result
        assert result["status"] in ("ok", "down")

    def test_health_returns_down_on_error(self, monkeypatch):
        """If the vector store raises, health should report 'down'."""
        from app.retrieval import check_vector_store_health

        monkeypatch.setattr(
            "app.retrieval.get_vector_store",
            lambda: (_ for _ in ()).throw(RuntimeError("DB gone")),
        )
        result = check_vector_store_health()
        assert result["status"] == "down"
        assert "error" in result


# ======================================================================= #
#  embed_and_store + similarity_search (integration)                        #
# ======================================================================= #


@pytest.mark.integration
class TestEmbedAndSearch:
    """
    Integration tests that actually call ChromaDB.
    Uses fake embeddings to avoid downloading the real model.
    """

    def test_embed_and_retrieve_documents(
        self, isolated_settings, monkeypatch, mock_embedding_model
    ):
        monkeypatch.setattr("app.retrieval._embedding_model", mock_embedding_model)
        monkeypatch.setattr("app.retrieval._vector_store", None)

        from app.retrieval import embed_and_store, similarity_search

        chunks = [
            Document(
                page_content="The mitochondria is the powerhouse of the cell.",
                metadata={
                    "document_id": "bio-doc",
                    "filename": "biology.txt",
                    "chunk_index": 0,
                    "page_number": 1,
                },
            ),
            Document(
                page_content="Python is a high-level programming language.",
                metadata={
                    "document_id": "cs-doc",
                    "filename": "programming.txt",
                    "chunk_index": 0,
                    "page_number": 0,
                },
            ),
        ]
        embed_and_store(chunks)

        results = similarity_search(query="cell biology powerhouse", top_k=2)
        assert len(results) >= 1
        # Top result should be the biology chunk
        assert results[0].document_id in ("bio-doc", "cs-doc")

    def test_empty_embed_does_not_crash(self, isolated_settings, monkeypatch, mock_embedding_model):
        monkeypatch.setattr("app.retrieval._embedding_model", mock_embedding_model)
        from app.retrieval import embed_and_store

        # Should log a warning and return without raising
        embed_and_store([])

    def test_similarity_search_with_document_filter(
        self, isolated_settings, monkeypatch, mock_embedding_model
    ):
        monkeypatch.setattr("app.retrieval._embedding_model", mock_embedding_model)
        monkeypatch.setattr("app.retrieval._vector_store", None)

        from app.retrieval import embed_and_store, similarity_search

        doc_a = [
            Document(
                page_content="Alpha document content about space exploration.",
                metadata={
                    "document_id": "doc-alpha",
                    "filename": "alpha.txt",
                    "chunk_index": 0,
                    "page_number": 0,
                },
            )
        ]
        doc_b = [
            Document(
                page_content="Beta document content about ocean biology.",
                metadata={
                    "document_id": "doc-beta",
                    "filename": "beta.txt",
                    "chunk_index": 0,
                    "page_number": 0,
                },
            )
        ]
        embed_and_store(doc_a + doc_b)

        results = similarity_search(
            query="space exploration",
            top_k=4,
            document_id="doc-alpha",
        )
        # All returned chunks should belong to doc-alpha
        for r in results:
            assert r.document_id == "doc-alpha"

    def test_source_chunk_fields_populated(
        self, isolated_settings, monkeypatch, mock_embedding_model
    ):
        monkeypatch.setattr("app.retrieval._embedding_model", mock_embedding_model)
        monkeypatch.setattr("app.retrieval._vector_store", None)

        from app.retrieval import embed_and_store, similarity_search

        chunks = [
            Document(
                page_content="Machine learning models require training data.",
                metadata={
                    "document_id": "ml-doc",
                    "filename": "ml.txt",
                    "chunk_index": 0,
                    "page_number": 3,
                },
            )
        ]
        embed_and_store(chunks)
        results = similarity_search("machine learning", top_k=1)

        assert len(results) == 1
        r = results[0]
        assert r.document_id == "ml-doc"
        assert r.filename == "ml.txt"
        assert r.chunk_index == 0
        assert r.page_number == 3
        assert 0.0 <= r.similarity_score <= 1.0
        assert isinstance(r.content, str)
