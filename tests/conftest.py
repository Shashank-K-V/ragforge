"""
conftest.py — Shared pytest fixtures for RAGForge tests.

Fixtures defined here are automatically available in all test files
without any imports.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch, tmp_path):
    """
    Ensure every test runs with:
    - A fresh temporary ChromaDB directory (no cross-test pollution)
    - A fresh temporary upload directory
    - Dummy API keys so settings validation passes

    autouse=True means this runs for every test automatically.
    """
    chroma_dir = str(tmp_path / "chroma")
    upload_dir = str(tmp_path / "uploads")

    monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
    monkeypatch.setenv("UPLOAD_DIR", upload_dir)

    # Provide a dummy HF key so settings don't raise on validation
    if not os.environ.get("HUGGINGFACE_API_KEY"):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_dummy")

    # Reset singletons that cache settings-derived state
    monkeypatch.setattr("app.retrieval._vector_store", None)
    monkeypatch.setattr("app.retrieval._embedding_model", None)
    monkeypatch.setattr("app.pipeline._llm_instance", None)

    # Patch the registry path to point to tmp_path
    from pathlib import Path
    monkeypatch.setattr(
        "app.retrieval._REGISTRY_PATH",
        Path(chroma_dir) / "document_registry.json",
    )
