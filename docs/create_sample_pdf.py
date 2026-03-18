"""
Script to create a sample PDF for testing RAGForge.

Run this once to generate docs/sample.pdf:
    python docs/create_sample_pdf.py

Requires: pip install reportlab
"""

from pathlib import Path


def create_sample_pdf(output_path: str = "docs/sample.pdf") -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        print("reportlab not installed. Creating a text-based fallback.")
        _create_txt_fallback(output_path.replace(".pdf", "_sample.txt"))
        return

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    content = [
        ("RAG and Document Intelligence — A Primer", "Title"),
        (
            "Retrieval-Augmented Generation (RAG) is a technique in natural language processing "
            "that combines information retrieval with language model generation. Instead of relying "
            "solely on a language model's parametric knowledge, RAG retrieves relevant context from "
            "an external knowledge base and uses it to ground the generated answer.",
            "BodyText",
        ),
        ("What is a RAG Pipeline?", "Heading1"),
        (
            "A RAG pipeline consists of three main components: an embedding model that converts "
            "text into dense vector representations, a vector store that indexes these embeddings "
            "for similarity search, and a language model that generates answers from retrieved context.",
            "BodyText",
        ),
        ("Chunking and Overlap", "Heading1"),
        (
            "Document chunking is the process of splitting large documents into smaller pieces "
            "that fit within the context window of an embedding model. Chunk overlap ensures that "
            "context at the boundary between chunks is not lost. A typical chunk size is 500 tokens "
            "with 50 tokens of overlap. The overlap creates redundancy that improves recall for "
            "queries that touch chunk boundaries.",
            "BodyText",
        ),
        ("Embedding Models", "Heading1"),
        (
            "Embedding models convert text into dense numerical vectors that capture semantic meaning. "
            "Sentence-transformers are a popular family of embedding models. The all-MiniLM-L6-v2 model "
            "produces 384-dimensional embeddings and runs on CPU without requiring a GPU. It achieves "
            "strong performance on semantic similarity benchmarks.",
            "BodyText",
        ),
        ("Cosine Similarity and Vector Search", "Heading1"),
        (
            "Cosine similarity measures the angle between two vectors in high-dimensional space. "
            "A cosine similarity of 1.0 means the vectors are identical in direction; 0.0 means they "
            "are orthogonal (unrelated). Vector databases use approximate nearest-neighbour (ANN) "
            "algorithms to find the top-k most similar vectors to a query vector efficiently, even "
            "over millions of documents.",
            "BodyText",
        ),
        ("ChromaDB as a Vector Store", "Heading1"),
        (
            "ChromaDB is an open-source vector database that runs in-process as a Python library. "
            "It persists data to a local directory and supports filtered metadata queries. ChromaDB "
            "is a good choice for development and small-scale production deployments because it "
            "requires no external infrastructure.",
            "BodyText",
        ),
        ("Answer Synthesis", "Heading1"),
        (
            "After retrieval, the top-k chunks are assembled into a context block and sent to a "
            "language model along with the original question. The language model generates an answer "
            "that is grounded in the retrieved context. Constraining the model to use only the "
            "provided context reduces hallucination.",
            "BodyText",
        ),
    ]

    for text, style in content:
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"Created: {output_path}")


def _create_txt_fallback(path: str) -> None:
    content = """RAG and Document Intelligence — A Primer

Retrieval-Augmented Generation (RAG) is a technique in natural language processing
that combines information retrieval with language model generation. Instead of relying
solely on a language model's parametric knowledge, RAG retrieves relevant context from
an external knowledge base and uses it to ground the generated answer.

What is a RAG Pipeline?

A RAG pipeline consists of three main components: an embedding model that converts
text into dense vector representations, a vector store that indexes these embeddings
for similarity search, and a language model that generates answers from retrieved context.

Chunking and Overlap

Document chunking is the process of splitting large documents into smaller pieces
that fit within the context window of an embedding model. Chunk overlap ensures that
context at the boundary between chunks is not lost. A typical chunk size is 500 tokens
with 50 tokens of overlap.

Embedding Models

Sentence-transformers are a popular family of embedding models. The all-MiniLM-L6-v2
model produces 384-dimensional embeddings and runs on CPU. Cosine similarity measures
the angle between two vectors.

ChromaDB as a Vector Store

ChromaDB is an open-source vector database that runs in-process as a Python library.
It persists data to a local directory and supports filtered metadata queries.

Answer Synthesis

After retrieval, the top-k chunks are assembled into a context block and sent to a
language model along with the original question. The language model generates an answer
that is grounded in the retrieved context.
"""
    Path(path).write_text(content, encoding="utf-8")
    print(f"Created text fallback: {path}")


if __name__ == "__main__":
    create_sample_pdf()
