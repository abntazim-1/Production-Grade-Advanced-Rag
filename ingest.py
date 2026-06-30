"""
CLI ingestion script.
Usage: python ingest.py --path ./documents
"""
import argparse
import os

from config import get_settings
from ingestion.loaders import load_directory, load_file
from ingestion.chunker import StructureAwareChunker
from ingestion.metadata import MetadataEnricher
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store

settings = get_settings()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    parser.add_argument("--no-metadata", action="store_true", help="Skip metadata enrichment")
    args = parser.parse_args()

    # Load documents
    print(f"Loading from: {args.path}")
    if os.path.isdir(args.path):
        docs = load_directory(args.path)
    else:
        docs = [load_file(args.path)]
    print(f"Loaded {len(docs)} documents")

    # Chunk
    chunker = StructureAwareChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    print(f"Produced {len(all_chunks)} chunks")

    # Enrich metadata
    if not args.no_metadata:
        print("Enriching metadata (summaries, keywords, questions)...")
        enricher = MetadataEnricher()
        all_chunks = enricher.enrich_batch(all_chunks)
        print("Metadata enrichment complete")

    # Embed + index
    print("Building vector index...")
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    vector_store.add(all_chunks)
    vector_store.save()

    print("Building BM25 index...")
    bm25_store = BM25Store()
    bm25_store.build(all_chunks)
    bm25_store.save()

    print(f"\n✅ Done. {len(all_chunks)} chunks indexed and saved.")


if __name__ == "__main__":
    main()
