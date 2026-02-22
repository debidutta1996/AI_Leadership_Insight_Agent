import os
import json
from ingestion.pdf_loader import load_pdfs
from ingestion.semantic_chunker import semantic_chunk_documents
from ingestion.vector_store import create_vector_store, get_existing_vector_store
from generation.answer_generator import generate_answer
from retrieval.retriever import hybrid_retrieve
from retrieval.reranker import rerank

CHROMA_STORE_PATH = "chroma_store"  # or your configured chroma persist path
CHUNK_CACHE_PATH = "data/chunk_cache.json"

def chroma_store_exists():
    return os.path.exists(CHROMA_STORE_PATH) and os.listdir(CHROMA_STORE_PATH)

def main():
    if chroma_store_exists():
        print("Chroma vector store exists; skipping ingestion, chunking, and embedding. You can query instantly!")
        collection = get_existing_vector_store()
        # Load cached chunks for BM25
        with open(CHUNK_CACHE_PATH, "r") as f:
            chunks = json.load(f)
        # --- Q&A Loop ---
        while True:
            question = input("\nAsk leadership question (type 'exit'): ")
            if question.lower() == "exit":
                break
            docs, metadatas, raw_scores = hybrid_retrieve(question, collection, chunks)
            reranked_docs, reranked_metas, reranked_scores = rerank(
                question,
                docs,
                metadatas,
                top_k=4
            )
            answer = generate_answer(
                question,
                reranked_docs,
                reranked_metas,
                reranked_scores
            )
            print("\n", answer)
        return

    # If Chroma DB doesn't exist, run the full ingestion + chunking + embedding pipeline
    print("No existing chroma_store; running full ingestion, chunking, and indexing.")
    print("Loading PDFs...")
    documents = load_pdfs("data/documents")
    print(f"Loaded documents: {len(documents)}")
    for doc in documents:
        print(f"{doc['id']} -> text length: {len(doc['text'])}")

    if not documents:
        print("❌ No new or changed PDF documents to process or no extractable text found. Exiting.")
        return

    print("\nSemantic chunking...")
    chunks = semantic_chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("❌ No chunks created. Check PDF extraction or chunking logic.")
        return

    print("\nCreating Chroma vector store...")
    collection = create_vector_store(chunks)

    # Store chunk cache for future runs
    with open(CHUNK_CACHE_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    while True:
        question = input("\nAsk leadership question (type 'exit'): ")
        if question.lower() == "exit":
            break
        docs, metadatas, raw_scores = hybrid_retrieve(question, collection, chunks)
        reranked_docs, reranked_metas, reranked_scores = rerank(
            question,
            docs,
            metadatas,
            top_k=4
        )
        answer = generate_answer(
            question,
            reranked_docs,
            reranked_metas,
            reranked_scores
        )
        print("\n", answer)

if __name__ == "__main__":
    main()