import numpy as np
from rank_bm25 import BM25Okapi
from config import TOP_K


def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def hybrid_retrieve(query, collection, chunks):

    # -------- VECTOR SEARCH --------
    vector_results = collection.query(
        query_texts=[query],
        n_results=TOP_K * 2
    )

    vector_docs = vector_results["documents"][0]
    vector_metas = vector_results["metadatas"][0]
    vector_distances = vector_results["distances"][0]

    # Convert distance → similarity score
    vector_scores = [1 / (1 + d) for d in vector_distances]

    # -------- BM25 SEARCH --------
    bm25 = build_bm25_index(chunks)
    tokenized_query = tokenize(query)

    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top BM25 results
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:TOP_K * 2]

    bm25_docs = [chunks[i]["text"] for i in bm25_top_indices]
    bm25_metas = [chunks[i]["metadata"] for i in bm25_top_indices]
    bm25_scores = [bm25_scores[i] for i in bm25_top_indices]

    # -------- MERGE RESULTS --------
    merged_docs = vector_docs + bm25_docs
    merged_metas = vector_metas + bm25_metas
    merged_scores = vector_scores + bm25_scores

    return merged_docs, merged_metas, merged_scores