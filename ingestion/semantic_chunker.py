import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

SIMILARITY_THRESHOLD = 0.80


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_chunk_document(document):

    paragraphs = [p.strip() for p in document["text"].split("\n") if p.strip()]
    
    chunks = []
    current_chunk = paragraphs[0]
    current_embedding = get_embedding(current_chunk)

    for para in paragraphs[1:]:

        para_embedding = get_embedding(para)
        similarity = cosine_similarity(current_embedding, para_embedding)

        if similarity > SIMILARITY_THRESHOLD:
            current_chunk += " " + para
            current_embedding = get_embedding(current_chunk)
        else:
            chunks.append(current_chunk)
            current_chunk = para
            current_embedding = para_embedding

    chunks.append(current_chunk)

    return [
        {
            "id": f"{document['id']}_chunk_{i}",
            "text": chunk,
            "metadata": document["metadata"]
        }
        for i, chunk in enumerate(chunks)
    ]


def semantic_chunk_documents(documents):

    all_chunks = []

    for doc in documents:
        doc_chunks = semantic_chunk_document(doc)
        all_chunks.extend(doc_chunks)

    return all_chunks