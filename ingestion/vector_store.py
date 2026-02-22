import chromadb
from chromadb.utils import embedding_functions
from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR


def get_existing_vector_store():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.Client(
        settings=chromadb.Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            is_persistent=True
        )
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=openai_ef
    )
    return collection


def create_vector_store(chunks):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.Client(
        settings=chromadb.Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            is_persistent=True
        )
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=openai_ef
    )
    collection.add(
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        ids=[chunk["id"] for chunk in chunks]
    )
    return collection
