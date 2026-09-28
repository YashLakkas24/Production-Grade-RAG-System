import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    conn = psycopg.connec(DATABASE_URL)

    register_vector(conn)

    return conn


# Global storage for FAISS index and chunk metadata
vector_index = None
stored_chunks = []


def store_chunks(chunks: list[dict], embeddings: list[list[float]]):
    global vector_index, stored_chunks

    stored_chunks = chunks
    embeddings_np = np.array(embeddings).astype("float32")

    dimension = len(embeddings[0])
    vector_index = faiss.IndexFlatL2(dimension)

    vector_index.add(embeddings_np)

    return {"chunks_stored": len(chunks), "vector_count": vector_index.ntotal}


def search_similar(query_embedding: list[float], k: int = 3) -> list[dict]:

    if vector_index is None or vector_index.ntotal == 0:
        return []

    query_np = np.array([query_embedding]).astype("float32")

    k = min(k, vector_index.ntotal)

    distances, indices = vector_index.search(query_np, k)

    relevant_chunks = []

    for distance, index in zip(distances[0], indices[0]):
        if index != -1:
            chunk = stored_chunks[index].copy()

            chunk["distance"] = float(distance)
            relevant_chunks.append(chunk)

    return relevant_chunks
