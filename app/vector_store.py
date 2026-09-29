import os

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    conn = psycopg.connect(DATABASE_URL, connect_timeout=5)

    register_vector(conn)

    return conn


def store_chunks(chunks: list[dict], embeddings: list[list[float]]):
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must be the same.")

    if not chunks:
        return {"chunks_stored": 0}

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunks WHERE source = %s", (chunks[0]["source"],)
            )

            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        source,
                        text,
                        page_numbers,
                        start_offset,
                        end_offset,
                        embedding
                    )
                    VALUES (%s,%s,%s,%s,%s,%s)
                """,
                    (
                        chunk["source"],
                        chunk["text"],
                        chunk["page_numbers"],
                        chunk["start"],
                        chunk["end"],
                        Vector(embedding),
                    ),
                )
    return {"chunks_stored": len(chunks)}


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
