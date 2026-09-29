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

    if k <= 0:
        return []

    query_vector = Vector(query_embedding)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    id,
                    source,
                    text,
                    page_numbers,
                    start_offset,
                    end_offset,
                    embedding <=> %s AS cosine distance
                FROM chunks
                ORDER BY embedding <=>%s
                LIMIT %s
    """,
                (query_vector, query_vector, k),
            )

            rows = cursor.fetchall()

    relevant_chunks = []

    for row in rows:
        (
            chunk_id,
            source,
            text,
            page_numbers,
            start_offset,
            end_offset,
            cosine_distance,
        ) = row

        relevant_chunks.append(
            {
                "id": chunk_id,
                "source": source,
                "text": text,
                "page_numbers": page_numbers,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "cosine_distance": float(cosine_distance),
            }
        )

    return relevant_chunks
