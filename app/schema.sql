CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,

    source TEXT NOT NULL,

    text TEXT NOT NULL,

    page_numbers INTEGER[] NOT NULL,

    start_offset INTEGER NOT NULL,

    end_offset INTEGER NOT NULL,

    embedding VECTOR(1536) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (start_offset >= 0),
    CHECK (end_offset >= start_offset)
);