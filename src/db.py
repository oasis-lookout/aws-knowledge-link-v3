"""PostgreSQL + pgvector access (pooled connections)."""
from contextlib import contextmanager

from psycopg2 import pool
from psycopg2.extras import Json, execute_values

from .config import cfg, cfg_int
from .providers import embed_dim

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(
            1,
            cfg_int("DB_POOL_MAX", 5),
            host=cfg("DB_HOST"),
            port=cfg("DB_PORT", "5432"),
            dbname=cfg("DB_NAME", "postgres"),
            user=cfg("DB_USER", "postgres"),
            password=cfg("DB_PASSWORD"),
            sslmode=cfg("DB_SSLMODE", "prefer"),
            connect_timeout=10,
        )
    return _pool


@contextmanager
def conn():
    p = _get_pool()
    c = p.getconn()
    if c.closed:  # stale connection (e.g. idle timeout)
        p.putconn(c, close=True)
        c = p.getconn()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        p.putconn(c)


def vec_literal(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


# ------------------------------------------------------------------ schema
def existing_dim(table="csv_documents"):
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
            (table,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def init_schema(reset=False):
    dim = embed_dim()
    with conn() as c, c.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if reset:
            cur.execute("DROP TABLE IF EXISTS csv_documents")
            cur.execute("ALTER TABLE IF EXISTS chat_history DROP COLUMN IF EXISTS embedding")
    have = existing_dim()
    if have is not None and have != dim:
        raise RuntimeError(
            f"csv_documents.embedding is vector({have}) but your embedding setup uses {dim}. "
            "Run: python setup_db.py --reset   (this drops and recreates csv_documents)"
        )
    with conn() as c, c.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS csv_documents (
                id BIGSERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                row_index INT NOT NULL,
                company_id TEXT,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector({dim}),
                UNIQUE (file_name, row_index)
            )"""
        )
        cur.execute("CREATE INDEX IF NOT EXISTS csv_documents_company_idx ON csv_documents (company_id)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT,
                company_id TEXT,
                user_query TEXT,
                bot_response TEXT,
                retrieved_sources JSONB,
                top_similarity REAL,
                latency_ms INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        cur.execute("CREATE INDEX IF NOT EXISTS chat_history_session_idx ON chat_history (session_id)")
        # conversation memory: embedding of each Q&A turn + whether it was answered from records
        cur.execute(f"ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS embedding vector({dim})")
        cur.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS grounded BOOLEAN")


def ensure_vector_index():
    """Build the HNSW index after bulk load (much faster than indexing while inserting)."""
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS csv_documents_emb_idx "
            "ON csv_documents USING hnsw (embedding vector_cosine_ops)"
        )
        cur.execute("ANALYZE csv_documents")


# ------------------------------------------------------------------ writes
def upsert_documents(rows):
    """rows: (file_name, row_index, company_id, content, metadata_dict, embedding_list)"""
    values = [(f, i, co, txt, Json(meta), vec_literal(vec)) for f, i, co, txt, meta, vec in rows]
    with conn() as c, c.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO csv_documents (file_name, row_index, company_id, content, metadata, embedding)
            VALUES %s
            ON CONFLICT (file_name, row_index) DO UPDATE SET
                company_id = EXCLUDED.company_id,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """,
            values,
            template="(%s, %s, %s, %s, %s, %s::vector)",
        )


def save_chat(session_id, company_id, query, response, sources, top_similarity, latency_ms,
              embedding=None, grounded=True):
    """Persist one Q&A turn. `embedding` (of 'Q: ..\\nA: ..') makes it searchable as future context."""
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO chat_history
               (session_id, company_id, user_query, bot_response, retrieved_sources,
                top_similarity, latency_ms, embedding, grounded)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)""",
            (session_id, company_id, query, response, Json(sources), top_similarity, latency_ms,
             vec_literal(embedding) if embedding else None, grounded),
        )


# ------------------------------------------------------------------- reads
def search(query_vec, top_k=5, company_id=None):
    """Cosine-similarity search, optionally restricted to one company (tenant)."""
    lit = vec_literal(query_vec)
    sql = "SELECT content, metadata, 1 - (embedding <=> %s::vector) AS similarity FROM csv_documents"
    params = [lit]
    if company_id:
        sql += " WHERE company_id = %s"
        params.append(company_id)
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params += [lit, top_k]
    with conn() as c, c.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = 100")
        cur.execute(sql, params)
        return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]


def list_companies():
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT DISTINCT company_id FROM csv_documents WHERE company_id IS NOT NULL ORDER BY 1")
        return [r[0] for r in cur.fetchall()]


def load_history(session_id):
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT user_query, bot_response, created_at FROM chat_history "
            "WHERE session_id = %s ORDER BY created_at ASC",
            (session_id,),
        )
        return cur.fetchall()


def search_past(query_vec, top_k=3, company_id=None, exclude_session=None):
    """Find earlier answered Q&A turns similar to the question (conversation memory)."""
    lit = vec_literal(query_vec)
    sql = (
        "SELECT user_query, bot_response, created_at, 1 - (embedding <=> %s::vector) AS similarity "
        "FROM chat_history WHERE embedding IS NOT NULL AND grounded IS TRUE"
    )
    params = [lit]
    if company_id:
        sql += " AND company_id = %s"
        params.append(company_id)
    if exclude_session:
        sql += " AND session_id <> %s"
        params.append(exclude_session)
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params += [lit, top_k]
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, params)
        return [(r[0], r[1], r[2], float(r[3])) for r in cur.fetchall()]


def list_sessions(company_id=None, limit=25):
    """Recent conversations: (session_id, started_at, n_turns, first_question)."""
    sql = (
        "SELECT session_id, min(created_at), count(*), "
        "(array_agg(user_query ORDER BY created_at))[1] FROM chat_history"
    )
    params = []
    if company_id:
        sql += " WHERE company_id = %s"
        params.append(company_id)
    sql += " GROUP BY session_id ORDER BY max(created_at) DESC LIMIT %s"
    params.append(limit)
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()
