import psycopg2
import psycopg2.extras
from typing import List, Optional, Tuple, Dict, Any
from src.utils.logger import get_logger


OPS_MAPPING = {
    "cosine": "vector_cosine_ops",
    "euclidean": "vector_l2_ops",
    "ip": "vector_ip_ops",
}


class Database:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.conn = psycopg2.connect(
            host=cfg["PG_HOST"],
            port=cfg["PG_PORT"],
            dbname=cfg["PG_DB"],
            user=cfg["PG_USER"],
            password=cfg["PG_PASSWORD"],
        )
        self.conn.autocommit = True
        self.logger = get_logger("Database")

    def init_db(self, embedding_dim: int = 384, metric: str = "cosine"):
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create table for knowledge base
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS kb_entries (
                    id SERIAL PRIMARY KEY,
                    category TEXT,
                    tags TEXT[],
                    content TEXT NOT NULL,
                    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    embedding vector({embedding_dim})
                );
                """
            )

            # Lexical search index
            cur.execute(
                "CREATE INDEX IF NOT EXISTS kb_content_tsv_idx ON kb_entries USING GIN (content_tsv);"
            )

            # Vector index (IVFFlat)
            ops = OPS_MAPPING.get(metric, OPS_MAPPING["cosine"])

            cur.execute(
                f"CREATE INDEX IF NOT EXISTS kb_embedding_ivf_idx ON kb_entries USING ivfflat (embedding {ops}) WITH (lists = 100);"
            )

            # Try HNSW if available (catch if unsupported)
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS kb_embedding_hnsw_idx ON kb_entries USING hnsw (embedding {ops}) WITH (m = 16, ef_construction = 64);"
                )
            except Exception:
                self.logger.info("HNSW index not available; skipping")

    def clear_all(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE kb_entries RESTART IDENTITY;")

    def upsert_entries(self, rows: List[Tuple[str, List[str], str, List[float]]]):
        logger = get_logger("ingest")
        logger.info(f"Upserting {len(rows)} entries to kb_entries")
        logger.info(f"Sample row: {rows[0]}")
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO kb_entries (category, tags, content, embedding)
                VALUES (%s, %s, %s, %s::vector);
                """,
                [(cat, tags, content, self._vector_literal(emb)) for cat, tags, content, emb in rows],
                page_size=500,
            )

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
        category: Optional[str] = None,
        metric: str = "cosine",
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        ops = OPS_MAPPING.get(metric, OPS_MAPPING["cosine"])
        vector_literal = self._vector_literal(query_vector)

        where_cat = "" if not category else "AND category = %s"

        # For cosine distance, convert to similarity by (1 - distance)
        semantic_score_sql = (
            "(1 - (embedding <-> %s::vector))" if metric == "cosine" else "(1 / (1 + (embedding <-> %s::vector)))"
        )

        sql = f"""
            SELECT id, category, tags, content,
                   {semantic_score_sql} AS semantic_score,
                   ts_rank(content_tsv, plainto_tsquery(%s)) AS lexical_score,
                   {semantic_weight} * {semantic_score_sql} + {lexical_weight} * ts_rank(content_tsv, plainto_tsquery(%s)) AS rank
            FROM kb_entries
            WHERE TRUE {where_cat}
            ORDER BY rank DESC
            LIMIT %s;
        """

        params = [vector_literal, query_text, vector_literal, query_text]
        if category:
            params.append(category)
        params.append(top_k)

        results: List[Dict[str, Any]] = []
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                item = dict(row)
                if threshold is not None and item["rank"] < threshold:
                    continue
                results.append(item)
        return results

    @staticmethod
    def _vector_literal(vec: List[float]) -> str:
        return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"