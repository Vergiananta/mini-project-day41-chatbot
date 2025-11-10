from typing import List, Dict, Any, Optional

from src.config import load_config
from src.db import Database
from src.embeddings import EmbeddingService


class SemanticSearch:
    def __init__(self):
        self.cfg = load_config()
        self.db = Database(self.cfg)
        self.emb = EmbeddingService(self.cfg["EMBEDDING_MODEL_NAME"])
        self.db.init_db(embedding_dim=self.emb.dim, metric=self.cfg["DEFAULT_DISTANCE_METRIC"])

    def query(
        self,
        text: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.cfg["DEFAULT_TOP_K"]
        metric = metric or self.cfg["DEFAULT_DISTANCE_METRIC"]

        vec = self.emb.embed_one(text, normalize=True)
        results = self.db.search(
            vec,
            text,
            top_k=top_k,
            category=category,
            metric=metric,
            semantic_weight=self.cfg["SEMANTIC_WEIGHT"],
            lexical_weight=self.cfg["LEXICAL_WEIGHT"],
            threshold=threshold,
        )
        return results
