import os
import re
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.db import Database
from src.embeddings import EmbeddingService
from src.utils.logger import get_logger


def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def guess_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["refund", "return", "policy"]):
        return "policy"
    if any(k in t for k in ["error", "issue", "troubleshoot", "bug"]):
        return "troubleshooting"
    if any(k in t for k in ["contact", "support", "help"]):
        return "contact"
    if any(k in t for k in ["price", "payment", "billing"]):
        return "faq"
    return "general"


def extract_tags(text: str) -> List[str]:
    keywords = ["account", "payment", "delivery", "refund", "error", "login", "shipping", "support", "policy"]
    t = text.lower()
    tags = sorted({k for k in keywords if k in t})
    return tags or ["general"]


def ingest_csv(path: str = "dataset/dataset_assignment.csv"):
    logger = get_logger("ingest")
    cfg = load_config()
    db = Database(cfg)
    emb = EmbeddingService(cfg["EMBEDDING_MODEL_NAME"])

    db.init_db(embedding_dim=emb.dim, metric=cfg["DEFAULT_DISTANCE_METRIC"])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)
    possible_text_cols = [c for c in df.columns if c.lower() in ["text", "content", "question", "answer", "kb"]]
    text_col = possible_text_cols[0] if possible_text_cols else df.columns[0]

    category_col = next((c for c in df.columns if c.lower() == "category"), None)
    tags_col = next((c for c in df.columns if c.lower() == "tags"), None)

    texts = [clean_text(str(x)) for x in df[text_col].tolist()]

    categories = [str(x) if category_col else guess_category(t) for x, t in zip(df[category_col].tolist() if category_col else [None] * len(texts), texts)]
    all_tags = []
    if tags_col:
        for x in df[tags_col].tolist():
            if isinstance(x, str) and x.strip():
                all_tags.append([t.strip() for t in x.split(",")])
            else:
                all_tags.append(["general"])
    else:
        all_tags = [extract_tags(t) for t in texts]

    if len(texts) < 50:
        logger.warning(f"Hanya {len(texts)} entries. Disarankan >= 50 untuk kualitas.")

    logger.info("Menghitung embeddings (batch)...")
    vectors = emb.embed(texts, normalize=True, batch_size=64)

    rows: List[Tuple[str, List[str], str, List[float]]] = []
    for cat, tags, content, vec in tqdm(zip(categories, all_tags, texts, vectors), total=len(texts)):
        rows.append((cat or "general", tags, content, vec))

    logger.info("Mengunggah data ke Postgres (pgvector)...")
    db.upsert_entries(rows)
    logger.info("Selesai ingestion.")


if __name__ == "__main__":
    ingest_csv()