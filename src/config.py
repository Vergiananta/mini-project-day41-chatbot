import os
from dotenv import load_dotenv, find_dotenv


def load_config():
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path=dotenv_path, override=True)
    cfg = {
        "PG_HOST": os.getenv("PG_HOST", "localhost"),
        "PG_PORT": int(os.getenv("PG_PORT", "5430")),
        "PG_DB": os.getenv("PG_DB", "customer_kb"),
        "PG_USER": os.getenv("PG_USER", "postgres"),
        "PG_PASSWORD": os.getenv("DB_PASSWORD", ""),
        "EMBEDDING_MODEL_NAME": os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "DEFAULT_DISTANCE_METRIC": os.getenv("DEFAULT_DISTANCE_METRIC", "cosine"),
        "DEFAULT_TOP_K": int(os.getenv("DEFAULT_TOP_K", "5")),
        "SEMANTIC_WEIGHT": float(os.getenv("SEMANTIC_WEIGHT", "0.7")),
        "LEXICAL_WEIGHT": float(os.getenv("LEXICAL_WEIGHT", "0.3")),
    }
    return cfg