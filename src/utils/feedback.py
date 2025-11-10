import csv
import os
from datetime import datetime
from typing import Dict, Optional


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "feedback.csv")


def _ensure_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "query",
                "source_id",
                "category",
                "tags",
                "rank",
                "action",
                "rating",
                "comment",
            ])


def log_feedback(query: str, source: Dict, action: str, rating: Optional[int] = None, comment: Optional[str] = None):
    _ensure_log_file()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            query,
            source.get("id"),
            source.get("category", ""),
            ",".join(source.get("tags", []) or []),
            round(source.get("rank", 0), 6),
            action,
            rating if rating is not None else "",
            (comment or "").strip(),
        ])