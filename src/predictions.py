"""Prediction history storage."""
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict

DB_PATH = Path(__file__).parent.parent / "users.db"


def init_db() -> None:
    """Initialize predictions table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_filename TEXT,
            file_hash TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            ensemble_prob REAL NOT NULL,
            effnet_prob REAL NOT NULL,
            densenet_prob REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.commit()
    conn.close()


def add_prediction(
    user_id: int,
    image_filename: Optional[str],
    file_hash: str,
    predicted_class: str,
    confidence: float,
    ensemble_prob: float,
    effnet_prob: float,
    densenet_prob: float,
) -> None:
    """Store a prediction record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            user_id, image_filename, file_hash, predicted_class,
            confidence, ensemble_prob, effnet_prob, densenet_prob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            image_filename,
            file_hash,
            predicted_class,
            confidence,
            ensemble_prob,
            effnet_prob,
            densenet_prob,
        ),
    )
    conn.commit()
    conn.close()


def list_predictions(user_id: int, limit: int = 50) -> List[Dict]:
    """Return recent predictions for a user."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, image_filename, file_hash, predicted_class, confidence,
               ensemble_prob, effnet_prob, densenet_prob, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "image_filename": row["image_filename"],
            "file_hash": row["file_hash"],
            "predicted_class": row["predicted_class"],
            "confidence": row["confidence"],
            "ensemble_prob": row["ensemble_prob"],
            "effnet_prob": row["effnet_prob"],
            "densenet_prob": row["densenet_prob"],
            "created_at": row["created_at"],
        })
    return results


init_db()
