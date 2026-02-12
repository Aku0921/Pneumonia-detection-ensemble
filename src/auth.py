"""
Authentication module for user management.
"""
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
import sqlite3
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "users.db"


def init_db():
    """Initialize the database with users table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, email: str, password: str) -> Optional[int]:
    """Create a new user. Returns user_id or None if username/email exists."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None


def verify_user(username: str, password: str) -> Optional[dict]:
    """Verify username and password. Returns user dict or None."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {"id": row[0], "username": row[1], "email": row[2]}
    return None


def create_session(user_id: int, duration_hours: int = 24) -> str:
    """Create a new session for a user. Returns session_id."""
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=duration_hours)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (session_id, user_id, expires_at) VALUES (?, ?, ?)",
        (session_id, user_id, expires_at)
    )
    conn.commit()
    conn.close()
    return session_id


def get_session_user(session_id: str) -> Optional[dict]:
    """Get user from session_id if session is valid."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.username, u.email 
        FROM sessions s 
        JOIN users u ON s.user_id = u.id 
        WHERE s.session_id = ? AND s.expires_at > datetime('now')
    """, (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {"id": row[0], "username": row[1], "email": row[2]}
    return None


def delete_session(session_id: str):
    """Delete a session (logout)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def cleanup_expired_sessions():
    """Remove expired sessions from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE expires_at < datetime('now')")
    conn.commit()
    conn.close()


# Initialize database on import
init_db()
