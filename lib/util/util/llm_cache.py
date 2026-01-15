import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from env_util import ENV
from util.types import ChatMessage


class LLMCache:
    def __init__(self, db_path: str | None = None):
        print("WARN: util/llm_cache.py is deprecated; migrate to util/llm/llm_cache.py instead.")

        if db_path is None:
            if ENV.LLM_CACHE_PATH is None:
                raise Exception("LLM_CACHE_PATH is not set in .env")

            cache_dir = Path(ENV.LLM_CACHE_PATH)
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "llm_cache.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    completion TEXT,
                    model_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _create_key(self, messages: list[ChatMessage], model_name: str) -> str:
        """Create a deterministic hash key from messages and model."""
        # Convert messages to a stable string representation
        message_str = json.dumps(
            [{"role": msg["role"], "content": msg["content"]} for msg in messages], sort_keys=True
        )

        # Combine with model name and hash
        key_str = f"{message_str}:{model_name}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, messages: list[ChatMessage], model_name: str) -> str | None:
        """Get cached completion for a conversation if it exists."""
        key = self._create_key(messages, model_name)

        with self._get_connection() as conn:
            cursor = conn.execute("SELECT completion FROM llm_cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else None

    def set(self, messages: list[ChatMessage], model_name: str, completion: str) -> None:
        """Cache a completion for a conversation."""
        key = self._create_key(messages, model_name)

        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, completion, model_name) VALUES (?, ?, ?)",
                (key, completion, model_name),
            )
            conn.commit()

    def clear(self) -> None:
        """Clear all cached completions."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM llm_cache")
            conn.commit()
