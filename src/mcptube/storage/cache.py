"""Persistent hash-based caches for avoiding redundant LLM calls."""

import hashlib
import logging
import sqlite3
from pathlib import Path

from mcptube.config import settings

logger = logging.getLogger(__name__)

_FRAME_CACHE_DB = "frame_cache.db"
_PROMPT_CACHE_DB = "prompt_cache.db"


class FrameCacheDB:
    """Persistent SQLite cache for frame descriptions indexed by image hash."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.data_dir / _FRAME_CACHE_DB
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._hits = 0
        self._misses = 0
        self._initialize()
        logger.info("Frame cache initialized: %s", self.db_path)

    def _initialize(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS frame_descriptions (
                content_hash TEXT PRIMARY KEY,
                description TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_hash ON frame_descriptions(content_hash)
        """)
        self._conn.commit()

    @property
    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return {"hits": self._hits, "misses": self._misses}

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = 0

    def compute_hash(self, image_path: Path) -> str:
        """Compute SHA256 hash of image file content."""
        h = hashlib.sha256()
        h.update(image_path.read_bytes())
        return h.hexdigest()

    def get(self, image_path: Path) -> str | None:
        """Get cached description for image, or None if not cached."""
        content_hash = self.compute_hash(image_path)
        cursor = self._conn.execute(
            "SELECT description FROM frame_descriptions WHERE content_hash = ?",
            (content_hash,),
        )
        row = cursor.fetchone()
        if row:
            self._hits += 1
            logger.debug("Frame cache hit: %s -> %s", image_path.name, content_hash[:16])
            return row["description"]
        self._misses += 1
        logger.debug("Frame cache miss: %s", image_path.name)
        return None

    def put(self, image_path: Path, description: str) -> None:
        """Store image hash → description mapping."""
        content_hash = self.compute_hash(image_path)
        try:
            self._conn.execute(
                "INSERT INTO frame_descriptions (content_hash, description) VALUES (?, ?)",
                (content_hash, description),
            )
            self._conn.commit()
            logger.debug("Frame cached: %s -> %s", image_path.name, content_hash[:16])
        except sqlite3.IntegrityError:
            logger.debug("Frame already cached: %s", image_path.name)

    def close(self) -> None:
        self._conn.close()


class PromptCacheDB:
    """Persistent SQLite cache for LLM prompt responses indexed by prompt hash."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.data_dir / _PROMPT_CACHE_DB
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._hits = 0
        self._misses = 0
        self._initialize()
        logger.info("Prompt cache initialized: %s", self.db_path)

    def _initialize(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_responses (
                prompt_hash TEXT PRIMARY KEY,
                response TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_hash ON prompt_responses(prompt_hash)
        """)
        self._conn.commit()

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}

    def _compute_hash(self, prompt: str) -> str:
        """Compute SHA256 hash of prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> str | None:
        """Get cached response for prompt, or None if not cached."""
        prompt_hash = self._compute_hash(prompt)
        cursor = self._conn.execute(
            "SELECT response FROM prompt_responses WHERE prompt_hash = ?",
            (prompt_hash,),
        )
        row = cursor.fetchone()
        if row:
            self._hits += 1
            logger.debug("Prompt cache hit: %s...", prompt[:50])
            return row["response"]
        self._misses += 1
        logger.debug("Prompt cache miss")
        return None

    def put(self, prompt: str, response: str) -> None:
        """Store prompt hash → response mapping."""
        prompt_hash = self._compute_hash(prompt)
        try:
            self._conn.execute(
                "INSERT INTO prompt_responses (prompt_hash, response) VALUES (?, ?)",
                (prompt_hash, response),
            )
            self._conn.commit()
            logger.debug("Prompt cached: %s...", prompt[:50])
        except sqlite3.IntegrityError:
            pass

    def close(self) -> None:
        self._conn.close()
