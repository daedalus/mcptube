"""Persistent hash-based caches for avoiding redundant LLM calls."""

import hashlib
import json
import logging
import math
import os
import sqlite3
from pathlib import Path

from mcptube.config import settings

logger = logging.getLogger(__name__)

_FRAME_CACHE_DB = "frame_cache.db"
_PROMPT_CACHE_DB = "prompt_cache.db"


class BloomFilter:
    """Fast probabilistic duplicate checker using bloom filter."""

    def __init__(self, capacity: int = 100000, false_positive_rate: float = 0.01):
        self.size = self._optimal_size(capacity, false_positive_rate)
        self.hash_count = self._optimal_hash_count(capacity, self.size)
        self.array = [False] * self.size

    def _optimal_size(self, n: int, p: float) -> int:
        return int(-n * math.log(p) / (math.log(2) ** 2))

    def _optimal_hash_count(self, n: int, m: int) -> int:
        return max(1, int((m / n) * math.log(2)))

    def _hashes(self, item: str) -> list:
        result = []
        for i in range(self.hash_count):
            h = hashlib.md5((item + str(i)).encode()).hexdigest()
            result.append(int(h, 16) % self.size)
        return result

    def add(self, item: str):
        for idx in self._hashes(item):
            self.array[idx] = True

    def __contains__(self, item: str) -> bool:
        return all(self.array[idx] for idx in self._hashes(item))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"array_size": self.size, "hash_count": self.hash_count, "array": self.array}, f
            )

    @classmethod
    def load(cls, path: str) -> "BloomFilter":
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            data = json.load(f)
        bf = cls.__new__(cls)
        bf.size = data["array_size"]
        bf.hash_count = data["hash_count"]
        bf.array = data["array"]
        return bf


class FrameCacheDB:
    """Persistent SQLite cache for frame descriptions indexed by image hash.

    Uses a bloom filter for fast pre-filtering before SQLite lookup.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.data_dir / _FRAME_CACHE_DB
        self.bloom_path = str(self.db_path) + ".bloom"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._hash_set = set()
        self._hits = 0
        self._misses = 0
        self.bloom = BloomFilter()
        self._initialize()
        self._load()
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

    def _load(self) -> None:
        cursor = self._conn.execute("SELECT content_hash FROM frame_descriptions")
        self._hash_set = {row[0] for row in cursor.fetchall()}
        if self._hash_set and os.path.exists(self.bloom_path):
            self.bloom = BloomFilter.load(self.bloom_path)
        else:
            self.bloom = BloomFilter(capacity=max(1000, len(self._hash_set) * 2))
            for h in self._hash_set:
                self.bloom.add(h)

    def flush(self) -> None:
        """Persist bloom filter to disk."""
        self._conn.commit()
        self.bloom.save(self.bloom_path)

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
        try:
            content_hash = self.compute_hash(image_path)
        except (OSError, IOError):
            return None

        if content_hash in self.bloom:
            if content_hash in self._hash_set:
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

        self._misses += 1
        logger.debug("Frame cache miss (bloom): %s", image_path.name)
        return None

    def put(self, image_path: Path, description: str) -> None:
        """Store image hash → description mapping."""
        try:
            content_hash = self.compute_hash(image_path)
        except (OSError, IOError):
            return

        self._hash_set.add(content_hash)
        try:
            self._conn.execute(
                "INSERT INTO frame_descriptions (content_hash, description) VALUES (?, ?)",
                (content_hash, description),
            )
        except sqlite3.IntegrityError:
            logger.debug("Frame already cached: %s", image_path.name)
            return

        self.bloom.add(content_hash)
        logger.debug("Frame cached: %s -> %s", image_path.name, content_hash[:16])

    def close(self) -> None:
        self.flush()
        self._conn.close()


class PromptCacheDB:
    """Persistent SQLite cache for LLM prompt responses indexed by prompt hash.

    Uses a bloom filter for fast pre-filtering before SQLite lookup.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.data_dir / _PROMPT_CACHE_DB
        self.bloom_path = str(self.db_path) + ".bloom"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._hash_set = set()
        self._hits = 0
        self._misses = 0
        self.bloom = BloomFilter()
        self._initialize()
        self._load()
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

    def _load(self) -> None:
        cursor = self._conn.execute("SELECT prompt_hash FROM prompt_responses")
        self._hash_set = {row[0] for row in cursor.fetchall()}
        if self._hash_set and os.path.exists(self.bloom_path):
            self.bloom = BloomFilter.load(self.bloom_path)
        else:
            self.bloom = BloomFilter(capacity=max(1000, len(self._hash_set) * 2))
            for h in self._hash_set:
                self.bloom.add(h)

    def flush(self) -> None:
        """Persist bloom filter to disk."""
        self._conn.commit()
        self.bloom.save(self.bloom_path)

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}

    def _compute_hash(self, prompt: str) -> str:
        """Compute SHA256 hash of prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> str | None:
        """Get cached response for prompt, or None if not cached."""
        prompt_hash = self._compute_hash(prompt)

        if prompt_hash in self.bloom:
            if prompt_hash in self._hash_set:
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

        self._misses += 1
        logger.debug("Prompt cache miss (bloom)")
        return None

    def put(self, prompt: str, response: str) -> None:
        """Store prompt hash → response mapping."""
        prompt_hash = self._compute_hash(prompt)
        self._hash_set.add(prompt_hash)
        try:
            self._conn.execute(
                "INSERT INTO prompt_responses (prompt_hash, response) VALUES (?, ?)",
                (prompt_hash, response),
            )
        except sqlite3.IntegrityError:
            logger.debug("Prompt already cached: %s...", prompt[:50])
            return

        self.bloom.add(prompt_hash)
        logger.debug("Prompt cached: %s...", prompt[:50])

    def close(self) -> None:
        self.flush()
        self._conn.close()
