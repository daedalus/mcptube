"""Persistent caches for avoiding redundant calls."""

import hashlib
import json
import logging
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable

from mcptube.config import settings

logger = logging.getLogger(__name__)

_FRAME_CACHE_DB = "frame_cache.db"
_PROMPT_CACHE_DB = "prompt_cache.db"
_SUBTITLE_CACHE_DB = "subtitle_cache.db"


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
                {
                    "array_size": self.size,
                    "hash_count": self.hash_count,
                    "array": self.array,
                },
                f,
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


class SQLiteCache:
    """Generic SQLite cache with bloom filter for fast pre-filtering."""

    def __init__(
        self,
        db_path: Path,
        table: str,
        key_col: str,
        value_col: str,
        key_hash: Callable[[str], str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.bloom_path = str(db_path) + ".bloom"
        self.table = table
        self.key_col = key_col
        self.value_col = value_col
        self.key_hash = key_hash or (lambda x: x)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._key_set: set[str] = set()
        self._hits = 0
        self._misses = 0
        self.bloom = BloomFilter()
        self._initialize()
        self._load()
        logger.info("Cache initialized: %s", db_path)

    def _initialize(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                {self.key_col} TEXT PRIMARY KEY,
                {self.value_col} TEXT NOT NULL
            )
        """)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.key_col} ON {self.table}({self.key_col})
        """)
        self._conn.commit()

    def _load(self) -> None:
        cursor = self._conn.execute(f"SELECT {self.key_col} FROM {self.table}")
        self._key_set = {row[0] for row in cursor.fetchall()}
        if self._key_set and os.path.exists(self.bloom_path):
            self.bloom = BloomFilter.load(self.bloom_path)
        else:
            self.bloom = BloomFilter(capacity=max(1000, len(self._key_set) * 10))
            for k in self._key_set:
                self.bloom.add(k)

    def flush(self) -> None:
        self._conn.commit()
        self.bloom.save(self.bloom_path)

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    def _compute_hash(self, key: str) -> str:
        """Compute hash for key. Override in subclasses."""
        return self.key_hash(key)

    @property
    def _hash_set(self) -> set[str]:
        """Access internal key set (for tests)."""
        return self._key_set

    def get(self, key: str) -> Any | None:
        """Get cached value for key, or None if not cached."""
        cache_key = self.key_hash(key)

        if cache_key in self.bloom:
            if cache_key in self._key_set:
                cursor = self._conn.execute(
                    f"SELECT {self.value_col} FROM {self.table} WHERE {self.key_col} = ?",
                    (cache_key,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[self.value_col])
            return None

        return None

    def put(self, key: str, value: Any) -> None:
        """Store key → value mapping."""
        cache_key = self.key_hash(key)
        self._key_set.add(cache_key)
        value_json = json.dumps(value)
        try:
            self._conn.execute(
                f"INSERT INTO {self.table} ({self.key_col}, {self.value_col}) VALUES (?, ?)",
                (cache_key, value_json),
            )
        except sqlite3.IntegrityError:
            return

        self.bloom.add(cache_key)

    def close(self) -> None:
        self.flush()
        self._conn.close()


class FrameCacheDB(SQLiteCache):
    """Persistent cache for frame descriptions indexed by image content hash."""

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(
            db_path or settings.data_dir / _FRAME_CACHE_DB,
            "frame_descriptions",
            "content_hash",
            "description",
        )

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

        result = super().get(content_hash)
        if result is not None:
            self._hits += 1
            logger.debug(
                "Frame cache hit: %s -> %s", image_path.name, content_hash[:16]
            )
        else:
            self._misses += 1
            logger.debug("Frame cache miss: %s", image_path.name)
        return result

    def put(self, image_path: Path, description: str) -> None:
        """Store image hash → description mapping."""
        try:
            content_hash = self.compute_hash(image_path)
        except (OSError, IOError):
            return
        super().put(content_hash, description)
        logger.debug("Frame cached: %s -> %s", image_path.name, content_hash[:16])


class PromptCacheDB(SQLiteCache):
    """Persistent cache for LLM prompt responses indexed by prompt hash."""

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(
            db_path or settings.data_dir / _PROMPT_CACHE_DB,
            "prompt_responses",
            "prompt_hash",
            "response",
            key_hash=lambda x: hashlib.sha256(x.encode()).hexdigest(),
        )

    def get(self, prompt: str) -> str | None:
        """Get cached response for prompt, or None if not cached."""
        result = super().get(prompt)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, prompt: str, response: str) -> None:
        """Store prompt hash → response mapping."""
        super().put(prompt, response)


class SubtitleCacheDB(SQLiteCache):
    """Persistent cache for subtitles indexed by video ID."""

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(
            db_path or settings.data_dir / _SUBTITLE_CACHE_DB,
            "subtitles",
            "video_id",
            "transcript",
        )

    def get(self, video_id: str) -> list[dict] | None:
        """Get cached transcript for video ID, or None if not cached."""
        result = super().get(video_id)
        if result is not None:
            logger.debug("Subtitle cache hit: %s", video_id)
        else:
            logger.debug("Subtitle cache miss: %s", video_id)
        return result

    def put(self, video_id: str, transcript: list[Any]) -> None:
        """Store video ID → transcript mapping."""
        super().put(video_id, [s.model_dump() for s in transcript])
        logger.debug("Subtitle cached: %s", video_id)
