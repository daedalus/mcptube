"""Tests for storage cache module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from mcptube.storage.cache import BloomFilter, FrameCacheDB, PromptCacheDB


class TestBloomFilter:
    def test_initializes_with_optimal_size(self):
        bf = BloomFilter(capacity=1000)
        assert bf.size > 0
        assert bf.hash_count > 0

    def test_add_and_check(self):
        bf = BloomFilter(capacity=1000)
        bf.add("test-item")
        assert "test-item" in bf

    def test_non_member_returns_false(self):
        bf = BloomFilter(capacity=1000)
        assert "neither-added" not in bf

    def test_save_and_load(self, tmp_path):
        bf = BloomFilter(capacity=1000)
        bf.add("item1")
        path = tmp_path / "bloom.json"
        bf.save(str(path))
        loaded = BloomFilter.load(str(path))
        assert "item1" in loaded

    def test_load_nonexistent_returns_new(self, tmp_path):
        loaded = BloomFilter.load(str(tmp_path / "nonexistent.json"))
        assert loaded.size > 0


class TestFrameCacheDB:
    @pytest.fixture
    def cache(self, tmp_path):
        return FrameCacheDB(tmp_path / "cache.db")

    def test_initializes_table(self, cache):
        cursor = cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='frame_descriptions'"
        )
        assert cursor.fetchone() is not None

    def test_stats_initial(self, cache):
        assert cache.stats == {"hits": 0, "misses": 0}

    def test_compute_hash_deterministic(self, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg content")
        h1 = cache.compute_hash(path)
        h2 = cache.compute_hash(path)
        assert h1 == h2
        assert len(h1) == 64

    def test_put_and_get_roundtrip(self, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg content")
        cache.put(path, "A slide showing neural network")
        desc = cache.get(path)
        assert desc == "A slide showing neural network"

    def test_get_returns_none_for_missing(self, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg content")
        desc = cache.get(path)
        assert desc is None

    def test_different_images_different_hashes(self, cache, tmp_path):
        path1 = tmp_path / "frame1.jpg"
        path1.write_bytes(b"image 1")
        path2 = tmp_path / "frame2.jpg"
        path2.write_bytes(b"image 2")
        assert cache.compute_hash(path1) != cache.compute_hash(path2)

    def test_stats_track_hits_and_misses(self, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"content")
        cache.get(path)
        assert cache.stats["misses"] == 1
        cache.put(path, "description")
        cache.get(path)
        assert cache.stats["hits"] == 1

    def test_bloom_filter_used(self, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"content")
        cache.put(path, "desc")
        content_hash = cache.compute_hash(path)
        assert content_hash in cache.bloom

    def test_flush_persists_bloom(self, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"content")
        cache.put(path, "desc")
        cache.flush()
        assert Path(cache.bloom_path).exists()

    def test_close_calls_flush(self, cache):
        cache.flush = MagicMock()
        cache.close()
        cache.flush.assert_called_once()

    def test_get_handles_missing_file(self, cache, tmp_path):
        path = tmp_path / "nonexistent.jpg"
        result = cache.get(path)
        assert result is None

    def test_stats_property_returns_copy(self, cache):
        stats = cache.stats
        stats["hits"] = 999
        assert cache.stats["hits"] == 0


class TestPromptCacheDB:
    @pytest.fixture
    def cache(self, tmp_path):
        return PromptCacheDB(tmp_path / "cache.db")

    def test_initializes_table(self, cache):
        cursor = cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_responses'"
        )
        assert cursor.fetchone() is not None

    def test_stats_initial(self, cache):
        assert cache.stats == {"hits": 0, "misses": 0}

    def test_put_and_get_roundtrip(self, cache):
        cache.put("What is 2+2?", "4")
        response = cache.get("What is 2+2?")
        assert response == "4"

    def test_get_returns_none_for_missing(self, cache):
        result = cache.get("Unknown prompt?")
        assert result is None

    def test_stats_track_hits_and_misses(self, cache):
        cache.get("missing")
        assert cache.stats["misses"] == 1
        cache.put("prompt", "response")
        cache.get("prompt")
        assert cache.stats["hits"] == 1

    def test_compute_hash_deterministic(self, cache):
        h1 = cache._compute_hash("test prompt")
        h2 = cache._compute_hash("test prompt")
        assert h1 == h2
        assert len(h1) == 64

    def test_bloom_filter_used(self, cache):
        cache.put("prompt", "response")
        assert "prompt" in cache._hash_set or len(cache._hash_set) > 0

    def test_flush_persists_bloom(self, cache):
        cache.put("prompt", "response")
        cache.flush()
        assert Path(cache.bloom_path).exists()

    def test_close_calls_flush(self, cache):
        cache.flush = MagicMock()
        cache.close()
        cache.flush.assert_called_once()

    def test_stats_property_returns_copy(self, cache):
        stats = cache.stats
        stats["hits"] = 999
        assert cache.stats["hits"] == 0


class TestCacheIntegration:
    def test_frame_cache_persists_across_instances(self, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"content")
        cache1 = FrameCacheDB(tmp_path / "cache.db")
        cache1.put(path, "description")
        cache1.close()
        cache2 = FrameCacheDB(tmp_path / "cache.db")
        result = cache2.get(path)
        assert result == "description"
        cache2.close()

    def test_prompt_cache_persists_across_instances(self, tmp_path):
        cache1 = PromptCacheDB(tmp_path / "cache.db")
        cache1.put("prompt", "response")
        cache1.close()
        cache2 = PromptCacheDB(tmp_path / "cache.db")
        result = cache2.get("prompt")
        assert result == "response"
        cache2.close()
