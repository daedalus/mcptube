"""Tests for vision model frame description."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mcptube.llm import LLMClient, LLMError
from mcptube.ingestion.vision import VisionDescriber
from mcptube.storage.cache import FrameCacheDB
from mcptube.wiki.models import FrameDescription


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.available = True
    return llm


@pytest.fixture
def describer(mock_llm):
    return VisionDescriber(mock_llm, model="gpt-4o")


@pytest.fixture
def fake_frames(tmp_path):
    """Create fake JPEG frame files."""
    frames = []
    for i in range(3):
        path = tmp_path / f"scene_{i + 1:04d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        frames.append(
            {
                "path": path,
                "timestamp": float(i * 10),
                "index": i,
            }
        )
    return frames


@pytest.fixture
def many_fake_frames(tmp_path):
    """Create more than 5 fake frames to trigger batch mode."""
    frames = []
    for i in range(8):
        path = tmp_path / f"scene_{i + 1:04d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        frames.append(
            {
                "path": path,
                "timestamp": float(i * 5),
                "index": i,
            }
        )
    return frames


class TestInit:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
    def test_detects_anthropic(self, mock_llm):
        d = VisionDescriber(mock_llm)
        assert "anthropic" in d._model or "claude" in d._model

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    def test_detects_openai(self, mock_llm):
        d = VisionDescriber(mock_llm)
        assert "gpt" in d._model

    @patch.dict(
        "os.environ",
        {"GOOGLE_API_KEY": "test-key", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""},
        clear=False,
    )
    def test_detects_google(self, mock_llm):
        d = VisionDescriber(mock_llm)
        assert "gemini" in d._model


class TestDescribeFrames:
    def test_empty_frames_returns_empty(self, describer):
        result = describer.describe_frames([])
        assert result == []

    def test_unavailable_llm_raises(self, mock_llm):
        mock_llm.available = False
        d = VisionDescriber(mock_llm, model="gpt-4o")
        with pytest.raises(LLMError):
            d.describe_frames([{"path": Path("x.jpg"), "timestamp": 0.0, "index": 0}])


class TestDescribeIndividually:
    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_describes_each_frame(self, mock_completion, describer, fake_frames):
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(message=MagicMock(content="A slide showing neural network diagram."))
            ]
        )
        results = describer._describe_individually(fake_frames)

        assert len(results) == 3
        assert all(isinstance(r, FrameDescription) for r in results)
        assert results[0].filename == "scene_0001.jpg"
        assert results[0].timestamp == 0.0
        assert "neural network" in results[0].description.lower()

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_handles_single_frame_failure(self, mock_completion, describer, fake_frames):
        mock_completion.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="Slide 1"))]),
            Exception("API error"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Slide 3"))]),
        ]
        results = describer._describe_individually(fake_frames)

        assert len(results) == 3
        assert results[0].description == "Slide 1"
        assert results[1].description == "(description unavailable)"
        assert results[2].description == "Slide 3"


class TestDescribeSingleFrame:
    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_sends_base64_image(self, mock_completion, describer, fake_frames):
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Description of frame."))]
        )
        desc = describer._describe_single_frame(fake_frames[0]["path"])

        assert desc == "Description of frame."
        call_args = mock_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_raises_on_api_error(self, mock_completion, describer, fake_frames):
        mock_completion.side_effect = Exception("Connection refused")
        with pytest.raises(LLMError, match="Vision model failed"):
            describer._describe_single_frame(fake_frames[0]["path"])


class TestDescribeBatch:
    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_batch_describes_all(self, mock_completion, describer, many_fake_frames):
        import json

        descriptions = [f"Frame {i} description" for i in range(8)]
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(descriptions)))]
        )
        results = describer._describe_batch(many_fake_frames)

        assert len(results) == 8
        assert all(isinstance(r, FrameDescription) for r in results)
        assert results[0].description == "Frame 0 description"
        assert results[7].description == "Frame 7 description"

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_batch_handles_markdown_fences(self, mock_completion, describer, many_fake_frames):
        import json

        descriptions = [f"Desc {i}" for i in range(8)]
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(message=MagicMock(content=f"```json\n{json.dumps(descriptions)}\n```"))
            ]
        )
        results = describer._describe_batch(many_fake_frames)
        assert len(results) == 8
        assert results[0].description == "Desc 0"

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_batch_fewer_descriptions_than_frames(
        self, mock_completion, describer, many_fake_frames
    ):
        import json

        descriptions = ["Only three", "descriptions", "here"]
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(descriptions)))]
        )
        results = describer._describe_batch(many_fake_frames)
        assert len(results) == 8
        assert results[0].description == "Only three"
        assert results[3].description == "(description unavailable)"

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_batch_fallback_handles_error_gracefully(
        self, mock_completion, describer, many_fake_frames
    ):
        mock_completion.side_effect = Exception("Batch failed")
        results = describer._describe_batch(many_fake_frames)
        assert len(results) == 8
        assert all(isinstance(r, FrameDescription) for r in results)


class TestRouting:
    @patch.object(VisionDescriber, "_describe_individually")
    def test_small_batch_uses_individual(self, mock_individual, describer, fake_frames):
        mock_individual.return_value = [
            FrameDescription(filename=f["path"].name, timestamp=f["timestamp"], description="Desc")
            for f in fake_frames
        ]
        describer.describe_frames(fake_frames)
        mock_individual.assert_called_once()

    @patch.object(VisionDescriber, "_describe_batch")
    def test_large_batch_uses_batch(self, mock_batch, describer, many_fake_frames):
        mock_batch.return_value = [
            FrameDescription(filename=f["path"].name, timestamp=f["timestamp"], description="Desc")
            for f in many_fake_frames
        ]
        describer.describe_frames(many_fake_frames)
        mock_batch.assert_called_once()


class TestFrameCacheDB:
    def test_cache_initializes_table(self, tmp_path):
        db = FrameCacheDB(tmp_path / "cache.db")
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='frame_descriptions'"
        )
        assert cursor.fetchone() is not None
        db.close()

    def test_compute_hash_deterministic(self, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg content")
        db = FrameCacheDB(tmp_path / "cache.db")
        h1 = db.compute_hash(path)
        h2 = db.compute_hash(path)
        assert h1 == h2
        assert len(h1) == 64
        db.close()

    def test_put_and_get_roundtrip(self, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg content")
        db = FrameCacheDB(tmp_path / "cache.db")
        db.put(path, "A slide showing neural network")
        desc = db.get(path)
        assert desc == "A slide showing neural network"
        db.close()

    def test_get_returns_none_for_missing(self, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg content")
        db = FrameCacheDB(tmp_path / "cache.db")
        desc = db.get(path)
        assert desc is None
        db.close()

    def test_different_images_different_hashes(self, tmp_path):
        path1 = tmp_path / "frame1.jpg"
        path1.write_bytes(b"image 1")
        path2 = tmp_path / "frame2.jpg"
        path2.write_bytes(b"image 2")
        db = FrameCacheDB(tmp_path / "cache.db")
        assert db.compute_hash(path1) != db.compute_hash(path2)
        db.close()


class TestVisionDescriberWithCache:
    @pytest.fixture
    def cache(self, tmp_path):
        return FrameCacheDB(tmp_path / "cache.db")

    @pytest.fixture
    def describer_with_cache(self, mock_llm, cache):
        return VisionDescriber(mock_llm, cache, model="gpt-4o")

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_describe_single_frame_uses_cache(self, mock_completion, mock_llm, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg")
        describer = VisionDescriber(mock_llm, cache, model="gpt-4o")

        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="First call"))]
        )

        result = describer._describe_single_frame(path)
        assert result == "First call"
        assert mock_completion.call_count == 1

        mock_completion.reset_mock()
        result = describer._describe_single_frame(path)
        assert result == "First call"
        assert mock_completion.call_count == 0

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_cache_hit_avoids_api_call(self, mock_completion, mock_llm, cache, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"fake jpeg")
        db = FrameCacheDB(tmp_path / "cache.db")
        db.put(path, "Cached description")

        describer = VisionDescriber(mock_llm, db, model="gpt-4o")

        result = describer._describe_single_frame(path)
        assert result == "Cached description"
        mock_completion.assert_not_called()

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_batch_cache_miss_queries_llm(self, mock_completion, describer_with_cache, fake_frames):
        import json

        descriptions = [f"Frame {i}" for i in range(3)]
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(descriptions)))]
        )

        results = describer_with_cache._describe_batch(fake_frames)
        assert len(results) == 3
        mock_completion.assert_called_once()

    def test_batch_returns_cached_descriptions_when_all_cached(
        self, describer_with_cache, tmp_path
    ):
        """When all frames are cached, no LLM call is made."""
        paths = []
        for i in range(3):
            p = tmp_path / f"frame_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]) * 100)
            paths.append({"path": p, "timestamp": i * 10.0, "index": i})
            describer_with_cache._cache.put(p, f"Cached {i}")

        results = describer_with_cache._describe_batch(paths)

        assert len(results) == 3
        assert results[0].description.startswith("Cached")
        assert results[1].description.startswith("Cached")
        assert results[2].description.startswith("Cached")
