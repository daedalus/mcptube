"""Tests for vision model frame description."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mcptube.llm import LLMClient, LLMError
from mcptube.ingestion.vision import VisionDescriber
from mcptube.wiki.models import FrameDescription


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.available = True
    return llm


@pytest.fixture
def describer(mock_llm):
    return VisionDescriber(mock_llm)


@pytest.fixture
def fake_frames(tmp_path):
    """Create fake JPEG frame files."""
    frames = []
    for i in range(3):
        path = tmp_path / f"scene_{i+1:04d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        frames.append({
            "path": path,
            "timestamp": float(i * 10),
            "index": i,
        })
    return frames


@pytest.fixture
def many_fake_frames(tmp_path):
    """Create more than 5 fake frames to trigger batch mode."""
    frames = []
    for i in range(8):
        path = tmp_path / f"scene_{i+1:04d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        frames.append({
            "path": path,
            "timestamp": float(i * 5),
            "index": i,
        })
    return frames


class TestInit:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
    def test_detects_anthropic(self, mock_llm):
        d = VisionDescriber(mock_llm)
        assert "anthropic" in d._model or "claude" in d._model

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_detects_openai(self, mock_llm):
        d = VisionDescriber(mock_llm)
        assert "gpt" in d._model

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=False)
    def test_detects_google(self, mock_llm):
        d = VisionDescriber(mock_llm)
        assert "gemini" in d._model


class TestDescribeFrames:
    def test_empty_frames_returns_empty(self, describer):
        result = describer.describe_frames([])
        assert result == []

    def test_unavailable_llm_raises(self, mock_llm):
        mock_llm.available = False
        d = VisionDescriber(mock_llm)
        with pytest.raises(LLMError):
            d.describe_frames([{"path": Path("x.jpg"), "timestamp": 0.0, "index": 0}])


class TestDescribeIndividually:
    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_describes_each_frame(self, mock_completion, describer, fake_frames):
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="A slide showing neural network diagram."))]
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
        descriptions = [f"Frame {i} description" for i in range(8)]
        import json
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
            choices=[MagicMock(message=MagicMock(content=f"```json\n{json.dumps(descriptions)}\n```"))]
        )
        results = describer._describe_batch(many_fake_frames)
        assert len(results) == 8
        assert results[0].description == "Desc 0"

    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_batch_fewer_descriptions_than_frames(self, mock_completion, describer, many_fake_frames):
        import json
        descriptions = ["Only three", "descriptions", "here"]
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(descriptions)))]
        )
        results = describer._describe_batch(many_fake_frames)
        assert len(results) == 8
        assert results[0].description == "Only three"
        assert results[3].description == "(description unavailable)"

    @patch.object(VisionDescriber, "_describe_individually")
    @patch("mcptube.ingestion.vision.litellm.completion")
    def test_batch_falls_back_to_individual(self, mock_completion, mock_individual, describer, many_fake_frames):
        mock_completion.side_effect = Exception("Batch failed")
        mock_individual.return_value = [
            FrameDescription(filename=f["path"].name, timestamp=f["timestamp"], description="Fallback")
            for f in many_fake_frames
        ]
        results = describer._describe_batch(many_fake_frames)
        assert len(results) == 8
        mock_individual.assert_called_once()


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
