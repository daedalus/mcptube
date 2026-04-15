# tests/test_cli_integration.py
"""CLI integration tests using Typer's CliRunner."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mcptube.cli import app
from mcptube.models import Video
from mcptube.storage.sqlite import SQLiteVideoRepository
from mcptube.storage.vectorstore import ChromaVectorStore

from mcptube.wiki.engine import WikiEngine
from mcptube.wiki.storage import FileWikiRepository
import tempfile, pathlib


runner = CliRunner()


@pytest.fixture
def mock_service(sample_video):
    """Patch _get_service to return a service with in-memory backends."""
    repo = SQLiteVideoRepository(":memory:")
    # store = ChromaVectorStore(":memory:")

    from mcptube.ingestion.youtube import YouTubeExtractor

    extractor = YouTubeExtractor()

    with patch.object(extractor, "extract", return_value=sample_video):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from mcptube.llm import LLMClient

            with patch("mcptube.llm.litellm.completion") as mock_comp:
                SAMPLE_WIKI = '{"video_summary": "A guide to ML.", "key_timestamps": {"00:00": "Intro"}, "entities": [], "topics": [{"name": "Neural Networks", "content": "Intro to NNs.", "timestamps": ["00:00"], "tags": ["AI"]}], "concepts": []}'
                SAMPLE_CLASSIFY = '["AI", "Tutorial"]'
                responses = [SAMPLE_CLASSIFY, SAMPLE_WIKI]
                call_count = {"i": 0}

                def pick_response(*args, **kwargs):
                    idx = min(call_count["i"], len(responses) - 1)
                    call_count["i"] += 1
                    resp = MagicMock()
                    resp.choices = [MagicMock()]
                    resp.choices[0].message.content = responses[idx]
                    return resp

                mock_comp.side_effect = pick_response

                from mcptube.service import McpTubeService

                wiki_dir = pathlib.Path(tempfile.mkdtemp()) / "wiki"
                wiki_repo = FileWikiRepository(wiki_dir=wiki_dir, db_path=":memory:")
                wiki_engine = WikiEngine(repo=wiki_repo, llm=LLMClient())
                svc = McpTubeService(
                    repository=repo,
                    extractor=extractor,
                    llm_client=LLMClient(),
                    wiki_engine=wiki_engine,
                )
                with patch("mcptube.cli._get_service", return_value=svc):
                    yield svc


class TestCLI:
    def test_add_and_list(self, mock_service):
        result = runner.invoke(app, ["add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        assert result.exit_code == 0
        assert "Added" in result.stdout

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "dQw4w9WgXcQ" in result.stdout

    def test_add_duplicate_error(self, mock_service):
        runner.invoke(app, ["add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        result = runner.invoke(app, ["add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        assert result.exit_code == 1

    def test_info_by_id(self, mock_service):
        runner.invoke(app, ["add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        result = runner.invoke(app, ["info", "dQw4w9WgXcQ"])
        assert result.exit_code == 0
        assert "Intro to Machine Learning" in result.stdout

    def test_info_by_index(self, mock_service):
        runner.invoke(app, ["add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        result = runner.invoke(app, ["info", "1"])
        assert result.exit_code == 0
        assert "Intro to Machine Learning" in result.stdout

    def test_remove_and_list(self, mock_service):
        runner.invoke(app, ["add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        result = runner.invoke(app, ["remove", "dQw4w9WgXcQ"])
        assert result.exit_code == 0
        assert "Removed" in result.stdout

        result = runner.invoke(app, ["list"])
        assert "empty" in result.stdout.lower()

    def test_search_returns_results(self, mock_service):
        runner.invoke(app, ["add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        result = runner.invoke(app, ["search", "neural networks"])
        assert result.exit_code == 0
        assert "neural" in result.stdout.lower()

    def test_add_with_frame_stats(self, mock_service):
        result = runner.invoke(
            app, ["--show-frame-stats", "add", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        )
        assert result.exit_code == 0
        assert "Frames:" in result.stdout
        assert "ffmpeg:" in result.stdout
        assert "LLM:" in result.stdout

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 2
        assert "mcptube" in result.stdout.lower()
