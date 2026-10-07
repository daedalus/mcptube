# tests/test_frames.py
"""Tests for frame extraction."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcptube.ingestion.frames import FrameExtractionError, FrameExtractor


class TestFrameExtractor:
    @patch("mcptube.ingestion.frames.subprocess.run")
    @patch.object(
        FrameExtractor,
        "_resolve_stream_url",
        return_value="https://stream.example.com/video.mp4",
    )
    def test_extract_frame_calls_ffmpeg(self, mock_resolve, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        extractor = FrameExtractor()
        cache_path = extractor._cache_path("abc123", 10.0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"\xff\xd8fake-jpeg")

        with patch.object(
            FrameExtractor, "_cache_path", return_value=tmp_path / "test.jpg"
        ):
            # File doesn't exist at tmp_path, so ffmpeg should be called
            result_path = tmp_path / "test.jpg"
            mock_run.return_value = MagicMock(returncode=0)

            with patch.object(extractor, "_extract_with_ffmpeg") as mock_ffmpeg:
                mock_ffmpeg.side_effect = lambda url, ts, out: out.write_bytes(
                    b"\xff\xd8fake"
                )
                path = extractor.extract_frame("abc123", 10.0)

            mock_resolve.assert_called_once_with("abc123")

    @patch.object(FrameExtractor, "_cache_path")
    def test_extract_frame_cached(self, mock_cache_path, tmp_path):
        cached = tmp_path / "cached_frame.jpg"
        cached.write_bytes(b"\xff\xd8fake-jpeg")
        mock_cache_path.return_value = cached

        extractor = FrameExtractor()
        path = extractor.extract_frame("abc123", 10.0)
        assert path == cached

    def test_extract_with_ffmpeg_not_found(self, tmp_path):
        extractor = FrameExtractor()
        output = tmp_path / "out.jpg"

        with patch(
            "mcptube.ingestion.frames.subprocess.run", side_effect=FileNotFoundError
        ):
            with pytest.raises(FrameExtractionError, match="ffmpeg not found"):
                extractor._extract_with_ffmpeg("https://stream.url", 10.0, output)

    def test_extract_with_ffmpeg_fails(self, tmp_path):
        extractor = FrameExtractor()
        output = tmp_path / "out.jpg"

        mock_result = MagicMock(returncode=1, stderr="Some error")
        with patch("mcptube.ingestion.frames.subprocess.run", return_value=mock_result):
            with pytest.raises(FrameExtractionError, match="ffmpeg failed"):
                extractor._extract_with_ffmpeg("https://stream.url", 10.0, output)

    def test_extract_with_ffmpeg_timeout(self, tmp_path):
        extractor = FrameExtractor()
        output = tmp_path / "out.jpg"

        with patch(
            "mcptube.ingestion.frames.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30),
        ):
            with pytest.raises(FrameExtractionError, match="timed out"):
                extractor._extract_with_ffmpeg("https://stream.url", 10.0, output)

    @patch("mcptube.ingestion.frames.time.sleep")
    def test_403_on_fresh_url_is_retried(self, mock_sleep, tmp_path):
        """Freshly resolved URLs 403 briefly; ffmpeg is retried."""
        extractor = FrameExtractor()
        output = tmp_path / "out.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                return MagicMock(returncode=8, stderr="HTTP error 403 Forbidden")
            output.write_bytes(b"\xff\xd8fake")
            return MagicMock(returncode=0, stderr="")

        with patch("mcptube.ingestion.frames.subprocess.run", side_effect=fake_run):
            extractor._extract_with_ffmpeg("https://stream.url", 10.0, output)

        assert calls["n"] == 2
        mock_sleep.assert_called_once()

    @patch("mcptube.ingestion.frames.time.sleep")
    def test_non_403_failure_is_not_retried(self, mock_sleep, tmp_path):
        extractor = FrameExtractor()
        output = tmp_path / "out.jpg"

        with patch(
            "mcptube.ingestion.frames.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="Some error"),
        ) as mock_run:
            with pytest.raises(FrameExtractionError, match="ffmpeg failed"):
                extractor._extract_with_ffmpeg("https://stream.url", 10.0, output)

        assert mock_run.call_count == 1
        mock_sleep.assert_not_called()

    def test_cache_path_deterministic(self):
        p1 = FrameExtractor._cache_path("abc123", 10.0)
        p2 = FrameExtractor._cache_path("abc123", 10.0)
        assert p1 == p2

    def test_cache_path_different_for_different_input(self):
        p1 = FrameExtractor._cache_path("abc123", 10.0)
        p2 = FrameExtractor._cache_path("abc123", 20.0)
        assert p1 != p2

    @patch("mcptube.ingestion.frames.yt_dlp.YoutubeDL")
    def test_resolve_stream_url(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "url": "https://stream.example.com/video.mp4"
        }
        mock_ydl_class.return_value.__enter__ = lambda s: mock_ydl
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

        extractor = FrameExtractor()
        url = extractor._resolve_stream_url("abc123")
        assert url == "https://stream.example.com/video.mp4"

    def test_ffmpeg_cmd_passes_user_agent(self, tmp_path):
        extractor = FrameExtractor()
        output = tmp_path / "out.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)

        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            output.write_bytes(b"\xff\xd8fake")
            return MagicMock(returncode=0, stderr="")

        with patch("mcptube.ingestion.frames.subprocess.run", side_effect=fake_run):
            extractor._extract_with_ffmpeg("https://stream.url", 10.0, output)

        assert "-user_agent" in captured_cmd

    @patch("mcptube.ingestion.frames.build_ydl_opts")
    @patch("mcptube.ingestion.frames.yt_dlp.YoutubeDL")
    def test_resolve_opts_prefer_mweb_android_clients(
        self, mock_ydl_class, mock_build_opts
    ):
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {"url": "https://s/v.mp4"}
        mock_ydl_class.return_value.__enter__ = lambda s: mock_ydl
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_build_opts.side_effect = lambda base: base

        extractor = FrameExtractor()
        extractor._resolve_stream_url("https://youtube.com/watch?v=abc")
        base = mock_build_opts.call_args[0][0]
        assert base["extractor_args"]["youtube"]["player_client"] == ["mweb", "android"]
        assert "bestvideo[ext=mp4]" in base["format"]
