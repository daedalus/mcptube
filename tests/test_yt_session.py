"""Tests for the shared yt-dlp session helpers (network opts + retry)."""

from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

from mcptube.config import settings
from mcptube.ingestion import yt_session


class TestBuildYdlOpts:
    def test_sets_quiet_defaults(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts()
        assert opts["quiet"] is True
        assert opts["no_warnings"] is True
        assert opts["skip_download"] is True
        assert opts["remote_components"] == ["ejs:github"]

    def test_remote_components_respects_base_override(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts({"remote_components": ["ejs:npm"]})
        assert opts["remote_components"] == ["ejs:npm"]

    def test_injects_cookiefile(self, monkeypatch, tmp_path):
        ck = tmp_path / "c.txt"
        monkeypatch.setattr(settings, "cookies_file", ck)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts({})
        assert opts["cookiefile"] == str(ck)

    def test_injects_js_runtimes(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", "node")
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts({})
        assert opts["js_runtimes"] == {"node": {}}

    def test_injects_proxy(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", "http://proxy:8080")
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts({})
        assert opts["proxy"] == "http://proxy:8080"

    def test_no_proxy_clears_proxy(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", True)
        monkeypatch.setattr(settings, "proxy", "http://proxy:8080")
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts({})
        assert opts["proxy"] == ""

    def test_injects_cookies_from_browser(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", "chrome")
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts({})
        assert opts["cookies_from_browser"] == ("chrome", {})

    def test_injects_format(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", "best")
        opts = yt_session.build_ydl_opts({})
        assert opts["format"] == "best"

    def test_does_not_mutate_input(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", "node")
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        base = {"quiet": True}
        yt_session.build_ydl_opts(base)
        assert "js_runtimes" not in base

    def test_merges_with_base_opts(self, monkeypatch):
        monkeypatch.setattr(settings, "cookies_file", None)
        monkeypatch.setattr(settings, "js_runtimes", None)
        monkeypatch.setattr(settings, "no_proxy", False)
        monkeypatch.setattr(settings, "proxy", None)
        monkeypatch.setattr(settings, "cookies_from_browser", None)
        monkeypatch.setattr(settings, "format", None)
        opts = yt_session.build_ydl_opts({"writesubtitles": True, "quiet": False})
        assert opts["writesubtitles"] is True
        assert opts["quiet"] is False  # setdefault respects existing values


class TestExtractInfoWithRetry:
    def _make_ydl_that_fails_n_times(self, mock_cls, n_failures, info):
        ydl = MagicMock()
        effects = [yt_dlp.utils.DownloadError("bot check")] * n_failures
        effects.append(info)
        ydl.extract_info.side_effect = effects
        mock_cls.return_value.__enter__ = lambda s: ydl
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        return ydl

    @patch("mcptube.ingestion.yt_session.time.sleep")
    @patch("mcptube.ingestion.yt_session.yt_dlp.YoutubeDL")
    def test_retries_transient_failure_then_succeeds(self, mock_cls, _sleep):
        ydl = self._make_ydl_that_fails_n_times(mock_cls, 2, {"id": "x"})
        info = yt_session.extract_info_with_retry("https://u", {}, attempts=3)
        assert info == {"id": "x"}
        assert ydl.extract_info.call_count == 3

    @patch("mcptube.ingestion.yt_session.time.sleep")
    @patch("mcptube.ingestion.yt_session.yt_dlp.YoutubeDL")
    def test_raises_after_exhausting_attempts(self, mock_cls, _sleep):
        self._make_ydl_that_fails_n_times(mock_cls, 5, {"id": "x"})
        with pytest.raises(yt_dlp.utils.DownloadError):
            yt_session.extract_info_with_retry("https://u", {}, attempts=3)

    @patch("mcptube.ingestion.yt_session.yt_dlp.YoutubeDL")
    def test_success_first_try_no_retry(self, mock_cls):
        ydl = self._make_ydl_that_fails_n_times(mock_cls, 0, {"id": "ok"})
        info = yt_session.extract_info_with_retry("https://u", {})
        assert info == {"id": "ok"}
        assert ydl.extract_info.call_count == 1
