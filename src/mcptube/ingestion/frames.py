"""Frame extraction from videos via yt-dlp + ffmpeg."""

import logging
import subprocess
from pathlib import Path

import yt_dlp

from mcptube.config import settings
from mcptube.ingestion.yt_session import build_ydl_opts, extract_info_with_retry

logger = logging.getLogger(__name__)


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""


class FrameExtractor:
    """Extracts individual frames from videos.

    Uses yt-dlp to resolve direct stream URLs (ffmpeg cannot read
    page URLs directly), then ffmpeg to seek and extract
    a single frame. Frames are cached on disk to avoid re-extraction.
    """

    def extract_frame(
        self, video_id: str, timestamp: float, source_url: str = ""
    ) -> Path:
        """Extract a single frame at the given timestamp.

        Args:
            video_id: Namespaced video ID (used for cache path).
            timestamp: Time in seconds to extract frame at.
            source_url: Canonical URL of the video (YouTube, TikTok, etc.).
                        If empty, falls back to YouTube URL from video_id.

        Returns:
            Path to the extracted JPEG frame.

        Raises:
            FrameExtractionError: If extraction fails.
        """
        # Check cache first
        cache_path = self._cache_path(video_id, timestamp)
        if cache_path.exists():
            logger.info("Frame cache hit: %s", cache_path)
            return cache_path

        # Resolve direct stream URL via yt-dlp
        stream_url = self._resolve_stream_url(source_url or video_id)

        # Extract frame via ffmpeg
        self._extract_with_ffmpeg(stream_url, timestamp, cache_path)

        return cache_path

    def _resolve_stream_url(self, source: str) -> str:
        """Resolve a direct stream URL from a video URL or ID.

        Args:
            source: A full video URL, or a YouTube video ID (legacy).
        """
        # If it looks like a URL, use it directly; otherwise build a YouTube URL
        if source.startswith("http"):
            url = source
        else:
            url = f"https://www.youtube.com/watch?v={source}"

        ydl_opts = build_ydl_opts(
            {
                "format": "best[ext=mp4]/best",
            }
        )

        try:
            info = extract_info_with_retry(url, ydl_opts)
            if info is None:
                raise FrameExtractionError(f"yt-dlp returned no info for: {source}")
            stream_url = info.get("url")
            if not stream_url:
                raise FrameExtractionError(f"No stream URL resolved for: {source}")
            return stream_url
        except yt_dlp.utils.DownloadError as e:
            raise FrameExtractionError(f"Failed to resolve stream URL: {e}") from e

    def _extract_with_ffmpeg(
        self, stream_url: str, timestamp: float, output: Path
    ) -> None:
        """Use ffmpeg to seek and extract a single JPEG frame."""
        output.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            stream_url,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0 or not output.exists():
                raise FrameExtractionError(
                    f"ffmpeg failed (code {result.returncode}): {result.stderr[:200]}"
                )
            logger.info("Frame extracted: %s", output)
        except subprocess.TimeoutExpired:
            raise FrameExtractionError(
                f"ffmpeg timed out extracting frame at {timestamp}s"
            )
        except FileNotFoundError:
            raise FrameExtractionError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )

    @staticmethod
    def _cache_path(video_id: str, timestamp: float) -> Path:
        """Generate a deterministic cache path for a frame."""
        settings.frames_dir.mkdir(parents=True, exist_ok=True)
        return settings.frames_dir / f"{video_id}_{timestamp:.2f}.jpg"
