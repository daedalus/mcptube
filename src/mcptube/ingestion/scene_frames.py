"""Scene-change frame extraction from YouTube videos via ffmpeg."""

import logging
import subprocess
from pathlib import Path

import yt_dlp

from mcptube.config import settings
from mcptube.ingestion.youtube import _get_cookie_file

logger = logging.getLogger(__name__)


class SceneFrameError(Exception):
    """Raised when scene-change frame extraction fails."""


class SceneFrameExtractor:
    """Extracts key frames from YouTube videos using ffmpeg scene-change detection.

    Uses ffmpeg's scene filter to detect visual transitions and extract
    only frames where significant visual change occurs. This is ideal
    for lectures, slides, demos, and presentations where the screen
    content changes at meaningful moments.

    The scene threshold (0.0–1.0) controls sensitivity:
    - Lower = more frames (catches subtle changes)
    - Higher = fewer frames (only major transitions)
    - Default 0.4 is a good balance for most content
    """

    _DEFAULT_THRESHOLD = 0.4
    _MAX_FRAMES = 50  # safety cap
    _SCALE_WIDTH = 1280

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize scene frame extractor.

        Args:
            threshold: Scene-change sensitivity (0.0–1.0). Default 0.4.
        """
        self._threshold = threshold or self._DEFAULT_THRESHOLD

    def extract_scene_frames(
        self,
        video_id: str,
        max_frames: int | None = None,
    ) -> list[dict]:
        """Extract key frames at scene-change points from a YouTube video.

        Args:
            video_id: YouTube video ID.
            max_frames: Maximum frames to extract. Defaults to _MAX_FRAMES.

        Returns:
            List of dicts with keys: "path" (Path), "timestamp" (float), "index" (int)

        Raises:
            SceneFrameError: If extraction fails.
        """
        max_frames = max_frames or self._MAX_FRAMES

        # Ensure output directory exists
        output_dir = self._output_dir(video_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check cache — if frames already extracted, return them
        cached = self._load_cached(output_dir)
        if cached:
            logger.info(
                "Scene frames cache hit: %d frames for %s", len(cached), video_id
            )
            return cached[:max_frames]

        # Resolve direct stream URL
        stream_url = self._resolve_stream_url(video_id)

        # Extract frames via ffmpeg scene filter
        frames = self._extract_with_ffmpeg(stream_url, output_dir, max_frames)

        logger.info("Extracted %d scene-change frames for %s", len(frames), video_id)
        return frames

    def _resolve_stream_url(self, video_id: str) -> str:
        """Resolve a direct stream URL from a YouTube video ID."""
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best",
            "skip_download": True,
        }
        cookie_file = _get_cookie_file()
        if cookie_file:
            ydl_opts["cookiefile"] = str(cookie_file)
            logger.debug("Using cookies for scene frames: %s", cookie_file)
        else:
            logger.warning("NO COOKIE FILE FOUND for scene frames!")
        if settings.js_runtimes:
            ydl_opts["js_runtimes"] = {settings.js_runtimes: {}}
            logger.debug("Using JS runtime for scene frames: %s", settings.js_runtimes)
        else:
            logger.warning("NO JS_RUNTIMES for scene frames!")
        if settings.no_proxy:
            ydl_opts["proxy"] = ""
            logger.debug("Proxy disabled for scene frames")
        elif settings.proxy:
            ydl_opts["proxy"] = settings.proxy
            logger.debug("Using proxy for scene frames: %s", settings.proxy)
        if settings.cookies_from_browser:
            ydl_opts["cookies_from_browser"] = (settings.cookies_from_browser, {})
            logger.debug(
                "Using cookies from browser for scene frames: %s",
                settings.cookies_from_browser,
            )
        logger.info("scene_frames ydl_opts: %s", ydl_opts)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise SceneFrameError(f"yt-dlp returned no info for: {video_id}")
                stream_url = info.get("url")
                if not stream_url:
                    raise SceneFrameError(f"No stream URL resolved for: {video_id}")
                return stream_url
        except yt_dlp.utils.DownloadError as e:
            raise SceneFrameError(f"Failed to resolve stream URL: {e}") from e

    def _extract_with_ffmpeg(
        self, stream_url: str, output_dir: Path, max_frames: int
    ) -> list[dict]:
        """Use ffmpeg scene filter to extract key frames."""
        output_pattern = str(output_dir / "scene_%04d.jpg")

        # ffmpeg command:
        # -vf "select='gt(scene,T)',scale=W:-1" — detect scene changes, scale down
        # -vsync vfr — variable frame rate (only output selected frames)
        # -frame_pts 1 — write PTS as frame number (for timestamp recovery)
        cmd = [
            "ffmpeg",
            "-i",
            stream_url,
            "-vf",
            f"select='gt(scene,{self._threshold})',scale={self._SCALE_WIDTH}:-1,showinfo",
            "-vsync",
            "vfr",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            "-y",
            output_pattern,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                # ffmpeg may return non-zero but still produce frames
                if not any(output_dir.glob("scene_*.jpg")):
                    raise SceneFrameError(
                        f"ffmpeg failed (code {result.returncode}): {result.stderr[:300]}"
                    )

        except subprocess.TimeoutExpired:
            raise SceneFrameError("ffmpeg timed out during scene detection")
        except FileNotFoundError:
            raise SceneFrameError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )

        # Parse timestamps from ffmpeg showinfo output
        timestamps = self._parse_showinfo_timestamps(result.stderr)

        # Build frame list
        frames = []
        for i, path in enumerate(sorted(output_dir.glob("scene_*.jpg"))):
            timestamp = timestamps[i] if i < len(timestamps) else 0.0
            frames.append(
                {
                    "path": path,
                    "timestamp": timestamp,
                    "index": i,
                }
            )

        # Save timestamp metadata for cache
        self._save_metadata(output_dir, frames)

        return frames

    @staticmethod
    def _parse_showinfo_timestamps(stderr: str) -> list[float]:
        """Parse frame timestamps from ffmpeg showinfo filter output.

        showinfo outputs lines like:
            [Parsed_showinfo_2 ...] n: 0 pts: 12345 pts_time:1.234 ...
        """
        import re

        timestamps = []
        pattern = re.compile(r"pts_time:\s*([\d.]+)")
        for line in stderr.split("\n"):
            if "showinfo" in line:
                match = pattern.search(line)
                if match:
                    timestamps.append(float(match.group(1)))
        return timestamps

    def _load_cached(self, output_dir: Path) -> list[dict] | None:
        """Load cached frames if metadata file exists."""
        import json

        meta_path = output_dir / "metadata.json"
        if not meta_path.exists():
            return None

        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            frames = []
            for entry in data:
                path = output_dir / entry["filename"]
                if path.exists():
                    frames.append(
                        {
                            "path": path,
                            "timestamp": entry["timestamp"],
                            "index": entry["index"],
                        }
                    )
            return frames if frames else None
        except Exception:
            return None

    @staticmethod
    def _save_metadata(output_dir: Path, frames: list[dict]) -> None:
        """Save frame metadata for caching."""
        import json

        meta = [
            {
                "filename": f["path"].name,
                "timestamp": f["timestamp"],
                "index": f["index"],
            }
            for f in frames
        ]
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def _output_dir(video_id: str) -> Path:
        """Get the output directory for scene frames."""
        return settings.frames_dir / f"{video_id}_scenes"
