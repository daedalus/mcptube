"""Scene-change frame extraction from videos via ffmpeg."""

import logging
import subprocess
import time
from pathlib import Path

import yt_dlp

from mcptube.config import settings
from mcptube.ingestion.yt_session import build_ydl_opts, extract_info_with_retry

logger = logging.getLogger(__name__)

# See frames.py: resolution opts for direct stream URLs. The mweb/android
# player clients are required so ffmpeg gets fetchable (non-SABR) URLs.
_STREAM_RESOLVE_OPTS = {
    "format": "bestvideo[ext=mp4][height<=720]/bestvideo[ext=mp4]/best[ext=mp4]/best",
    "extractor_args": {"youtube": {"player_client": ["mweb", "android"]}},
}

_FFMPEG_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

# Freshly resolved googlevideo URLs reliably return 403 for the first
# few seconds (PO-token propagation on the CDN), then succeed.
_RETRIES_403 = 3
_RETRY_403_DELAY_S = 5


class SceneFrameError(Exception):
    """Raised when scene-change frame extraction fails."""


class SceneFrameExtractor:
    """Extracts key frames from videos using ffmpeg scene-change detection.

    Uses ffmpeg's scene filter to detect visual transitions and extract
    only frames where significant visual change occurs. This is ideal
    for lectures, slides, demos, and presentations where the screen
    content changes at meaningful moments.

    The scene threshold (0.0-1.0) controls sensitivity:
    - Lower = more frames (catches subtle changes)
    - Higher = fewer frames (only major transitions)
    - Default 0.4 is a good balance for most content
    """

    _DEFAULT_THRESHOLD = 0.4
    _MAX_FRAMES = 50  # safety cap
    _SCALE_WIDTH = 1280
    # Fallback thresholds tried (in order) when the configured threshold
    # matches nothing. Lecture/slide videos often have scene scores far
    # below the 0.4 default (measured max ~0.013), which would otherwise
    # yield zero frames.
    _FALLBACK_THRESHOLDS = (0.05, 0.005)

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize scene frame extractor.

        Args:
            threshold: Scene-change sensitivity (0.0-1.0). Default 0.4.
        """
        self._threshold = threshold or self._DEFAULT_THRESHOLD

    def extract_scene_frames(
        self,
        video_id: str,
        max_frames: int | None = None,
        source_url: str = "",
    ) -> list[dict]:
        """Extract key frames at scene-change points from a video.

        Args:
            video_id: Namespaced video ID (used for the output directory).
            max_frames: Maximum frames to extract. Defaults to _MAX_FRAMES.
            source_url: Canonical URL of the video. If empty, builds
                        a YouTube URL from video_id.

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
        stream_url = self._resolve_stream_url(source_url or video_id)

        # Extract frames via ffmpeg scene filter
        frames = self._extract_with_ffmpeg(stream_url, output_dir, max_frames)

        logger.info("Extracted %d scene-change frames for %s", len(frames), video_id)
        return frames

    def _resolve_stream_url(self, source: str) -> str:
        """Resolve a direct stream URL from a video URL or ID.

        Args:
            source: A full video URL, or a YouTube video ID (legacy).
        """
        if source.startswith("http"):
            url = source
        else:
            url = f"https://www.youtube.com/watch?v={source}"

        ydl_opts = build_ydl_opts(_STREAM_RESOLVE_OPTS)

        try:
            info = extract_info_with_retry(url, ydl_opts)
            if info is None:
                raise SceneFrameError(f"yt-dlp returned no info for: {source}")
            stream_url = info.get("url")
            if not stream_url:
                raise SceneFrameError(f"No stream URL resolved for: {source}")
            return stream_url
        except yt_dlp.utils.DownloadError as e:
            raise SceneFrameError(f"Failed to resolve stream URL: {e}") from e

    def _extract_with_ffmpeg(
        self, stream_url: str, output_dir: Path, max_frames: int
    ) -> list[dict]:
        """Use ffmpeg scene filter to extract key frames.

        Tries the configured threshold first, then progressively lower
        fallback thresholds while zero frames are produced.
        """
        output_pattern = str(output_dir / "scene_%04d.jpg")

        result: subprocess.CompletedProcess[str] | None = None
        ladder = self._threshold_ladder()
        for attempt, threshold in enumerate(ladder):
            if attempt > 0:
                logger.info(
                    "No scene changes at threshold %.3f, retrying with %.3f",
                    ladder[attempt - 1],
                    threshold,
                )

            cmd = self._build_scene_cmd(
                threshold, stream_url, output_pattern, max_frames
            )
            try:
                result = self._run_ffmpeg(cmd, timeout=300)
            except subprocess.TimeoutExpired:
                raise SceneFrameError("ffmpeg timed out during scene detection")
            except FileNotFoundError:
                raise SceneFrameError(
                    "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
                )

            if any(output_dir.glob("scene_*.jpg")):
                break
            if result.returncode != 0:
                # ffmpeg may return non-zero but still produce frames —
                # checked above — so this is a real failure.
                raise SceneFrameError(
                    f"ffmpeg failed (code {result.returncode}): {result.stderr[:300]}"
                )
            # rc == 0 but no frames: try the next (lower) threshold.

        assert result is not None

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

        if not frames:
            logger.warning(
                "No scene changes found even at threshold %.3f",
                ladder[-1],
            )

        # Save timestamp metadata for cache
        self._save_metadata(output_dir, frames)

        return frames

    def _build_scene_cmd(
        self, threshold: float, stream_url: str, output_pattern: str, max_frames: int
    ) -> list[str]:
        """Build the ffmpeg scene-detection command.

        format=yuvj420p after scale: without an explicit pix_fmt the
        mjpeg encoder fails to open when no frames pass the select
        filter (ffmpeg exits 234 instead of 0).
        """
        return [
            "ffmpeg",
            "-user_agent",
            _FFMPEG_USER_AGENT,
            "-i",
            stream_url,
            "-vf",
            (
                f"select='gt(scene,{threshold})',"
                f"scale={self._SCALE_WIDTH}:-2,format=yuvj420p,showinfo"
            ),
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            "-y",
            output_pattern,
        ]

    @staticmethod
    def _run_ffmpeg(cmd: list[str], timeout: int) -> "subprocess.CompletedProcess[str]":
        """Run an ffmpeg command, retrying transient 403s on fresh URLs."""
        result: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, _RETRIES_403 + 1):
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0 or "403" not in (result.stderr or ""):
                return result
            if attempt < _RETRIES_403:
                logger.warning(
                    "ffmpeg got 403 on freshly resolved URL "
                    "(attempt %d/%d), retrying in %ds",
                    attempt,
                    _RETRIES_403,
                    _RETRY_403_DELAY_S,
                )
                time.sleep(_RETRY_403_DELAY_S)
        assert result is not None
        return result

    def _threshold_ladder(self) -> list[float]:
        """Configured threshold followed by lower fallbacks, in order."""
        ladder = [self._threshold]
        for fallback in self._FALLBACK_THRESHOLDS:
            if fallback < ladder[-1]:
                ladder.append(fallback)
        return ladder

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
