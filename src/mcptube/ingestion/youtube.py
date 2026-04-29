"""YouTube video ingestion via yt-dlp."""

import json
import logging
import random
import time
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen, Request

import yt_dlp

from mcptube.llm import get_llm

from mcptube.models import Chapter, TranscriptSegment, Video
from mcptube.storage.cache import SubtitleCacheDB

logger = logging.getLogger(__name__)


def _get_cookie_file() -> Path | None:
    """Get the cookie file path from settings, mcptube data directory, or current dir."""
    from mcptube.config import settings

    if settings.cookies_file:
        return settings.cookies_file
    try:
        cookie_path = settings.data_dir / ".cookies.txt"
        if cookie_path.exists():
            return cookie_path
    except Exception:
        pass
    # Fallback to current directory
    fallback = Path(".cookies.txt")
    return fallback if fallback.exists() else None


def _translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English using LLM if not already in English."""
    if source_lang.startswith('en'):
        return text
    
    try:
        llm = get_llm()
        prompt = f"""Translate the following text from {source_lang} to English. 
        Only return the translated text, no explanations or additional formatting:

        {text}"""
        
        response = llm.complete(prompt)
        return str(response).strip()
    except Exception as e:
        logger.warning("Translation failed for %s: %s", source_lang, e)
        return text  # Return original if translation fails


class ExtractionError(Exception):
    """Raised when video extraction fails."""


class YouTubeExtractor:
    """Extracts metadata and transcripts from YouTube videos via yt-dlp.

    Single responsibility: given a YouTube URL, return a populated Video model.
    All yt-dlp interaction is encapsulated here.
    """

    _URL_PATTERNS = [
        re.compile(r"(?:youtube\.com/watch\?.*v=)([\w-]{11})"),
        re.compile(r"(?:youtu\.be/)([\w-]{11})"),
        re.compile(r"(?:youtube\.com/embed/)([\w-]{11})"),
        re.compile(r"(?:youtube\.com/v/)([\w-]{11})"),
    ]

    _LANG_PREFERENCE = ("en", "en-orig", "en-US", "en-GB")
    _FALLBACK_LANGS = ("es", "es-ES", "es-MX", "fr", "fr-FR", "de", "de-DE", "it", "it-IT", "pt", "pt-BR", "ru", "ja", "ko", "zh", "zh-CN", "zh-TW")

    def __init__(self, subtitle_cache: SubtitleCacheDB | None = None) -> None:
        self._subtitle_cache = subtitle_cache

    def extract(self, url: str) -> Video:
        """Extract metadata and transcript from a YouTube video URL.

        Args:
            url: YouTube video URL in any standard format.

        Returns:
            Populated Video model.

        Raises:
            ExtractionError: If extraction fails.
        """
        video_id = self.parse_video_id(url)
        info = self._fetch_info(url)
        transcript = self._get_cached_transcript(video_id) or self._extract_transcript(info)
        if transcript:
            self._cache_transcript(video_id, transcript)
        chapters = self._extract_chapters(info)
        video_stats = self._extract_video_stats(info)

        return Video(
            video_id=video_id,
            title=info.get("title", ""),
            description=info.get("description", ""),
            channel=info.get("channel", "") or info.get("uploader", ""),
            duration=float(info.get("duration", 0) or 0),
            thumbnail_url=info.get("thumbnail", ""),
            chapters=chapters,
            transcript=transcript,
            format=video_stats.get("format", ""),
            file_size=video_stats.get("file_size", 0),
            width=video_stats.get("width", 0),
            height=video_stats.get("height", 0),
            vcodec=video_stats.get("vcodec", ""),
            acodec=video_stats.get("acodec", ""),
        )

    @classmethod
    def parse_video_id(cls, url: str) -> str:
        """Extract the 11-character video ID from a YouTube URL.

        Supports youtube.com/watch, youtu.be, /embed/, and /v/ formats.

        Raises:
            ExtractionError: If the URL cannot be parsed.
        """
        for pattern in cls._URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)

        # Fallback: query parameter parsing
        parsed = urlparse(url)
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id and len(video_id) == 11:
            return video_id

        raise ExtractionError(f"Could not extract video ID from URL: {url}")

    def _fetch_info(self, url: str) -> dict:
        """Fetch video info dict from yt-dlp without downloading media."""
        from mcptube.config import settings

        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": list(self._LANG_PREFERENCE),
            "subtitlesformat": "json3",
            "skip_download": True,
        }
        cookie_file = _get_cookie_file()
        if cookie_file:
            ydl_opts["cookiefile"] = str(cookie_file)
            logger.info("Using cookies from: %s", cookie_file)
        if settings.js_runtimes:
            ydl_opts["js_runtimes"] = {settings.js_runtimes: {}}
            logger.info("Using JS runtime: %s", settings.js_runtimes)
        if settings.no_proxy:
            ydl_opts["proxy"] = ""
            logger.info("Proxy disabled for yt-dlp")
        elif settings.proxy:
            ydl_opts["proxy"] = settings.proxy
            logger.info("Using proxy: %s", settings.proxy)
        if settings.cookies_from_browser:
            ydl_opts["cookies_from_browser"] = (settings.cookies_from_browser, {})
            logger.info("Using cookies from browser: %s", settings.cookies_from_browser)
        if settings.format:
            ydl_opts["format"] = settings.format
            logger.info("Using video format: %s", settings.format)

        logger.debug("yt-dlp options: %s", ydl_opts)
        try:
            logger.debug("Starting yt-dlp extraction for: %s", url)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                logger.debug("yt-dlp extracted info: %s", info.get("id") if info else None)
                if info is None:
                    raise ExtractionError(f"yt-dlp returned no info for: {url}")
                return info
        except Exception as e:
            if "Sign in to confirm" in str(e) or "bot" in str(e).lower():
                raise ExtractionError(
                    f"Failed to extract video info: {e}\n\n"
                    "YouTube is blocking the request. Try:\n"
                    "  1. Use --cookies-from-browser chrome to export fresh cookies\n"
                    "  2. Or use a browser extension like 'Get cookies.txt LOCALLY'\n"
                    "  3. Ensure cookies are not expired"
                ) from e
            raise ExtractionError(f"Failed to extract video info: {e}") from e

    def _extract_video_stats(self, info: dict) -> dict:
        """Extract video format and size info from yt-dlp info dict."""
        result = {
            "format": "",
            "file_size": 0,
            "width": 0,
            "height": 0,
            "vcodec": "",
            "acodec": "",
        }
        # Try to get format info from requested_formats or formats list
        formats = info.get("formats") or []
        # Get the best quality format (usually last with video)
        best = formats[-1] if formats else {}

        if best:
            # Resolution
            resolution = best.get("resolution", "")
            if resolution and "x" in resolution:
                w, h = resolution.split("x")
                result["width"] = int(w) if w.isdigit() else 0
                result["height"] = int(h) if h.isdigit() else 0

                # Convert to user-friendly format label
                h_int = result["height"]
                if h_int >= 2160:
                    result["format"] = "4K"
                elif h_int >= 1440:
                    result["format"] = "2K"
                elif h_int >= 1080:
                    result["format"] = "1080p"
                elif h_int >= 720:
                    result["format"] = "720p"
                elif h_int >= 480:
                    result["format"] = "480p"
                else:
                    result["format"] = f"{h_int}p"

            # File size
            result["file_size"] = int(best.get("filesize", 0) or 0)
            if result["file_size"] == 0:
                result["file_size"] = int(best.get("filesize_approx", 0) or 0)

            # Codecs
            vcodec = best.get("vcodec", "")
            if vcodec and vcodec != "none":
                # Extract short codec name (e.g., "avc1.64001F" -> "avc1")
                result["vcodec"] = vcodec.split(".")[0].replace("vp9", "vp9").replace("av01", "av1")
            acodec = best.get("acodec", "")
            if acodec and acodec != "none":
                result["acodec"] = (
                    acodec.split(".")[0].replace("mp4a", "aac").replace("opus", "opus")
                )

        return result

    def _extract_transcript(self, info: dict) -> list[TranscriptSegment]:
        """Extract transcript segments, trying multiple languages and fallback options with translation."""
        subtitles = info.get("subtitles") or {}
        auto_captions = info.get("automatic_captions") or {}

        # Try manual subtitles first with language fallback
        sub_data, sub_lang = self._find_json3_with_lang(subtitles)
        if not sub_data:
            # Try automatic captions with language fallback
            sub_data, sub_lang = self._find_json3_with_lang(auto_captions)
         
        if not sub_data:
            logger.warning("No transcript available in any language for: %s", info.get("id"))
            return []

        return self._parse_json3(sub_data, sub_lang)

    def _get_cached_transcript(self, video_id: str) -> list[TranscriptSegment] | None:
        """Get cached transcript for video ID."""
        try:
            if self._subtitle_cache is None:
                self._subtitle_cache = SubtitleCacheDB()
            cached = self._subtitle_cache.get(video_id)
            if cached:
                return [TranscriptSegment(**s) for s in cached]
        except Exception:
            pass
        return None

    def _cache_transcript(self, video_id: str, transcript: list[TranscriptSegment]) -> None:
        """Cache transcript for video ID."""
        try:
            if self._subtitle_cache is None:
                self._subtitle_cache = SubtitleCacheDB()
            self._subtitle_cache.put(video_id, transcript)
        except Exception:
            pass

    def _find_json3_with_lang(self, subs: dict) -> tuple[dict | None, str]:
        """Find and download json3 subtitle data, returning both data and language code.
        Tries English first, then other languages with translation fallback.
        """
        # Try preferred language codes first (English variants)
        for lang in self._LANG_PREFERENCE:
            data = self._get_json3_for_lang(subs, lang)
            if data:
                return data, lang

        # Fallback: any en-* variant
        for lang in subs:
            if lang.startswith("en"):
                data = self._get_json3_for_lang(subs, lang)
                if data:
                    return data, lang

        # Try other languages if English not available
        for lang in self._FALLBACK_LANGS:
            data = self._get_json3_for_lang(subs, lang)
            if data:
                logger.info("Using %s subtitles as fallback", lang)
                return data, lang

        # Fallback: any other language variant
        for lang in subs:
            if not lang.startswith("en") and lang not in self._LANG_PREFERENCE:
                data = self._get_json3_for_lang(subs, lang)
                if data:
                    logger.info("Using %s subtitles as fallback", lang)
                    return data, lang

        return None, "en"  # Default to English if nothing found

    def _get_json3_for_lang(self, subs: dict, lang: str) -> dict | None:
        """Download json3 data for a specific language code if available."""
        formats = subs.get(lang)
        if not formats:
            return None
        for fmt in formats:
            if fmt.get("ext") == "json3":
                return self._download_json(fmt["url"])
        return None

    def _download_json(self, url: str) -> dict | None:
        """Download and parse JSON from a URL with retry logic for rate limiting."""
        max_retries = 3
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                # Create request with proper headers to avoid some blocking
                req = Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                with urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("Rate limited (429) downloading subtitles, retrying in %.1f seconds... (attempt %d/%d)", 
                                 delay, attempt + 1, max_retries)
                    time.sleep(delay)
                    continue
                else:
                    logger.warning("Failed to download subtitle data (HTTP %d): %s", e.code, e.reason)
                    return None
            except Exception as e:
                logger.warning("Failed to download subtitle data: %s", e)
                return None
        
        logger.warning("Failed to download subtitle data after %d attempts due to rate limiting", max_retries)
        return None

    def _parse_json3(self, data: dict, language: str = "en") -> list[TranscriptSegment]:
        """Parse YouTube json3 subtitle format into TranscriptSegment list.

        YouTube json3 structure:
            {"events": [{"tStartMs": int, "dDurationMs": int, "segs": [{"utf8": str}]}]}
        """
        segments = []
        for event in data.get("events", []):
            segs = event.get("segs")
            if not segs:
                continue

            text = "".join(s.get("utf8", "") for s in segs).strip()
            if not text or text == "\n":
                continue

            # Translate to English if needed
            if not language.startswith('en'):
                text = _translate_to_english(text, language)

            start_ms = event.get("tStartMs", 0)
            duration_ms = event.get("dDurationMs", 0)

            segments.append(
                TranscriptSegment(
                    start=start_ms / 1000.0,
                    duration=duration_ms / 1000.0,
                    text=text,
                )
            )

        return segments

    def _extract_chapters(self, info: dict) -> list[Chapter]:
        """Extract chapter markers when provided by the uploader."""
        return [
            Chapter(title=ch["title"], start=float(ch.get("start_time", 0)))
            for ch in (info.get("chapters") or [])
            if ch.get("title")
        ]
