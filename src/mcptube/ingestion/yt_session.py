"""Shared yt-dlp session helpers: network opts + retry with backoff.

Centralizes what was duplicated across frames.py, scene_frames.py, and
youtube.py: (1) conditional injection of cookies / proxy / JS runtime into
yt-dlp opts; (2) retry with exponential backoff for transient failures.
"""

import logging
import time
from pathlib import Path

import yt_dlp

from mcptube.config import settings

logger = logging.getLogger(__name__)

_BACKOFF_BASE_S = 2.0


def build_ydl_opts(base_opts: dict | None = None) -> dict:
    """Build yt-dlp options from settings, layered over optional base opts.

    Reads settings at call time (not import time) for testability.

    Args:
        base_opts: Optional starting opts dict. If None, uses defaults.

    Returns:
        Merged opts dict ready for yt_dlp.YoutubeDL().
    """
    opts = dict(base_opts) if base_opts else {}

    # Quiet mode — always set
    opts.setdefault("quiet", True)
    opts.setdefault("no_warnings", True)
    opts.setdefault("skip_download", True)

    # Cookies
    cookie_file = _resolve_cookie_file()
    if cookie_file:
        opts["cookiefile"] = str(cookie_file)
        logger.debug("Using cookies from: %s", cookie_file)

    # JS runtime (for yt-dlp 2026+ YouTube JS challenges)
    if settings.js_runtimes:
        opts["js_runtimes"] = {settings.js_runtimes: {}}
        logger.debug("Using JS runtime: %s", settings.js_runtimes)

    # Proxy
    if settings.no_proxy:
        opts["proxy"] = ""
        logger.debug("Proxy disabled for yt-dlp")
    elif settings.proxy:
        opts["proxy"] = settings.proxy
        logger.debug("Using proxy: %s", settings.proxy)

    # Browser cookies
    if settings.cookies_from_browser:
        opts["cookies_from_browser"] = (settings.cookies_from_browser, {})
        logger.debug("Using cookies from browser: %s", settings.cookies_from_browser)

    # Format preference
    if settings.format:
        opts["format"] = settings.format
        logger.debug("Using video format: %s", settings.format)

    return opts


def extract_info_with_retry(
    url: str,
    opts: dict,
    attempts: int = 3,
) -> dict | None:
    """yt-dlp extract_info with retry and exponential backoff.

    Handles transient failures (bot checks, rate limits, network blips)
    that are common with YouTube, especially on datacenter IPs.

    Args:
        url: Video URL to extract.
        opts: yt-dlp options dict.
        attempts: Max retry attempts.

    Returns:
        yt-dlp info dict, or None if extraction yielded nothing.

    Raises:
        yt_dlp.utils.DownloadError: If all attempts fail.
    """
    last_error: yt_dlp.utils.DownloadError | None = None

    for attempt in range(1, attempts + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            if attempt < attempts:
                delay = _BACKOFF_BASE_S * attempt
                logger.warning(
                    "yt-dlp failed (attempt %d/%d) for %s: %s — retrying in %.1fs",
                    attempt,
                    attempts,
                    url,
                    str(e)[:120],
                    delay,
                )
                time.sleep(delay)

    assert last_error is not None
    raise last_error


def _resolve_cookie_file() -> Path | None:
    """Resolve cookie file from settings, data dir, or current directory."""
    if settings.cookies_file:
        return settings.cookies_file

    # Check data dir
    try:
        cookie_path = settings.data_dir / ".cookies.txt"
        if cookie_path.exists():
            return cookie_path
    except Exception:
        pass

    # Check current directory
    fallback = Path(".cookies.txt")
    return fallback if fallback.exists() else None
