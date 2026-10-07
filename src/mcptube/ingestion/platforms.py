"""Platform allowlist and resolver — SSRF gate for ingestion.

Every URL that reaches yt-dlp must pass through resolve_platform() first.
This prevents SSRF attacks where an attacker tricks yt-dlp into fetching
an internal host (e.g. 169.254.169.254 metadata endpoint).
"""

from urllib.parse import urlparse

from mcptube.ingestion.youtube import ExtractionError


class UnsupportedPlatformError(ExtractionError):
    """URL host is not in the supported platform allowlist."""


# Host suffix -> platform name. Matches if host equals the suffix exactly
# or ends with ".<suffix>" (subdomain). This is the SSRF gate.
_ALLOWLIST: dict[str, str] = {
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "instagram.com": "instagram",
    "tiktok.com": "tiktok",
    "vm.tiktok.com": "tiktok",
    "facebook.com": "facebook",
    "fb.watch": "facebook",
}


def resolve_platform(url: str) -> str:
    """Return the platform for an allowlisted URL, or raise.

    Args:
        url: A video URL from a supported platform.

    Returns:
        Platform name string (e.g. "youtube", "tiktok").

    Raises:
        UnsupportedPlatformError: If the host is not in the allowlist.
    """
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for suffix, platform in _ALLOWLIST.items():
        if host == suffix or host.endswith("." + suffix):
            return platform
    raise UnsupportedPlatformError(f"Unsupported platform for URL: {url}")


def namespaced_id(platform: str, native_id: str) -> str:
    """Unique, path-safe video ID: '{platform}_{native_id}'."""
    return f"{platform}_{native_id}"
