"""Tests for the platform allowlist and resolver (SSRF gate)."""

import pytest

from mcptube.ingestion.platforms import (
    UnsupportedPlatformError,
    namespaced_id,
    resolve_platform,
)


@pytest.mark.parametrize(
    "url,platform",
    [
        ("https://www.youtube.com/watch?v=BpibZSMGtdY", "youtube"),
        ("https://youtu.be/BpibZSMGtdY", "youtube"),
        ("https://www.instagram.com/reel/CxYz/", "instagram"),
        ("https://www.tiktok.com/@u/video/7263", "tiktok"),
        ("https://vm.tiktok.com/ZMabc/", "tiktok"),
        ("https://www.facebook.com/watch/?v=123", "facebook"),
        ("https://fb.watch/abc/", "facebook"),
    ],
)
def test_resolves_allowlisted_hosts(url, platform):
    assert resolve_platform(url) == platform


@pytest.mark.parametrize(
    "url",
    [
        "https://169.254.169.254.attacker.tld/youtube.com/watch?v=BpibZSMGtdY",
        "https://youtube.com.attacker.tld/x",
        "https://example.com/not-supported",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ],
)
def test_rejects_non_allowlisted_hosts(url):
    with pytest.raises(UnsupportedPlatformError):
        resolve_platform(url)


def test_namespaced_id():
    assert namespaced_id("youtube", "BpibZSMGtdY") == "youtube_BpibZSMGtdY"


def test_namespaced_id_tiktok():
    assert namespaced_id("tiktok", "7263") == "tiktok_7263"
