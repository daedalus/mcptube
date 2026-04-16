#!/usr/bin/env python3
"""Parse YouTube watch history and extract video IDs with pagination."""

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

import requests
import yt_dlp

try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_YOUTUBE_VIDEO_ID = re.compile(r"(?:youtube\.com/watch\?.*v=)([\w-]{11})")
_YOUTUBE_SHORT = re.compile(r"(?:youtu\.be/)([\w-]{11})")


def parse_cookies(path: str) -> dict:
    """Parse Netscape cookies file into dict."""
    cookies = {}
    if not Path(path).exists():
        return cookies

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies[parts[5]] = parts[6]

    logger.debug("Parsed %d cookies", len(cookies))
    return cookies


def extract_ids(text: str) -> list[str]:
    """Extract video IDs from text."""
    ids = set()

    for p in [_YOUTUBE_VIDEO_ID, _YOUTUBE_SHORT]:
        for m in p.finditer(text):
            if len(m.group(1)) == 11:
                ids.add(m.group(1))

    for m in re.finditer(r'"videoId":"([\w-]{11})"', text):
        ids.add(m.group(1))

    return sorted(ids)


def find_continuation_token(text: str) -> str | None:
    """Find continuation token in the page response."""
    # Pattern 1: continuationCommand
    match = re.search(r'"continuationCommand":{"token":"([^"]+)"', text)
    if match:
        return match.group(1)

    # Pattern 2: continuation token in browse_ajax response
    match = re.search(r'"token":"(action_continuation[^"]+)"', text)
    if match:
        return match.group(1)

    return None


def fetch_history_with_playwright(
    url: str, browser: str, proxy: str | None, max_pages: int = 10
) -> list[str]:
    """Use Playwright to scroll and load more history."""
    if not HAS_PLAYWRIGHT:
        logger.warning(
            "Playwright not installed. Install: pip install playwright && playwright install chromium"
        )
        return []

    all_ids = []
    seen_ids = set()

    try:
        with sync_playwright() as p:
            # Launch browser with proxy if provided
            launch_opts = {"headless": True}
            if proxy:
                launch_opts["proxy"] = {"server": proxy}

            browser = p.chromium.launch(**launch_opts)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()

            # Go to history page
            page.goto("https://www.youtube.com/history", timeout=30000)
            page.wait_for_load_state("networkidle")

            # Scroll to load more content
            scroll_count = 0
            max_scrolls = max_pages * 5  # Each scroll loads more items

            for _ in range(max_scrolls):
                # Scroll down
                page.evaluate("window.scrollBy(0, 1000)")
                page.wait_for_timeout(500)  # Wait for content to load

                # Extract video IDs from current view
                page_ids = extract_ids(page.content())
                new_ids = [vid for vid in page_ids if vid not in seen_ids]

                if new_ids:
                    all_ids.extend(new_ids)
                    seen_ids.update(page_ids)
                    scroll_count += 1
                    logger.info(
                        "After scroll %d: %d total, %d new",
                        scroll_count,
                        len(all_ids),
                        len(new_ids),
                    )

                # Stop if no new IDs for multiple scrolls
                if len(new_ids) == 0 and scroll_count > 5:
                    logger.info("No more content to load after %d scrolls", scroll_count)
                    break

            browser.close()
            logger.info("Playwright: found %d video IDs", len(all_ids))

    except Exception as e:
        logger.warning("Playwright failed: %s", e)

    return all_ids


def fetch_history(
    url: str, browser: str, proxy: str | None, max_pages: int = 10
) -> tuple[list[str], int]:
    """Fetch history page - tries both methods."""
    # Try Playwright first if available and no proxy (Playwright handles cookies better)
    if HAS_PLAYWRIGHT and not proxy:
        logger.info("Trying Playwright method...")
        ids = fetch_history_with_playwright(url, browser, proxy, max_pages)
        if ids:
            return ids, 1

    # Use requests method (works with proxy)
    logger.info("Using requests method...")
    cookie_file = "/tmp/yt_cookies.txt"

    try:
        subprocess.run(
            [
                "yt-dlp",
                "--cookies-from-browser",
                browser,
                "--cookies",
                cookie_file,
                "https://www.youtube.com",
            ],
            capture_output=True,
            timeout=60,
        )
    except Exception as e:
        logger.warning("Cookie export failed: %s", e)

    proxies = {"http": proxy, "https": proxy} if proxy else None
    cookies = parse_cookies(cookie_file)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    all_ids = []
    seen_ids = set()
    pages_fetched = 0

    try:
        resp = requests.get(
            "https://www.youtube.com/history",
            headers=headers,
            cookies=cookies,
            proxies=proxies,
            timeout=30,
        )

        if resp.status_code == 200:
            page_ids = extract_ids(resp.text)
            new_ids = [vid for vid in page_ids if vid not in seen_ids]
            all_ids.extend(new_ids)
            seen_ids.update(page_ids)
            pages_fetched += 1
            logger.info("Found %d IDs from history page", len(new_ids))

            # Try continuation token
            continuation = find_continuation_token(resp.text)
            if continuation:
                logger.info("Trying continuation: %s...", continuation[:30])

                # Use browse_ajax continuation (may be deprecated but try anyway)
                cont_url = f"https://www.youtube.com/browse_ajax?action_continuation=1&continuation={continuation}"

                resp2 = requests.get(
                    cont_url, headers=headers, cookies=cookies, proxies=proxies, timeout=30
                )
                if resp2.status_code == 200:
                    page_ids2 = extract_ids(resp2.text)
                    new_ids2 = [vid for vid in page_ids2 if vid not in seen_ids]
                    if new_ids2:
                        all_ids.extend(new_ids2)
                        seen_ids.update(page_ids2)
                        pages_fetched += 1
                        logger.info("Continuation: %d more IDs", len(new_ids2))

    except Exception as e:
        logger.warning("Request failed: %s", e)

    return all_ids, pages_fetched


def main():
    parser = argparse.ArgumentParser(description="Parse YouTube watch history")
    parser.add_argument("--url", default="https://www.youtube.com/history")
    parser.add_argument("--browser", default="chrome")
    parser.add_argument("--proxy")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-n", "--max-pages", type=int, default=5, help="Max pages to fetch")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Fetching YouTube history with max %d pages...", args.max_pages)
    ids, pages = fetch_history(args.url, args.browser, args.proxy, args.max_pages)

    logger.info("Found %d video IDs from %d pages", len(ids), pages)

    for vid in ids:
        print(f"https://www.youtube.com/watch?v={vid}")

    if args.output:
        Path(args.output).write_text(json.dumps(ids, indent=2))
        logger.info("Saved to: %s", args.output)


if __name__ == "__main__":
    main()
