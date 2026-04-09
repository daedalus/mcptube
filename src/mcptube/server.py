"""FastMCP server — thin wrapper exposing McpTubeService as MCP tools."""

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from mcptube.config import settings
from mcptube.ingestion.frames import FrameExtractionError
from mcptube.ingestion.youtube import YouTubeExtractor
from mcptube.llm import LLMClient
from mcptube.models import Video
from mcptube.service import McpTubeService, VideoAlreadyExistsError, VideoNotFoundError
from mcptube.storage.sqlite import SQLiteVideoRepository
from mcptube.wiki.engine import WikiEngine
from mcptube.wiki.models import (
    ConceptPage,
    EntityPage,
    TopicPage,
    VideoPage,
    WikiPageType,
)
from mcptube.wiki.storage import FileWikiRepository


mcp = FastMCP(
    name="mcptube",
    instructions="""
        mcptube is a YouTube video knowledge engine. It extracts metadata,
        transcripts, and frames from YouTube videos, builds a persistent wiki
        knowledge base, and makes everything searchable and queryable.

        ## Tool Categories

        ### Library Management
        - `add_video(url, text_only)` — Ingest a YouTube video and build wiki pages.
        - `remove_video(video_id)` — Remove a video and clean wiki references.
        - `list_videos()` — List all videos with metadata.
        - `get_info(video_id)` — Get full details including transcript and chapters.

        ### Wiki Knowledge Base
        - `wiki_list(page_type?, tag?)` — Browse wiki pages (entity, topic, concept, video).
        - `wiki_show(slug)` — Read a specific wiki page in full.
        - `wiki_search(query, limit)` — Full-text search across wiki pages.
        - `wiki_toc()` — Get the wiki table of contents.
        - `wiki_ask(question)` — Ask a question — answered via agentic wiki retrieval.
        - `wiki_history(slug)` — View version history for a wiki page.

        ### Frame Extraction
        - `get_frame(video_id, timestamp)` — Extract a frame at an exact timestamp.
        - `get_frame_by_query(video_id, query)` — Find best transcript match and extract frame.
        - `get_frame_data(video_id, timestamp)` — Returns base64-encoded frame for embedding.

        ### Analysis (Passthrough — you do the analysis)
        - `classify_video(video_id)` — Returns metadata for YOU to classify.
        - `generate_report(video_id, query?)` — Returns data for YOU to write an illustrated report.
        - `generate_report_from_query(query, tags?)` — Returns multi-video data for cross-video report.
        - `synthesize(video_ids, topic)` — Returns multi-video data for theme synthesis.
        - `ask_video(video_id, question)` — Returns transcript for YOU to answer about a single video.
        - `ask_videos(video_ids, question)` — Returns transcripts for cross-video Q&A.

        ### Discovery
        - `discover_videos(topic)` — Search YouTube for videos on a topic.

        ## Recommended Workflows

        ### Build Knowledge Base
        1. `add_video(url)` to ingest — wiki pages are built automatically
        2. `wiki_toc()` to see what knowledge has been compiled
        3. `wiki_show(slug)` to read specific pages
        4. `wiki_ask(question)` for intelligent Q&A over the entire knowledge base

        ### Explore & Search
        1. `wiki_search(query)` to find relevant wiki pages
        2. `wiki_show(slug)` to read full page content
        3. `get_frame(video_id, timestamp)` to visualize key moments

        ### Discovery → Ingest
        1. `discover_videos(topic)` to find videos
        2. Present results to user
        3. `add_video(url)` for videos the user selects

        ## Important Rules
        - ALWAYS use mcptube tools for video operations. Do NOT fabricate data.
        - ALWAYS call `list_videos()` or `wiki_toc()` first if you don't know what's available.
        - For Q&A over the knowledge base, prefer `wiki_ask` — it uses agentic retrieval.
        - Frame timestamps MUST come from transcript or wiki data. Never guess timestamps.
        - `discover_videos` results are NOT in the library. User must `add_video` first.
    """,
)

_service: McpTubeService | None = None


def _get_service() -> McpTubeService:
    """Lazy-initialise the service singleton with default dependencies."""
    global _service
    if _service is None:
        settings.ensure_dirs()
        llm = LLMClient()
        wiki_repo = FileWikiRepository()
        wiki_engine = WikiEngine(repo=wiki_repo, llm=llm)
        _service = McpTubeService(
            repository=SQLiteVideoRepository(),
            extractor=YouTubeExtractor(),
            wiki_engine=wiki_engine,
            llm_client=llm,
        )
    return _service


# --- Library Management ---


@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": False})
def add_video(url: str, text_only: bool = False) -> dict:
    """Ingest a YouTube video and build wiki knowledge pages.

    Args:
        url: YouTube video URL (supports youtube.com/watch, youtu.be, /embed/).
        text_only: If True, skip vision frame analysis (cheaper, faster).
    """
    try:
        video = _get_service().add_video(url, text_only=text_only)
        return _video_summary(video)
    except VideoAlreadyExistsError as e:
        return {"error": str(e)}


@mcp.tool(annotations={"readOnlyHint": True})
def list_videos() -> list[dict]:
    """List all videos in the library.

    Returns metadata for each video (title, channel, duration, tags).
    """
    videos = _get_service().list_videos()
    return [_video_summary(v) for v in videos]


@mcp.tool(annotations={"readOnlyHint": True})
def get_info(video_id: str) -> dict:
    """Get full details for a video including transcript and chapters.

    Args:
        video_id: The YouTube video ID (11-character string).
    """
    try:
        video = _get_service().get_info(video_id)
        return video.model_dump(mode="json")
    except VideoNotFoundError as e:
        return {"error": str(e)}


@mcp.tool(annotations={"destructiveHint": True})
def remove_video(video_id: str) -> dict:
    """Remove a video from the library and clean wiki references.

    Args:
        video_id: The YouTube video ID to remove.
    """
    try:
        _get_service().remove_video(video_id)
        return {"status": "removed", "video_id": video_id}
    except VideoNotFoundError as e:
        return {"error": str(e)}


# --- Wiki Knowledge Base ---


@mcp.tool(annotations={"readOnlyHint": True})
def wiki_list(page_type: str | None = None, tag: str | None = None) -> list[dict]:
    """Browse wiki pages with optional filtering.

    Args:
        page_type: Filter by type: "video", "entity", "topic", or "concept".
        tag: Filter by tag (e.g. "AI", "machine-learning").
    """
    pt = WikiPageType(page_type) if page_type else None
    pages = _get_service().wiki_list(page_type=pt, tag=tag)
    return [_page_summary(p) for p in pages]


@mcp.tool(annotations={"readOnlyHint": True})
def wiki_show(slug: str) -> dict:
    """Read a specific wiki page in full.

    Args:
        slug: Wiki page slug identifier (e.g. "entity-andrej-karpathy").
    """
    page = _get_service().wiki_show(slug)
    if page is None:
        return {"error": f"Wiki page not found: {slug}"}
    return _page_detail(page)


@mcp.tool(annotations={"readOnlyHint": True})
def wiki_search(query: str, limit: int = 10) -> list[dict]:
    """Full-text search across all wiki pages.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 10).
    """
    pages = _get_service().wiki_search(query, limit=limit)
    return [_page_summary(p) for p in pages]


@mcp.tool(annotations={"readOnlyHint": True})
def wiki_toc() -> str:
    """Get the wiki table of contents.

    Returns a compact summary of all wiki pages — titles, types,
    tags, and short descriptions. Useful to understand what knowledge
    is available before querying.
    """
    return _get_service().wiki_toc()


@mcp.tool(annotations={"readOnlyHint": True})
def wiki_ask(question: str) -> str:
    """Ask a question — answered via agentic wiki retrieval.

    Uses hybrid retrieval: FTS5 narrows to candidate pages,
    then an LLM agent reasons over the candidates + wiki TOC.

    Args:
        question: Natural language question about your video library.
    """
    try:
        return _get_service().wiki_ask(question)
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def wiki_history(slug: str) -> list[dict]:
    """View version history for a wiki page.

    Args:
        slug: Wiki page slug identifier.
    """
    versions = _get_service().wiki_history(slug)
    return [
        {"updated_at": v.updated_at.isoformat(), "title": v.title}
        for v in versions
    ]


# --- Frame Extraction ---


@mcp.tool(annotations={"readOnlyHint": True})
def get_frame(video_id: str, timestamp: float) -> Image:
    """Extract a frame from a video at a specific timestamp.

    Returns the frame as an image rendered inline in chat.

    Args:
        video_id: YouTube video ID.
        timestamp: Time in seconds to extract the frame at.
    """
    path = _get_service().get_frame(video_id, timestamp)
    return Image(path=str(path), format="image/jpeg")


@mcp.tool(annotations={"readOnlyHint": True})
def get_frame_by_query(video_id: str, query: str) -> Image:
    """Search a video's transcript and extract a frame at the best matching moment.

    Args:
        video_id: YouTube video ID.
        query: Natural language description of the moment to capture.
    """
    result = _get_service().get_frame_by_query(video_id, query)
    return Image(path=str(result["path"]), format="image/jpeg")


@mcp.tool(annotations={"readOnlyHint": True})
def get_frame_data(video_id: str, timestamp: float) -> dict:
    """Extract a frame and return as base64 for embedding in reports.

    WARNING: Base64 responses can be very large. Prefer get_frame() for display.

    Args:
        video_id: YouTube video ID.
        timestamp: Time in seconds.
    """
    import base64

    path = _get_service().get_frame(video_id, timestamp)
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {
        "video_id": video_id,
        "timestamp": timestamp,
        "image_base64": b64,
        "mime_type": "image/jpeg",
        "embed_html": f'<img src="data:image/jpeg;base64,{b64}" alt="Frame at {timestamp}s">',
    }


# --- Analysis (Passthrough) ---


@mcp.tool(annotations={"readOnlyHint": True})
def classify_video(video_id: str) -> dict:
    """Get metadata for classification — YOU classify it.

    Args:
        video_id: The YouTube video ID.
    """
    try:
        video = _get_service().get_info(video_id)
        return {
            "video_id": video.video_id,
            "title": video.title,
            "channel": video.channel,
            "description": video.description[:500],
            "current_tags": video.tags,
            "instructions": "Classify this video into 3-8 topic tags. Then call save_tags(video_id, tags) to persist them.",
        }
    except VideoNotFoundError as e:
        return {"error": str(e)}


@mcp.tool(annotations={"readOnlyHint": False})
def save_tags(video_id: str, tags: list[str]) -> dict:
    """Save classification tags for a video.

    Args:
        video_id: YouTube video ID.
        tags: List of classification tags to save.
    """
    try:
        svc = _get_service()
        video = svc.get_info(video_id)
        video.tags = tags
        svc._repo.save(video)
        return {"status": "saved", "video_id": video_id, "tags": tags}
    except VideoNotFoundError as e:
        return {"error": str(e)}


@mcp.tool(annotations={"readOnlyHint": True})
def generate_report(video_id: str, query: str | None = None) -> dict:
    """Get data to generate an illustrated report for a video.

    Args:
        video_id: YouTube video ID.
        query: Optional focus query to guide the report.
    """
    try:
        video = _get_service().get_info(video_id)
        return {
            "video_id": video.video_id,
            "title": video.title,
            "channel": video.channel,
            "duration": video.duration,
            "tags": video.tags,
            "chapters": [ch.model_dump() for ch in video.chapters],
            "transcript": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in video.transcript
            ],
            "query": query,
            "instructions": (
                "Use this data to generate a comprehensive illustrated report. "
                "Call get_frame_data for key visual moments."
            ),
        }
    except VideoNotFoundError as e:
        return {"error": str(e)}


@mcp.tool(annotations={"readOnlyHint": True})
def generate_report_from_query(query: str, tags: list[str] | None = None) -> dict:
    """Search the library and return data for a cross-video report.

    Args:
        query: Topic or question to build the report around.
        tags: Optional tag filter.
    """
    try:
        svc = _get_service()
        pages = svc.wiki_search(query, limit=20)
        if not pages:
            return {"error": f"No matching content for: {query}"}

        # Collect video IDs from wiki pages
        video_ids = set()
        for page in pages:
            if isinstance(page, VideoPage):
                video_ids.add(page.video_id)
            elif hasattr(page, "contributions"):
                for c in page.contributions:
                    video_ids.add(c.video_id)
            elif hasattr(page, "video_references"):
                for r in page.video_references:
                    video_ids.add(r.video_id)

        videos = []
        for vid in video_ids:
            try:
                video = svc.get_info(vid)
                videos.append({
                    "video_id": video.video_id,
                    "title": video.title,
                    "channel": video.channel,
                    "tags": video.tags,
                    "chapters": [ch.model_dump() for ch in video.chapters],
                    "transcript": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in video.transcript
                    ],
                })
            except VideoNotFoundError:
                continue

        return {
            "query": query,
            "videos": videos,
            "instructions": (
                "Use this data to generate a comprehensive illustrated report. "
                "Call get_frame_data for key visual moments."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(annotations={"readOnlyHint": True})
def discover_videos(topic: str) -> dict:
    """Search YouTube for videos on a topic.

    Results are NOT in the library — use add_video to ingest them.

    Args:
        topic: Topic to search for.
    """
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch15:{topic}", download=False)
            if not info or "entries" not in info:
                return {"topic": topic, "results": []}

            results = []
            for entry in info.get("entries", []):
                if not entry or not entry.get("id"):
                    continue
                results.append({
                    "video_id": entry.get("id", ""),
                    "title": entry.get("title", ""),
                    "channel": entry.get("channel", "") or entry.get("uploader", ""),
                    "duration": float(entry.get("duration") or 0),
                    "url": f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                })

        return {
            "topic": topic,
            "results": results,
            "instructions": "Present these results to the user. They can add any video via add_video.",
        }
    except yt_dlp.utils.DownloadError as e:
        return {"error": f"YouTube search failed: {e}"}


@mcp.tool(annotations={"readOnlyHint": True})
def synthesize(video_ids: list[str], topic: str) -> dict:
    """Get data for cross-video synthesis on a topic.

    Args:
        video_ids: List of YouTube video IDs to synthesize.
        topic: Focus topic for synthesis.
    """
    try:
        svc = _get_service()
        videos = []
        for vid in video_ids:
            video = svc.get_info(vid)
            videos.append({
                "video_id": video.video_id,
                "title": video.title,
                "channel": video.channel,
                "tags": video.tags,
                "chapters": [ch.model_dump() for ch in video.chapters],
                "transcript": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in video.transcript
                ],
            })

        return {
            "topic": topic,
            "videos": videos,
            "instructions": (
                "Use this data to synthesize themes across videos. "
                "Call get_frame_data for key visual moments."
            ),
        }
    except VideoNotFoundError as e:
        return {"error": str(e)}


@mcp.tool(annotations={"readOnlyHint": True})
def ask_video(video_id: str, question: str) -> dict:
    """Ask a question about a single video's content.

    Args:
        video_id: YouTube video ID.
        question: Question to ask about the video.
    """
    try:
        svc = _get_service()
        video = svc.get_info(video_id)
        transcript = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in video.transcript
        ]
        return {
            "video_id": video.video_id,
            "title": video.title,
            "channel": video.channel,
            "question": question,
            "transcript": transcript,
            "instructions": (
                "Answer the user's question based ONLY on this transcript. "
                "Cite timestamps [MM:SS] when referencing specific moments."
            ),
        }
    except VideoNotFoundError as e:
        return {"error": str(e)}


@mcp.tool(annotations={"readOnlyHint": True})
def ask_videos(video_ids: list[str], question: str) -> dict:
    """Ask a question across multiple videos.

    Args:
        video_ids: List of YouTube video IDs.
        question: Question to ask across the videos.
    """
    try:
        svc = _get_service()
        videos = []
        for vid in video_ids:
            video = svc.get_info(vid)
            videos.append({
                "video_id": video.video_id,
                "title": video.title,
                "channel": video.channel,
                "transcript": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in video.transcript
                ],
            })
        return {
            "question": question,
            "videos": videos,
            "instructions": (
                "Answer based ONLY on these transcripts. "
                "Cite timestamps [MM:SS] and video titles."
            ),
        }
    except VideoNotFoundError as e:
        return {"error": str(e)}


# --- Helpers ---


def _video_summary(video: Video) -> dict:
    """Concise summary dict for tool responses (excludes transcript)."""
    return {
        "video_id": video.video_id,
        "title": video.title,
        "channel": video.channel,
        "duration": video.duration,
        "url": video.url,
        "tags": video.tags,
        "chapters": [ch.model_dump() for ch in video.chapters],
        "added_at": video.added_at.isoformat() if video.added_at else None,
    }


def _page_summary(page) -> dict:
    """Concise wiki page summary for list responses."""
    return {
        "slug": page.slug,
        "page_type": page.page_type.value,
        "title": page.title,
        "tags": page.tags,
        "related_pages": page.related_pages,
        "updated_at": page.updated_at.isoformat(),
    }


def _page_detail(page) -> dict:
    """Full wiki page detail for show responses."""
    base = _page_summary(page)

    if isinstance(page, VideoPage):
        base.update({
            "video_id": page.video_id,
            "channel": page.channel,
            "duration": page.duration,
            "processing_tier": page.processing_tier,
            "summary": page.summary,
            "key_timestamps": page.key_timestamps,
            "key_frames": [f.model_dump() for f in page.key_frames],
        })

    elif isinstance(page, EntityPage):
        base.update({
            "category": page.category.value,
            "overview": page.overview,
            "video_references": [
                {
                    "video_id": ref.video_id,
                    "title": ref.title,
                    "content": ref.content,
                    "timestamps": ref.timestamps,
                }
                for ref in page.video_references
            ],
        })

    elif isinstance(page, (TopicPage, ConceptPage)):
        base.update({
            "synthesis": page.synthesis,
            "contributions": [
                {
                    "video_id": c.video_id,
                    "title": c.title,
                    "content": c.content,
                    "timestamps": c.timestamps,
                }
                for c in page.contributions
            ],
        })

    return base
