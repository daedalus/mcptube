# SPEC.md — mcptube

## Purpose

mcptube is a YouTube video knowledge engine that transforms YouTube videos into a persistent, structured wiki knowledge base using transcripts and visual frame analysis. It exposes functionality via both a CLI and an MCP server for integration with AI coding assistants.

## Scope

### In scope

- YouTube video ingestion (transcript + metadata via yt-dlp)
- Scene-change frame extraction (ffmpeg) and vision-model description
- Persistent wiki knowledge base (entities, topics, concepts, videos)
- Full-text search via SQLite FTS5
- Agentic Q&A over compiled wiki knowledge
- CLI interface for all operations
- MCP server (FastMCP) for tool-based access
- Video discovery and clustering via YouTube search
- Illustrated report generation (single-video and cross-video)
- Wiki export to markdown and HTML

### Not in scope

- Real-time streaming or live video processing
- User authentication or multi-tenancy
- Video hosting or playback
- Web frontend (CLI + MCP only)
- Automatic re-ingestion or polling for video updates

## Public API / Interface

### CLI Commands

| Command | Signature | Behavior |
|---------|-----------|----------|
| `mcptube add <url>` | `add(url: str, text_only: bool, reprocess: bool, max_frames: int)` | Ingest a YouTube video. Raises `VideoAlreadyExistsError`, `ExtractionError`. |
| `mcptube list` | `list_videos()` | List all videos with metadata. |
| `mcptube info <query>` | `info(query: str)` | Show full video details. Raises `VideoNotFoundError`, `AmbiguousVideoError`. |
| `mcptube remove <query>` | `remove(query: str)` | Remove video and clean wiki references. Raises `VideoNotFoundError`. |
| `mcptube search <query>` | `search(query: str, limit: int)` | Full-text wiki search. |
| `mcptube ask <question>` | `ask(question: str)` | Agentic Q&A over wiki. |
| `mcptube wiki list` | `wiki_list(page_type: str, tag: str)` | Browse wiki pages with filtering. |
| `mcptube wiki show <slug>` | `wiki_show(slug: str)` | Read a wiki page in full. |
| `mcptube wiki search <query>` | `wiki_search(query: str, limit: int)` | Full-text search across wiki pages. |
| `mcptube wiki toc` | `wiki_toc()` | Display table of contents. |
| `mcptube wiki history <slug>` | `wiki_history(slug: str)` | Version history for a wiki page. |
| `mcptube wiki export` | `wiki_export(fmt: str, output: str, slug: str)` | Export wiki pages. |
| `mcptube frame <query> <ts>` | `frame(query: str, timestamp: float)` | Extract frame at timestamp. |
| `mcptube frame-query <query> <desc>` | `frame_query(query: str, search_query: str)` | Extract frame by transcript match. |
| `mcptube classify <query>` | `classify(query: str)` | LLM classify + tag a video. |
| `mcptube report <query>` | `report(query: str, focus: str, fmt: str, output: str)` | Generate illustrated report. |
| `mcptube report-query <topic>` | `report_query(query: str, tags: list, fmt: str, output: str)` | Cross-video report. |
| `mcptube synthesize-cmd <topic>` | `synthesize_cmd(topic: str, videos: list, fmt: str, output: str)` | Cross-video synthesis. |
| `mcptube discover <topic>` | `discover(topic: str)` | Search YouTube, cluster results. |
| `mcptube serve` | `serve(stdio: bool, host: str, port: int, reload: bool)` | Start MCP server. |

### MCP Tools

| Tool | Signature | Behavior |
|------|-----------|----------|
| `add_video` | `(url: str, text_only: bool) -> dict` | Ingest video + build wiki. |
| `list_videos` | `() -> list[dict]` | List library. |
| `get_info` | `(video_id: str) -> dict` | Full video details. |
| `remove_video` | `(video_id: str) -> dict` | Remove video + clean wiki. |
| `wiki_list` | `(page_type: str, tag: str) -> list[dict]` | Browse wiki pages. |
| `wiki_show` | `(slug: str) -> dict` | Read wiki page. |
| `wiki_search` | `(query: str, limit: int) -> list[dict]` | Full-text search. |
| `wiki_toc` | `() -> str` | Table of contents. |
| `wiki_ask` | `(question: str) -> dict` | Agentic Q&A (passthrough). |
| `wiki_history` | `(slug: str) -> list[dict]` | Version history. |
| `get_frame` | `(video_id: str, timestamp: float) -> Image` | Extract frame (inline image). |
| `get_frame_by_query` | `(video_id: str, query: str) -> Image` | Frame by transcript match. |
| `get_frame_data` | `(video_id: str, timestamp: float) -> dict` | Base64 frame data. |
| `classify_video` | `(video_id: str) -> dict` | Metadata for classification. |
| `save_tags` | `(video_id: str, tags: list[str]) -> dict` | Save classification tags. |
| `generate_report` | `(video_id: str, query: str) -> dict` | Data for single-video report. |
| `generate_report_from_query` | `(query: str, tags: list[str]) -> dict` | Data for cross-video report. |
| `synthesize` | `(video_ids: list[str], topic: str) -> dict` | Data for theme synthesis. |
| `discover_videos` | `(topic: str) -> dict` | Search YouTube. |
| `ask_video` | `(video_id: str, question: str) -> dict` | Single-video Q&A data. |
| `ask_videos` | `(video_ids: list[str], question: str) -> dict` | Multi-video Q&A data. |

### Core Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `McpTubeService` | `service.py` | Core orchestration — all operations flow through this. |
| `LLMClient` | `llm.py` | LiteLLM wrapper for BYOK LLM operations. |
| `WikiEngine` | `wiki/engine.py` | Wiki merge semantics, ingestion, search, Q&A. |
| `ReportBuilder` | `report.py` | Illustrated report generation. |
| `VideoDiscovery` | `discovery.py` | YouTube search + LLM clustering. |

## Data Formats

### Video Model (Pydantic)

```python
{
    "video_id": "dQw4w9WgXcQ",
    "title": "string",
    "description": "string",
    "channel": "string",
    "duration": 25.0,
    "thumbnail_url": "https://...",
    "chapters": [{"title": "string", "start": 0.0}],
    "transcript": [{"start": 0.0, "duration": 5.0, "text": "string"}],
    "tags": ["AI", "Tutorial"],
    "added_at": "2025-06-15T12:00:00Z",
    "wiki_processed": true
}
```

### Wiki Page Types

- **VideoPage**: Immutable per-video summary + timestamps + key frames
- **EntityPage**: People, tools, companies — append-only references
- **TopicPage**: Broad themes — synthesis rewritten, contributions immutable
- **ConceptPage**: Specific ideas — synthesis rewritten, contributions immutable

### Storage

- Video metadata: SQLite (`~/.mcptube/mcptube.db`)
- Wiki pages: JSON files on disk (`~/.mcptube/wiki/{type}/{slug}.json`)
- FTS5 index: SQLite (`~/.mcptube/wiki.db`)
- Extracted frames: JPEG files (`~/.mcptube/frames/`)

## Edge Cases

1. **Video already in library**: `add_video` raises `VideoAlreadyExistsError` — must remove first or use `--reprocess`.
1. **Ambiguous query**: `resolve_video` raises `AmbiguousVideoError` when substring matches multiple videos.
1. **No LLM configured**: LLM-dependent operations (classify, report, ask) raise `RuntimeError` with clear message.
1. **No transcript available**: `get_frame_by_query` raises `RuntimeError` if video has no transcript.
1. **FTS5 special characters**: Query sanitization strips `?`, `!`, `"`, `(`, `)` to prevent FTS5 syntax errors.
1. **Empty wiki**: Search and Q&A gracefully return empty results or informative messages.
1. **Frame extraction failure**: `FrameExtractionError` raised when ffmpeg fails (missing video, invalid timestamp).
1. **YouTube extraction failure**: `ExtractionError` raised for unavailable/private videos or network errors.

## Performance & Constraints

- FTS5 search: sub-millisecond latency at thousands of pages
- Frame extraction: depends on video resolution and ffmpeg performance
- LLM operations: bounded by API rate limits and model latency
- Wiki merge: O(n) where n is existing entity/topic/concept count
- No hard memory limits; large videos may require significant disk for frames
- Dependencies: ffmpeg required for frame extraction, Python 3.12+
