"""Core business logic for mcptube-vision."""

import logging
from pathlib import Path

from mcptube.config import settings
from mcptube.ingestion.frames import FrameExtractionError, FrameExtractor
from mcptube.ingestion.youtube import ExtractionError, YouTubeExtractor
from mcptube.llm import LLMClient, LLMError
from mcptube.models import Video
from mcptube.report import Report, ReportBuilder
from mcptube.discovery import DiscoveryResult, VideoDiscovery
from mcptube.storage.repository import VideoRepository
from mcptube.wiki.engine import WikiEngine
from mcptube.wiki.models import WikiPageBase, WikiPageType

from mcptube.ingestion.scene_frames import SceneFrameError, SceneFrameExtractor
from mcptube.ingestion.vision import FrameCacheDB, VisionDescriber
from mcptube.wiki.models import VideoPage


logger = logging.getLogger(__name__)


class VideoNotFoundError(Exception):
    """Raised when a requested video is not in the library."""


class VideoAlreadyExistsError(Exception):
    """Raised when attempting to add a video that is already in the library."""


class AmbiguousVideoError(Exception):
    """Raised when a query matches multiple videos and cannot be disambiguated."""


class McpTubeService:
    """Core service layer — single orchestration point for all mcptube operations.

    Both the CLI and MCP server are thin wrappers over this class.
    Dependencies are injected via constructor for testability and
    backend swappability (DIP).
    """

    def __init__(
        self,
        repository: VideoRepository,
        extractor: YouTubeExtractor | None = None,
        wiki_engine: WikiEngine | None = None,
        frame_extractor: FrameExtractor | None = None,
        llm_client: LLMClient | None = None,
        scene_extractor: SceneFrameExtractor | None = None,
        vision_describer: VisionDescriber | None = None,
        max_frames: int | None = None,
    ) -> None:
        self._repo = repository
        self._extractor = extractor or YouTubeExtractor()
        self._wiki = wiki_engine
        self._frame_extractor = frame_extractor or FrameExtractor()
        self._llm = llm_client or LLMClient()
        self._report_builder: ReportBuilder | None = None
        self._discovery: VideoDiscovery | None = None
        self._scene_extractor = scene_extractor or SceneFrameExtractor()
        self._max_frames = max_frames or settings.max_frames
        frame_cache = FrameCacheDB() if vision_describer is None else vision_describer._cache
        self._vision_describer = vision_describer or VisionDescriber(self._llm, frame_cache)

        if self._llm.available:
            self._discovery = VideoDiscovery(llm=self._llm)
            self._report_builder = ReportBuilder(
                llm=self._llm,
                frame_extractor=self._frame_extractor,
            )

        settings.ensure_dirs()

    # --- Video ingestion ---

    def add_video(self, url: str, text_only: bool = False) -> Video:
        """Ingest a YouTube video into the library and wiki.

        Extracts metadata and transcript, persists to storage,
        and builds wiki pages from the content.

        Args:
            url: YouTube video URL in any standard format.
            text_only: If True, skip vision frame analysis.

        Returns:
            The ingested Video model.

        Raises:
            ExtractionError: If video extraction fails.
            VideoAlreadyExistsError: If the video is already in the library.
        """
        video_id = YouTubeExtractor.parse_video_id(url)

        if self._repo.exists(video_id):
            raise VideoAlreadyExistsError(
                f"Video already in library: {video_id}. Use remove_video() first to re-ingest."
            )

        logger.info("Ingesting video: %s", url)
        logger.debug("Extracting URL: %s", url)
        video = self._extractor.extract(url)
        logger.debug("Extracted video: %s - %s (%s)", video.video_id, video.title, video.duration)
        self._repo.save(video)

        # Auto-classify if LLM is available
        if self._llm and self._llm.available:
            try:
                video.tags = self._llm.classify(video.title, video.description, video.channel)
                self._repo.save(video)
                logger.info("Auto-classified: %s", video.tags)
            except LLMError as e:
                logger.warning("Auto-classification failed: %s", e)

        # Build wiki pages
        # Vision pipeline + wiki ingest
        frame_stats = {"ffmpeg_extracted": 0, "llm_processed": 0}
        # Scene frames extraction uses ffprobe, doesn't need LLM
        frames = []
        if not text_only:
            try:
                frames = self._scene_extractor.extract_scene_frames(
                    video.video_id, max_frames=self._max_frames
                )
                frame_stats["ffmpeg_extracted"] = len(frames)
                logger.info("Scene frames: extracted %d frames", len(frames))
            except SceneFrameError as e:
                logger.warning("Scene frame extraction failed: %s", e)

        # LLM processing (wiki ingest) needs LLM available
        if self._wiki and self._llm and self._llm.available:
            try:
                frame_descriptions = None
                # Only describe frames if we extracted frames AND LLM is available
                if frame_stats["ffmpeg_extracted"] > 0:
                    try:
                        frame_descriptions = self._vision_describer.describe_frames(frames)
                        frame_stats["llm_processed"] = len(frame_descriptions)
                        logger.info("Vision: described %d frames", len(frame_descriptions))
                    except LLMError as e:
                        logger.warning("Frame description failed: %s", e)

                stats = self._wiki.ingest_video(
                    video, frame_descriptions=frame_descriptions, text_only=text_only
                )
                logger.info("Wiki ingest: %s", stats)
            except LLMError as e:
                logger.warning("Wiki ingest failed: %s", e)

        video.frame_stats = frame_stats
        self._repo.save(video)
        logger.info("Video added: %s — %s", video.video_id, video.title)
        return video

    # --- Video management ---

    def list_videos(self) -> list[Video]:
        """List all videos in the library (metadata only, no transcripts)."""
        return self._repo.list_all()

    def get_info(self, video_id: str) -> Video:
        """Get full video information including transcript.

        Raises:
            VideoNotFoundError: If the video is not in the library.
        """
        video = self._repo.get(video_id)
        if video is None:
            raise VideoNotFoundError(f"Video not found: {video_id}")
        return video

    def remove_video(self, video_id: str) -> None:
        """Remove a video from the library and wiki.

        Raises:
            VideoNotFoundError: If the video is not in the library.
        """
        if not self._repo.exists(video_id):
            raise VideoNotFoundError(f"Video not found: {video_id}")
        self._repo.delete(video_id)

        # Clean wiki references
        if self._wiki:
            self._wiki.remove_video(video_id)

        logger.info("Video removed: %s", video_id)

    def reprocess_video(self, video_id: str, text_only: bool = False) -> Video:
        """Re-process an existing video without removing it first.

        Args:
            video_id: The video ID to re-process.
            text_only: Whether to skip vision frame analysis.

        Returns:
            The re-processed Video model.
        """
        if not self._repo.exists(video_id):
            raise VideoNotFoundError(f"Video not found: {video_id}")

        existing_video = self._repo.get(video_id)
        url = existing_video.url

        logger.info("Re-processing video: %s", video_id)

        # Clean existing wiki data (don't delete from repo)
        if self._wiki:
            self._wiki.remove_video(video_id)

        # Re-extract to get fresh metadata (file_size, format, etc.)
        logger.debug("Re-extracting video metadata from YouTube")
        video = self._extractor.extract(url)
        self._repo.save(video)

        # Auto-classify if LLM is available
        if self._llm and self._llm.available:
            try:
                video.tags = self._llm.classify(video.title, video.description, video.channel)
                self._repo.save(video)
                logger.info("Auto-classified: %s", video.tags)
            except LLMError as e:
                logger.warning("Auto-classification failed: %s", e)

        # Build wiki pages
        frame_stats = {"ffmpeg_extracted": 0, "llm_processed": 0}
        # Scene frames extraction uses ffprobe, doesn't need LLM
        frames = []
        if not text_only:
            try:
                frames = self._scene_extractor.extract_scene_frames(
                    video.video_id, max_frames=self._max_frames
                )
                frame_stats["ffmpeg_extracted"] = len(frames)
                logger.info("Scene frames: extracted %d frames", len(frames))
            except SceneFrameError as e:
                logger.warning("Scene frame extraction failed: %s", e)

        # LLM processing (wiki ingest) needs LLM available
        if self._wiki and self._llm and self._llm.available:
            try:
                frame_descriptions = None
                # Only describe frames if we extracted frames AND LLM is available
                if frame_stats["ffmpeg_extracted"] > 0:
                    try:
                        frame_descriptions = self._vision_describer.describe_frames(frames)
                        frame_stats["llm_processed"] = len(frame_descriptions)
                        logger.info("Vision: described %d frames", len(frame_descriptions))
                    except LLMError as e:
                        logger.warning("Frame description failed: %s", e)

                stats = self._wiki.ingest_video(
                    video, frame_descriptions=frame_descriptions, text_only=text_only
                )
                logger.info("Wiki ingest: %s", stats)
            except LLMError as e:
                logger.warning("Wiki ingest failed: %s", e)

        video.frame_stats = frame_stats
        self._repo.save(video)
        logger.info("Video re-processed: %s — %s", video.video_id, video.title)
        return video

    # --- Wiki operations ---

    def wiki_search(self, query: str, limit: int = 10) -> list[WikiPageBase]:
        """Search wiki pages via FTS5.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching wiki pages.

        Raises:
            RuntimeError: If wiki engine is not configured.
        """
        if not self._wiki:
            raise RuntimeError("Wiki search requires a wiki engine.")
        return self._wiki.search(query, limit=limit)

    def wiki_ask(self, question: str) -> str:
        """Ask a question — agentic hybrid retrieval over wiki.

        Args:
            question: User's question.

        Returns:
            Answer string.

        Raises:
            RuntimeError: If wiki engine is not configured.
        """
        if not self._wiki:
            raise RuntimeError("Wiki Q&A requires a wiki engine.")
        return self._wiki.ask(question)

    def wiki_list(
        self,
        page_type: WikiPageType | None = None,
        tag: str | None = None,
    ) -> list[WikiPageBase]:
        """List wiki pages with optional filtering.

        Args:
            page_type: Filter by page type.
            tag: Filter by tag.

        Returns:
            List of wiki pages.

        Raises:
            RuntimeError: If wiki engine is not configured.
        """
        if not self._wiki:
            raise RuntimeError("Wiki requires a wiki engine.")
        return self._wiki.list_pages(page_type=page_type, tag=tag)

    def wiki_show(self, slug: str) -> WikiPageBase | None:
        """Get a specific wiki page by slug.

        Args:
            slug: Page slug identifier.

        Returns:
            Wiki page or None.

        Raises:
            RuntimeError: If wiki engine is not configured.
        """
        if not self._wiki:
            raise RuntimeError("Wiki requires a wiki engine.")
        return self._wiki.get_page(slug)

    def wiki_toc(self) -> str:
        """Get the wiki table of contents.

        Returns:
            Formatted TOC string.

        Raises:
            RuntimeError: If wiki engine is not configured.
        """
        if not self._wiki:
            raise RuntimeError("Wiki requires a wiki engine.")
        return self._wiki.get_toc()

    def wiki_history(self, slug: str) -> list[WikiPageBase]:
        """Get version history for a wiki page.

        Args:
            slug: Page slug identifier.

        Returns:
            List of previous versions.

        Raises:
            RuntimeError: If wiki engine is not configured.
        """
        if not self._wiki:
            raise RuntimeError("Wiki requires a wiki engine.")
        return self._wiki.get_page_history(slug)

    # --- Search (backward compatible — now uses wiki) ---

    def search(
        self, query: str, video_id: str | None = None, limit: int = 10
    ) -> list[WikiPageBase]:
        """Search across the knowledge base.

        Uses wiki FTS5 search. For video-scoped search, falls back
        to transcript text search.

        Args:
            query: Natural language search query.
            video_id: If provided, scope search to a single video.
            limit: Maximum number of results.

        Returns:
            List of matching wiki pages.
        """
        if not self._wiki:
            raise RuntimeError("Search requires a wiki engine.")

        if video_id:
            # Scoped search — get the video page and search its content
            page = self._wiki.get_page(f"video-{video_id}")
            return [page] if page else []

        return self._wiki.search(query, limit=limit)

    # --- Frames ---

    def get_frame(self, video_id: str, timestamp: float) -> Path:
        """Extract a frame at a specific timestamp.

        Raises:
            VideoNotFoundError: If the video is not in the library.
            FrameExtractionError: If frame extraction fails.
        """
        if not self._repo.exists(video_id):
            raise VideoNotFoundError(f"Video not found: {video_id}")
        return self._frame_extractor.extract_frame(video_id, timestamp)

    def get_frame_by_query(self, video_id: str, query: str) -> dict:
        """Search transcript and extract a frame at the best matching moment.

        Uses wiki search to find the relevant moment, then extracts the frame.

        Raises:
            VideoNotFoundError: If the video is not in the library.
            FrameExtractionError: If frame extraction fails.
        """
        if not self._repo.exists(video_id):
            raise VideoNotFoundError(f"Video not found: {video_id}")

        # Get full video with transcript to find best timestamp
        video = self.get_info(video_id)
        if not video.transcript:
            raise RuntimeError(f"No transcript available for: {video_id}")

        # Simple keyword match over transcript segments
        query_lower = query.lower()
        best_seg = None
        best_score = -1

        for seg in video.transcript:
            text_lower = seg.text.lower()
            score = sum(1 for word in query_lower.split() if word in text_lower)
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_seg is None:
            raise VideoNotFoundError(f"No transcript match for query: {query}")

        frame_path = self._frame_extractor.extract_frame(video_id, best_seg.start)
        return {
            "path": frame_path,
            "start": best_seg.start,
            "end": best_seg.end,
            "text": best_seg.text,
            "score": best_score,
        }

    # --- Classification ---

    def classify_video(self, video_id: str) -> list[str]:
        """Classify or re-classify a video using LLM.

        Raises:
            VideoNotFoundError: If the video is not in the library.
            LLMError: If classification fails.
        """
        video = self.get_info(video_id)
        if not self._llm or not self._llm.available:
            raise RuntimeError("Classification requires an LLM. Set an API key.")
        tags = self._llm.classify(video.title, video.description, video.channel)
        video.tags = tags
        self._repo.save(video)
        return tags

    # --- Reports ---

    def generate_report(
        self, video_id: str, query: str | None = None, fmt: str = "markdown"
    ) -> tuple[Report, str]:
        """Generate an illustrated report for a single video.

        Raises:
            VideoNotFoundError: If video not in library.
            RuntimeError: If no LLM configured.
        """
        if not self._report_builder:
            raise RuntimeError("Report generation requires an LLM. Set an API key.")
        video = self.get_info(video_id)
        report = self._report_builder.generate_single(video, query=query)
        rendered = (
            self._report_builder.to_html(report)
            if fmt == "html"
            else self._report_builder.to_markdown(report)
        )
        return report, rendered

    def generate_report_from_query(
        self, query: str, tags: list[str] | None = None, fmt: str = "markdown"
    ) -> tuple[Report, str]:
        """Generate an illustrated report across matching library videos.

        Raises:
            RuntimeError: If no LLM or wiki configured.
        """
        if not self._report_builder:
            raise RuntimeError("Report generation requires an LLM. Set an API key.")
        if not self._wiki:
            raise RuntimeError("Query-based reports require a wiki engine.")

        # Find relevant videos via wiki search
        pages = self._wiki.search(query, limit=20)
        video_ids = []
        for page in pages:
            from mcptube.wiki.models import VideoPage

            if isinstance(page, VideoPage):
                video_ids.append(page.video_id)
            elif hasattr(page, "contributions"):
                for c in page.contributions:
                    if c.video_id not in video_ids:
                        video_ids.append(c.video_id)
            elif hasattr(page, "video_references"):
                for r in page.video_references:
                    if r.video_id not in video_ids:
                        video_ids.append(r.video_id)

        from mcptube.wiki.models import VideoPage

        # Collect key frames from wiki VideoPages
        wiki_frames = {}
        for vid in video_ids:
            page = self._wiki.get_page(f"video-{vid}")
            if isinstance(page, VideoPage) and page.key_frames:
                wiki_frames[vid] = page.key_frames

        if not video_ids:
            raise VideoNotFoundError(f"No matching content for: {query}")

        videos = [self.get_info(vid) for vid in video_ids]
        report = self._report_builder.generate_multi(videos, query)
        rendered = (
            self._report_builder.to_html(report)
            if fmt == "html"
            else self._report_builder.to_markdown(report)
        )
        return report, rendered

    # --- Discovery ---

    def discover_videos(self, topic: str) -> DiscoveryResult:
        """Search YouTube for videos on a topic, filter, and cluster.

        Raises:
            RuntimeError: If no LLM configured.
        """
        if not self._discovery:
            raise RuntimeError("Discovery requires an LLM. Set an API key.")
        return self._discovery.discover(topic)

    # --- Synthesis ---

    def synthesize(
        self, video_ids: list[str], topic: str, fmt: str = "markdown"
    ) -> tuple[Report, str]:
        """Cross-reference themes across multiple videos.

        Raises:
            VideoNotFoundError: If any video not found.
            RuntimeError: If no LLM configured.
        """
        if not self._report_builder:
            raise RuntimeError("Synthesis requires an LLM. Set an API key.")
        videos = [self.get_info(vid) for vid in video_ids]
        report = self._report_builder.generate_multi(videos, topic)
        rendered = (
            self._report_builder.to_html(report)
            if fmt == "html"
            else self._report_builder.to_markdown(report)
        )
        return report, rendered

    # --- Q&A ---

    def ask_video(self, video_id: str, question: str) -> str:
        """Ask a question about a single video.

        Raises:
            VideoNotFoundError: If video not in library.
            RuntimeError: If no LLM configured.
        """
        if not self._llm or not self._llm.available:
            raise RuntimeError("Asking questions requires an LLM. Set an API key.")
        video = self.get_info(video_id)
        transcript_text = self._format_transcript(video)
        return self._llm.answer_question(
            question,
            [
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "channel": video.channel,
                    "transcript_text": transcript_text,
                }
            ],
        )

    def ask_videos(self, video_ids: list[str], question: str) -> str:
        """Ask a question across multiple videos.

        Raises:
            VideoNotFoundError: If any video not found.
            RuntimeError: If no LLM configured.
        """
        if not self._llm or not self._llm.available:
            raise RuntimeError("Asking questions requires an LLM. Set an API key.")
        transcripts = []
        for vid in video_ids:
            video = self.get_info(vid)
            transcripts.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "channel": video.channel,
                    "transcript_text": self._format_transcript(video),
                }
            )
        return self._llm.answer_question(question, transcripts)

    # --- Resolution ---

    def resolve_video(self, query: str) -> Video:
        """Smart video resolver — tiered resolution strategy.

        Tier 1: Exact video ID match
        Tier 2: Numeric index from list
        Tier 3: Exact case-insensitive substring match on title/channel

        Raises:
            VideoNotFoundError: If no video can be resolved.
            AmbiguousVideoError: If multiple videos match.
        """
        # Tier 1: Exact video ID
        video = self._repo.get(query)
        if video is not None:
            return video

        # Tier 2: Numeric index
        if query.isdigit():
            videos = self._repo.list_all()
            idx = int(query) - 1
            if 0 <= idx < len(videos):
                return self._repo.get(videos[idx].video_id)
            raise VideoNotFoundError(
                f"Index {query} out of range. Library has {len(videos)} video(s)."
            )

        # Tier 3: Substring match
        videos = self._repo.list_all()
        q = query.lower()
        matches = [v for v in videos if q in v.title.lower() or q in v.channel.lower()]

        if len(matches) == 1:
            return self._repo.get(matches[0].video_id)
        if len(matches) > 1:
            raise AmbiguousVideoError(
                f"Multiple videos match '{query}':\n"
                + "\n".join(f"  {i + 1}. {v.title}" for i, v in enumerate(matches))
            )

        raise VideoNotFoundError(f"No video matching: {query}")

    @staticmethod
    def _format_transcript(video) -> str:
        """Format transcript segments with timestamps."""
        lines = []
        for seg in video.transcript:
            mins, secs = divmod(int(seg.start), 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {seg.text}")
        return "\n".join(lines)

    def cleanup_video_files(self, video_id: str) -> None:
        """Remove downloaded video and frame files for a video.

        Args:
            video_id: The video ID to clean up.
        """
        import shutil
        import glob

        logger.info("Cleaning up files for video: %s", video_id)

        frames_dir = settings.frames_dir
        if not frames_dir or not frames_dir.exists():
            return

        removed_count = 0

        frame_pattern = f"{video_id}_*.jpg"
        for frame_path in glob.glob(str(frames_dir / frame_pattern)):
            try:
                Path(frame_path).unlink(missing_ok=True)
                removed_count += 1
                logger.debug("Removed frame: %s", frame_path)
            except OSError as e:
                logger.warning("Failed to remove frame %s: %s", frame_path, e)

        scene_dir = frames_dir / f"{video_id}_scenes"
        if scene_dir.exists():
            try:
                shutil.rmtree(scene_dir)
                removed_count += 1
                logger.debug("Removed scene directory: %s", scene_dir)
            except OSError as e:
                logger.warning("Failed to remove scene directory %s: %s", scene_dir, e)

        logger.info("Cleanup complete: removed %d items for video %s", removed_count, video_id)
