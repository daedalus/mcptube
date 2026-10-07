"""Domain models for mcptube."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field, computed_field


class TranscriptSegment(BaseModel):
    """A single caption entry from the video transcript."""

    start: float  # start time in seconds
    duration: float  # duration in seconds
    text: str

    @computed_field
    @property
    def end(self) -> float:
        """End time in seconds."""
        return self.start + self.duration


class Chapter(BaseModel):
    """A chapter marker from the video."""

    title: str
    start: float  # start time in seconds


class Video(BaseModel):
    """Core domain entity representing an indexed video."""

    video_id: str  # namespaced ID: "{platform}_{native_id}" or YouTube ID
    platform: str = "youtube"  # source platform (youtube, tiktok, instagram, etc.)
    source_url: str = ""  # canonical URL from the platform
    title: str
    description: str = ""
    channel: str = ""
    duration: float = 0.0  # total duration in seconds
    thumbnail_url: str = ""
    chapters: list[Chapter] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_stats: dict = Field(
        default_factory=dict
    )  # {"ffmpeg_extracted": int, "llm_processed": int}
    format: str = ""  # video format (e.g., "1080p", "4K")
    file_size: int = 0  # total size in bytes
    width: int = 0  # video width in pixels
    height: int = 0  # video height in pixels
    vcodec: str = ""  # video codec (e.g., "avc1")
    acodec: str = ""  # audio codec (e.g., "mp4a")
    wiki_processed: bool = False  # whether wiki pages were created/updated

    @computed_field
    @property
    def url(self) -> str:
        """Canonical URL for this video."""
        if self.source_url:
            return self.source_url
        return f"https://www.youtube.com/watch?v={self.video_id}"
