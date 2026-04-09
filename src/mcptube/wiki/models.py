"""Wiki page models for mcptube-vision knowledge base."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class WikiPageType(str, Enum):
    """Types of wiki pages in the knowledge base."""
    VIDEO = "video"
    ENTITY = "entity"
    TOPIC = "topic"
    CONCEPT = "concept"


class EntityCategory(str, Enum):
    """Categories for entity pages."""
    PERSON = "person"
    COMPANY = "company"
    TOOL = "tool"
    PLACE = "place"
    OTHER = "other"


class FrameDescription(BaseModel):
    """A key frame extracted and described by the vision model."""
    filename: str
    timestamp: float
    description: str


class VideoContribution(BaseModel):
    """What a specific video said about a topic/concept — immutable once written."""
    video_id: str
    title: str
    channel: str
    content: str
    timestamps: list[str] = Field(default_factory=list)
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WikiPageBase(BaseModel):
    """Base model for all wiki pages."""
    slug: str  # URL/filename-safe identifier (e.g. "transformer-architecture")
    page_type: WikiPageType
    title: str
    tags: list[str] = Field(default_factory=list)
    related_pages: list[str] = Field(default_factory=list)  # slugs of related pages
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VideoPage(WikiPageBase):
    """Write-once page for an ingested video."""
    page_type: WikiPageType = WikiPageType.VIDEO
    video_id: str
    channel: str = ""
    duration: float = 0.0
    processing_tier: str = "full_analysis"  # "text_only" or "full_analysis"
    summary: str = ""
    key_timestamps: dict[str, str] = Field(default_factory=dict)  # {"00:30": "Introduction"}
    key_frames: list[FrameDescription] = Field(default_factory=list)
    transcript: str = ""  # full transcript — immutable


class EntityPage(WikiPageBase):
    """Append-only page for named entities (people, companies, tools, etc.)."""
    page_type: WikiPageType = WikiPageType.ENTITY
    category: EntityCategory = EntityCategory.OTHER
    overview: str = ""  # LLM synthesis — updated when new videos reference this entity
    video_references: list[VideoContribution] = Field(default_factory=list)


class TopicPage(WikiPageBase):
    """Topic page — synthesis rewritten, per-video contributions immutable."""
    page_type: WikiPageType = WikiPageType.TOPIC
    synthesis: str = ""  # LLM-generated overview — rewritten on new video ingest
    contributions: list[VideoContribution] = Field(default_factory=list)


class ConceptPage(WikiPageBase):
    """Concept page — synthesis rewritten, per-video contributions immutable."""
    page_type: WikiPageType = WikiPageType.CONCEPT
    synthesis: str = ""  # LLM-generated overview — rewritten on new video ingest
    contributions: list[VideoContribution] = Field(default_factory=list)
