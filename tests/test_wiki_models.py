"""Tests for wiki page models."""

import pytest
from datetime import datetime, timezone

from mcptube.wiki.models import (
    ConceptPage,
    EntityCategory,
    EntityPage,
    FrameDescription,
    TopicPage,
    VideoContribution,
    VideoPage,
    WikiPageType,
)


class TestWikiPageType:
    def test_enum_values(self):
        assert WikiPageType.VIDEO == "video"
        assert WikiPageType.ENTITY == "entity"
        assert WikiPageType.TOPIC == "topic"
        assert WikiPageType.CONCEPT == "concept"


class TestEntityCategory:
    def test_enum_values(self):
        assert EntityCategory.PERSON == "person"
        assert EntityCategory.COMPANY == "company"
        assert EntityCategory.TOOL == "tool"
        assert EntityCategory.PLACE == "place"
        assert EntityCategory.OTHER == "other"


class TestFrameDescription:
    def test_creation(self):
        f = FrameDescription(filename="scene_0001.jpg", timestamp=12.5, description="A slide about AI")
        assert f.filename == "scene_0001.jpg"
        assert f.timestamp == 12.5
        assert f.description == "A slide about AI"


class TestVideoContribution:
    def test_creation(self):
        c = VideoContribution(
            video_id="abc123",
            title="Test Video",
            channel="TestChannel",
            content="What this video says",
            timestamps=["01:30", "05:45"],
        )
        assert c.video_id == "abc123"
        assert c.timestamps == ["01:30", "05:45"]
        assert isinstance(c.added_at, datetime)

    def test_defaults(self):
        c = VideoContribution(video_id="x", title="t", channel="c", content="text")
        assert c.timestamps == []
        assert c.added_at is not None


class TestVideoPage:
    def test_creation(self):
        p = VideoPage(
            slug="video-abc123",
            title="Test Video",
            video_id="abc123",
            channel="TestChannel",
            duration=600.0,
            summary="A test video about AI",
            transcript="[00:00] Hello world",
        )
        assert p.page_type == WikiPageType.VIDEO
        assert p.video_id == "abc123"
        assert p.processing_tier == "full_analysis"
        assert p.key_frames == []
        assert p.key_timestamps == {}

    def test_text_only_tier(self):
        p = VideoPage(
            slug="video-abc123",
            title="Test",
            video_id="abc123",
            processing_tier="text_only",
        )
        assert p.processing_tier == "text_only"

    def test_with_frames(self):
        frames = [
            FrameDescription(filename="f1.jpg", timestamp=10.0, description="Slide 1"),
            FrameDescription(filename="f2.jpg", timestamp=30.0, description="Slide 2"),
        ]
        p = VideoPage(slug="video-x", title="T", video_id="x", key_frames=frames)
        assert len(p.key_frames) == 2


class TestEntityPage:
    def test_creation(self):
        p = EntityPage(
            slug="entity-openai",
            title="OpenAI",
            category=EntityCategory.COMPANY,
            overview="An AI research company.",
        )
        assert p.page_type == WikiPageType.ENTITY
        assert p.category == EntityCategory.COMPANY
        assert p.video_references == []

    def test_with_references(self):
        ref = VideoContribution(
            video_id="abc", title="V1", channel="C", content="Mentioned here"
        )
        p = EntityPage(
            slug="entity-openai",
            title="OpenAI",
            category=EntityCategory.COMPANY,
            overview="AI company",
            video_references=[ref],
        )
        assert len(p.video_references) == 1
        assert p.video_references[0].video_id == "abc"


class TestTopicPage:
    def test_creation(self):
        p = TopicPage(
            slug="topic-machine-learning",
            title="Machine Learning",
            synthesis="ML is a subfield of AI.",
            tags=["AI", "ML"],
        )
        assert p.page_type == WikiPageType.TOPIC
        assert p.contributions == []
        assert "ML" in p.tags

    def test_with_contributions(self):
        c = VideoContribution(
            video_id="v1", title="Video 1", channel="C",
            content="This video discusses ML basics.",
        )
        p = TopicPage(
            slug="topic-ml", title="ML",
            synthesis="Overview", contributions=[c],
        )
        assert len(p.contributions) == 1


class TestConceptPage:
    def test_creation(self):
        p = ConceptPage(
            slug="concept-scaling-laws",
            title="Scaling Laws",
            synthesis="Scaling laws describe how model performance improves with size.",
            tags=["AI", "scaling"],
        )
        assert p.page_type == WikiPageType.CONCEPT
        assert p.contributions == []

    def test_with_multiple_contributions(self):
        c1 = VideoContribution(video_id="v1", title="V1", channel="C", content="First view")
        c2 = VideoContribution(video_id="v2", title="V2", channel="C", content="Second view")
        p = ConceptPage(
            slug="concept-x", title="X",
            synthesis="Combined", contributions=[c1, c2],
        )
        assert len(p.contributions) == 2


class TestWikiPageBase:
    def test_related_pages(self):
        p = VideoPage(
            slug="video-x", title="X", video_id="x",
            related_pages=["entity-openai", "topic-ml"],
        )
        assert len(p.related_pages) == 2

    def test_tags(self):
        p = TopicPage(slug="topic-x", title="X", tags=["a", "b", "c"])
        assert p.tags == ["a", "b", "c"]

    def test_timestamps_auto_set(self):
        p = EntityPage(slug="entity-x", title="X")
        assert isinstance(p.created_at, datetime)
        assert isinstance(p.updated_at, datetime)

    def test_serialization_roundtrip(self):
        p = ConceptPage(
            slug="concept-test",
            title="Test Concept",
            synthesis="A test synthesis.",
            tags=["test"],
            contributions=[
                VideoContribution(video_id="v1", title="V1", channel="C", content="Content"),
            ],
        )
        json_str = p.model_dump_json()
        restored = ConceptPage.model_validate_json(json_str)
        assert restored.slug == p.slug
        assert restored.synthesis == p.synthesis
        assert len(restored.contributions) == 1
