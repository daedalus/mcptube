"""Tests for wiki knowledge extractor — LLM output parsing and page building."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from mcptube.models import Video, TranscriptSegment
from mcptube.llm import LLMClient, LLMError
from mcptube.wiki.extractor import KnowledgeExtractor, _slugify
from mcptube.wiki.models import (
    ConceptPage,
    EntityCategory,
    EntityPage,
    FrameDescription,
    TopicPage,
    VideoPage,
    WikiPageType,
)


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.available = True
    return llm


@pytest.fixture
def extractor(mock_llm):
    return KnowledgeExtractor(mock_llm)


@pytest.fixture
def sample_video():
    return Video(
        video_id="abc123",
        title="Intro to Transformers",
        description="A deep dive into transformer architecture",
        channel="AI Academy",
        duration=600.0,
        transcript=[
            TranscriptSegment(start=0.0, duration=5.0, text="Hello and welcome."),
            TranscriptSegment(
                start=5.0, duration=10.0, text="Today we discuss transformers."
            ),
            TranscriptSegment(
                start=15.0, duration=10.0, text="Attention is all you need."
            ),
        ],
        tags=["AI", "transformers"],
    )


SAMPLE_LLM_RESPONSE = """{
    "video_summary": "An introduction to transformer architecture and self-attention mechanisms.",
    "key_timestamps": {
        "00:00": "Introduction",
        "00:05": "Transformer overview",
        "00:15": "Attention mechanism"
    },
    "entities": [
        {
            "name": "Google",
            "category": "company",
            "context": "Google researchers published the original transformer paper.",
            "timestamps": ["00:05"]
        },
        {
            "name": "Ashish Vaswani",
            "category": "person",
            "context": "Lead author of Attention Is All You Need.",
            "timestamps": ["00:15"]
        }
    ],
    "topics": [
        {
            "name": "Neural Network Architecture",
            "content": "The video covers how transformers replaced RNNs as the dominant architecture.",
            "timestamps": ["00:05"],
            "tags": ["deep-learning", "architecture"]
        }
    ],
    "concepts": [
        {
            "name": "Self-Attention",
            "content": "Self-attention allows tokens to attend to all other tokens in a sequence.",
            "timestamps": ["00:15"],
            "tags": ["attention", "transformers"]
        },
        {
            "name": "Positional Encoding",
            "content": "Since transformers have no recurrence, positional encodings inject order information.",
            "timestamps": ["00:15"],
            "tags": ["transformers"]
        }
    ]
}"""


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert _slugify("OpenAI's GPT-4!") == "openais-gpt-4"

    def test_multiple_spaces(self):
        assert _slugify("too   many   spaces") == "too-many-spaces"

    def test_leading_trailing(self):
        assert _slugify("  trimmed  ") == "trimmed"

    def test_unicode(self):
        assert _slugify("café résumé") == "café-résumé"


class TestExtractParsing:
    def test_successful_extraction(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        result = extractor.extract(sample_video, text_only=True)

        assert "video_page" in result
        assert "entity_pages" in result
        assert "topic_pages" in result
        assert "concept_pages" in result

    def test_video_page_fields(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        result = extractor.extract(sample_video, text_only=True)
        vp = result["video_page"]

        assert isinstance(vp, VideoPage)
        assert vp.video_id == "abc123"
        assert vp.slug == "video-abc123"
        assert vp.title == "Intro to Transformers"
        assert vp.channel == "AI Academy"
        assert vp.processing_tier == "text_only"
        assert "transformer" in vp.summary.lower()
        assert len(vp.key_timestamps) == 3

    def test_entity_pages(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        result = extractor.extract(sample_video, text_only=True)
        entities = result["entity_pages"]

        assert len(entities) == 2
        google = next(e for e in entities if e.title == "Google")
        assert isinstance(google, EntityPage)
        assert google.category == EntityCategory.COMPANY
        assert google.slug == "entity-google"
        assert len(google.video_references) == 1
        assert google.video_references[0].video_id == "abc123"

    def test_topic_pages(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        result = extractor.extract(sample_video, text_only=True)
        topics = result["topic_pages"]

        assert len(topics) == 1
        assert isinstance(topics[0], TopicPage)
        assert topics[0].title == "Neural Network Architecture"
        assert len(topics[0].contributions) == 1
        assert "deep-learning" in topics[0].tags

    def test_concept_pages(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        result = extractor.extract(sample_video, text_only=True)
        concepts = result["concept_pages"]

        assert len(concepts) == 2
        attention = next(c for c in concepts if c.title == "Self-Attention")
        assert isinstance(attention, ConceptPage)
        assert "attention" in attention.tags
        assert len(attention.contributions) == 1


class TestExtractRelatedPages:
    def test_all_pages_link_to_video(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        result = extractor.extract(sample_video, text_only=True)

        video_slug = result["video_page"].slug
        for page in (
            result["entity_pages"] + result["topic_pages"] + result["concept_pages"]
        ):
            assert video_slug in page.related_pages


class TestExtractWithFrames:
    def test_full_analysis_tier(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        frames = [
            FrameDescription(
                filename="scene_0001.jpg", timestamp=5.0, description="Title slide"
            ),
        ]
        result = extractor.extract(
            sample_video, frame_descriptions=frames, text_only=False
        )
        vp = result["video_page"]

        assert vp.processing_tier == "full_analysis"
        assert len(vp.key_frames) == 1
        assert vp.key_frames[0].description == "Title slide"

    def test_text_only_ignores_frames(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_RESPONSE)
        frames = [
            FrameDescription(
                filename="scene_0001.jpg", timestamp=5.0, description="Slide"
            ),
        ]
        result = extractor.extract(
            sample_video, frame_descriptions=frames, text_only=True
        )
        vp = result["video_page"]

        assert vp.processing_tier == "text_only"
        assert len(vp.key_frames) == 0


class TestExtractErrorHandling:
    def test_llm_unavailable(self, mock_llm, sample_video):
        mock_llm.available = False
        extractor = KnowledgeExtractor(mock_llm)
        with pytest.raises(LLMError):
            extractor.extract(sample_video)

    def test_invalid_json_response(self, extractor, mock_llm, sample_video):
        mock_llm._complete = MagicMock(return_value="this is not json")
        with pytest.raises(LLMError, match="Failed to parse"):
            extractor.extract(sample_video)

    def test_json_with_markdown_fences(self, extractor, mock_llm, sample_video):
        wrapped = f"```json\n{SAMPLE_LLM_RESPONSE}\n```"
        mock_llm._complete = MagicMock(return_value=wrapped)
        result = extractor.extract(sample_video, text_only=True)
        assert result["video_page"] is not None

    def test_empty_entities(self, extractor, mock_llm, sample_video):
        response = '{"video_summary": "Summary", "key_timestamps": {}, "entities": [], "topics": [], "concepts": []}'
        mock_llm._complete = MagicMock(return_value=response)
        result = extractor.extract(sample_video, text_only=True)
        assert result["video_page"] is not None
        assert result["entity_pages"] == []
        assert result["topic_pages"] == []
        assert result["concept_pages"] == []

    def test_invalid_entity_category_defaults_to_other(
        self, extractor, mock_llm, sample_video
    ):
        response = """{
            "video_summary": "Summary",
            "key_timestamps": {},
            "entities": [{"name": "Foo", "category": "invalid_category", "context": "Bar", "timestamps": []}],
            "topics": [],
            "concepts": []
        }"""
        mock_llm._complete = MagicMock(return_value=response)
        result = extractor.extract(sample_video, text_only=True)
        assert result["entity_pages"][0].category == EntityCategory.OTHER
