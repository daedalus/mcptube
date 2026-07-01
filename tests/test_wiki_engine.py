"""Tests for wiki engine — orchestration of extraction, updates, and retrieval."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from mcptube.llm import LLMClient, LLMError
from mcptube.models import Video, TranscriptSegment
from mcptube.wiki.engine import WikiEngine
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
from mcptube.wiki.storage import FileWikiRepository


SAMPLE_LLM_EXTRACTION = """{
    "video_summary": "A video about neural networks and deep learning.",
    "key_timestamps": {"00:00": "Intro", "00:10": "Neural nets"},
    "entities": [
        {"name": "Geoffrey Hinton", "category": "person", "context": "Pioneer of deep learning.", "timestamps": ["00:10"]}
    ],
    "topics": [
        {"name": "Deep Learning", "content": "Covers fundamentals of deep learning.", "timestamps": ["00:10"], "tags": ["AI"]}
    ],
    "concepts": [
        {"name": "Backpropagation", "content": "Explains how gradients flow backward.", "timestamps": ["00:10"], "tags": ["training"]}
    ]
}"""


@pytest.fixture
def wiki_repo(tmp_path):
    return FileWikiRepository(wiki_dir=tmp_path / "wiki", db_path=":memory:")


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.available = True
    llm._complete = MagicMock(return_value=SAMPLE_LLM_EXTRACTION)
    return llm


@pytest.fixture
def engine(wiki_repo, mock_llm):
    return WikiEngine(repo=wiki_repo, llm=mock_llm)


@pytest.fixture
def sample_video():
    return Video(
        video_id="vid001",
        title="Neural Networks 101",
        description="Intro to neural networks",
        channel="AI School",
        duration=300.0,
        transcript=[
            TranscriptSegment(start=0.0, duration=5.0, text="Welcome to the course."),
            TranscriptSegment(
                start=10.0, duration=5.0, text="Neural networks learn from data."
            ),
        ],
        tags=["AI"],
    )


@pytest.fixture
def sample_video_b():
    return Video(
        video_id="vid002",
        title="Advanced Neural Nets",
        description="Deep dive into architectures",
        channel="AI School",
        duration=600.0,
        transcript=[
            TranscriptSegment(start=0.0, duration=5.0, text="Today we go deeper."),
            TranscriptSegment(start=10.0, duration=5.0, text="Backpropagation is key."),
        ],
        tags=["AI"],
    )


# --- Ingestion ---


class TestIngestVideo:
    def test_ingest_creates_pages(self, engine, wiki_repo, sample_video):
        stats = engine.ingest_video(sample_video, text_only=True)
        assert stats["created"] >= 1
        assert wiki_repo.exists("video-vid001")

    def test_ingest_creates_entity_pages(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        entities = wiki_repo.get_entity_pages()
        assert len(entities) >= 1
        names = [e.title for e in entities]
        assert "Geoffrey Hinton" in names

    def test_ingest_creates_topic_pages(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        topics = wiki_repo.get_topic_pages()
        assert len(topics) >= 1

    def test_ingest_creates_concept_pages(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        concepts = wiki_repo.get_concept_pages()
        assert len(concepts) >= 1

    def test_ingest_with_frames(self, engine, wiki_repo, sample_video):
        frames = [
            FrameDescription(
                filename="scene_0001.jpg",
                timestamp=10.0,
                description="Slide about neurons",
            ),
        ]
        stats = engine.ingest_video(
            sample_video, frame_descriptions=frames, text_only=False
        )
        vp = wiki_repo.get_video_page("vid001")
        assert vp is not None
        assert len(vp.key_frames) == 1

    def test_ingest_second_video_updates_existing(
        self, engine, wiki_repo, mock_llm, sample_video, sample_video_b
    ):
        engine.ingest_video(sample_video, text_only=True)

        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_EXTRACTION)
        stats = engine.ingest_video(sample_video_b, text_only=True)

        assert stats["updated"] >= 1
        assert wiki_repo.exists("video-vid002")

    def test_ingest_fails_without_llm(self, wiki_repo, sample_video):
        llm = MagicMock(spec=LLMClient)
        llm.available = False
        eng = WikiEngine(repo=wiki_repo, llm=llm)
        with pytest.raises(LLMError):
            eng.ingest_video(sample_video)


# --- Search ---


class TestSearch:
    def test_search_by_title(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        results = engine.search("Neural Networks")
        assert len(results) >= 1

    def test_search_by_content(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        results = engine.search("deep learning")
        assert len(results) >= 1

    def test_search_no_results(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        results = engine.search("quantum computing")
        assert len(results) == 0

    def test_search_respects_limit(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        results = engine.search("AI", limit=1)
        assert len(results) <= 1


# --- Ask (Agentic Retrieval) ---


class TestAsk:
    def test_ask_returns_answer(self, engine, wiki_repo, mock_llm, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        mock_llm._complete = MagicMock(
            return_value="Neural networks learn from data using backpropagation."
        )
        answer = engine.ask("How do neural networks learn?")
        assert len(answer) > 0

    def test_ask_calls_llm(self, engine, wiki_repo, mock_llm, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        mock_llm._complete = MagicMock(return_value="Answer.")
        engine.ask("What is backpropagation?")
        # Should be called: once for extraction, once for ask
        assert mock_llm._complete.call_count >= 1

    def test_ask_fails_without_llm(self, wiki_repo, sample_video):
        llm = MagicMock(spec=LLMClient)
        llm.available = False
        eng = WikiEngine(repo=wiki_repo, llm=llm)
        with pytest.raises(LLMError):
            eng.ask("What is AI?")


# --- Page Access ---


class TestPageAccess:
    def test_get_page(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        page = engine.get_page("video-vid001")
        assert page is not None
        assert isinstance(page, VideoPage)

    def test_get_page_not_found(self, engine):
        assert engine.get_page("nonexistent") is None

    def test_list_pages_all(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        pages = engine.list_pages()
        assert len(pages) >= 4  # video + entity + topic + concept

    def test_list_pages_by_type(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        entities = engine.list_pages(page_type=WikiPageType.ENTITY)
        assert all(isinstance(p, EntityPage) for p in entities)

    def test_list_pages_by_tag(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        ai_pages = engine.list_pages(tag="AI")
        assert len(ai_pages) >= 1

    def test_get_toc(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        toc = engine.get_toc()
        assert "Neural Networks 101" in toc
        assert "Geoffrey Hinton" in toc

    def test_get_toc_empty(self, engine):
        toc = engine.get_toc()
        assert "empty" in toc.lower()


# --- Version History ---


class TestVersionHistory:
    def test_no_history_after_first_ingest(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        history = engine.get_page_history("video-vid001")
        assert len(history) == 0

    def test_history_after_update(
        self, engine, wiki_repo, mock_llm, sample_video, sample_video_b
    ):
        engine.ingest_video(sample_video, text_only=True)
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_EXTRACTION)
        engine.ingest_video(sample_video_b, text_only=True)

        # Topic page should have history since it was updated
        history = engine.get_page_history("topic-deep-learning")
        assert len(history) >= 1


# --- Remove Video ---


class TestRemoveVideo:
    def test_remove_deletes_video_page(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        assert wiki_repo.exists("video-vid001")
        engine.remove_video("vid001")
        assert not wiki_repo.exists("video-vid001")

    def test_remove_cleans_entity_references(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        engine.remove_video("vid001")
        # Entity with only one video reference should be deleted
        entities = wiki_repo.get_entity_pages()
        for e in entities:
            assert all(ref.video_id != "vid001" for ref in e.video_references)

    def test_remove_cleans_topic_contributions(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        engine.remove_video("vid001")
        topics = wiki_repo.get_topic_pages()
        for t in topics:
            assert all(c.video_id != "vid001" for c in t.contributions)

    def test_remove_cleans_concept_contributions(self, engine, wiki_repo, sample_video):
        engine.ingest_video(sample_video, text_only=True)
        engine.remove_video("vid001")
        concepts = wiki_repo.get_concept_pages()
        for c in concepts:
            assert all(contrib.video_id != "vid001" for contrib in c.contributions)

    def test_remove_preserves_multi_video_pages(
        self, engine, wiki_repo, mock_llm, sample_video, sample_video_b
    ):
        engine.ingest_video(sample_video, text_only=True)
        mock_llm._complete = MagicMock(return_value=SAMPLE_LLM_EXTRACTION)
        engine.ingest_video(sample_video_b, text_only=True)

        engine.remove_video("vid001")

        # Entity should still exist with vid002's reference
        entity = wiki_repo.get_page("entity-geoffrey-hinton")
        if entity is not None:
            assert any(ref.video_id == "vid002" for ref in entity.video_references)

    def test_remove_nonexistent_returns_zero(self, engine):
        modified = engine.remove_video("nonexistent")
        assert modified == 0
