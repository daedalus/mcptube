"""Tests for wiki file storage + SQLite FTS5 index."""

import pytest
from pathlib import Path
from datetime import datetime, timezone

from mcptube.wiki.models import (
    ConceptPage,
    EntityCategory,
    EntityPage,
    TopicPage,
    VideoContribution,
    VideoPage,
    WikiPageType,
)
from mcptube.wiki.storage import FileWikiRepository


@pytest.fixture
def wiki_repo(tmp_path):
    """Create a FileWikiRepository with temp directory and in-memory DB."""
    return FileWikiRepository(
        wiki_dir=tmp_path / "wiki",
        db_path=":memory:",
    )


@pytest.fixture
def sample_video_page():
    return VideoPage(
        slug="video-abc123",
        title="Test Video",
        video_id="abc123",
        channel="TestChannel",
        duration=600.0,
        summary="A video about machine learning basics.",
        transcript="[00:00] Hello and welcome to this tutorial on machine learning.",
        tags=["AI", "tutorial"],
    )


@pytest.fixture
def sample_entity_page():
    return EntityPage(
        slug="entity-openai",
        title="OpenAI",
        category=EntityCategory.COMPANY,
        overview="An AI research company known for GPT models.",
        video_references=[
            VideoContribution(
                video_id="abc123",
                title="Test Video",
                channel="TestChannel",
                content="OpenAI is mentioned as a leader in AI research.",
                timestamps=["02:30"],
            )
        ],
        tags=["company", "AI"],
    )


@pytest.fixture
def sample_topic_page():
    return TopicPage(
        slug="topic-machine-learning",
        title="Machine Learning",
        synthesis="Machine learning is a subfield of AI focused on learning from data.",
        contributions=[
            VideoContribution(
                video_id="abc123",
                title="Test Video",
                channel="TestChannel",
                content="The video covers supervised and unsupervised learning.",
                timestamps=["05:00", "10:30"],
            )
        ],
        tags=["AI", "ML"],
    )


@pytest.fixture
def sample_concept_page():
    return ConceptPage(
        slug="concept-scaling-laws",
        title="Scaling Laws",
        synthesis="Performance improves predictably with compute and data scale.",
        contributions=[
            VideoContribution(
                video_id="abc123",
                title="Test Video",
                channel="TestChannel",
                content="Discusses how scaling laws guide model training decisions.",
                timestamps=["15:00"],
            )
        ],
        tags=["AI", "scaling"],
    )


class TestSavePage:
    def test_save_and_retrieve_video_page(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        retrieved = wiki_repo.get_page("video-abc123")
        assert retrieved is not None
        assert isinstance(retrieved, VideoPage)
        assert retrieved.video_id == "abc123"
        assert retrieved.summary == sample_video_page.summary

    def test_save_and_retrieve_entity_page(self, wiki_repo, sample_entity_page):
        wiki_repo.save_page(sample_entity_page)
        retrieved = wiki_repo.get_page("entity-openai")
        assert isinstance(retrieved, EntityPage)
        assert retrieved.category == EntityCategory.COMPANY
        assert len(retrieved.video_references) == 1

    def test_save_and_retrieve_topic_page(self, wiki_repo, sample_topic_page):
        wiki_repo.save_page(sample_topic_page)
        retrieved = wiki_repo.get_page("topic-machine-learning")
        assert isinstance(retrieved, TopicPage)
        assert len(retrieved.contributions) == 1

    def test_save_and_retrieve_concept_page(self, wiki_repo, sample_concept_page):
        wiki_repo.save_page(sample_concept_page)
        retrieved = wiki_repo.get_page("concept-scaling-laws")
        assert isinstance(retrieved, ConceptPage)
        assert "scaling" in retrieved.tags

    def test_upsert_overwrites(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        sample_video_page.summary = "Updated summary"
        wiki_repo.save_page(sample_video_page)
        retrieved = wiki_repo.get_page("video-abc123")
        assert retrieved.summary == "Updated summary"


class TestGetPage:
    def test_get_nonexistent_returns_none(self, wiki_repo):
        assert wiki_repo.get_page("nonexistent-slug") is None


class TestDeletePage:
    def test_delete_existing(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        wiki_repo.delete_page("video-abc123")
        assert wiki_repo.get_page("video-abc123") is None

    def test_delete_nonexistent_no_error(self, wiki_repo):
        wiki_repo.delete_page("nonexistent-slug")  # should not raise


class TestExists:
    def test_exists_true(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        assert wiki_repo.exists("video-abc123") is True

    def test_exists_false(self, wiki_repo):
        assert wiki_repo.exists("nonexistent") is False


class TestListPages:
    def test_list_all(
        self, wiki_repo, sample_video_page, sample_entity_page, sample_topic_page
    ):
        wiki_repo.save_page(sample_video_page)
        wiki_repo.save_page(sample_entity_page)
        wiki_repo.save_page(sample_topic_page)
        pages = wiki_repo.list_pages()
        assert len(pages) == 3

    def test_list_by_type(self, wiki_repo, sample_video_page, sample_entity_page):
        wiki_repo.save_page(sample_video_page)
        wiki_repo.save_page(sample_entity_page)
        videos = wiki_repo.list_pages(page_type=WikiPageType.VIDEO)
        assert len(videos) == 1
        assert isinstance(videos[0], VideoPage)

    def test_list_by_tag(self, wiki_repo, sample_video_page, sample_entity_page):
        wiki_repo.save_page(sample_video_page)
        wiki_repo.save_page(sample_entity_page)
        ai_pages = wiki_repo.list_pages(tag="AI")
        assert len(ai_pages) == 2

    def test_list_empty(self, wiki_repo):
        assert wiki_repo.list_pages() == []


class TestSearch:
    def test_search_by_title(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        results = wiki_repo.search("Test Video")
        assert len(results) >= 1
        assert results[0].slug == "video-abc123"

    def test_search_by_content(self, wiki_repo, sample_topic_page):
        wiki_repo.save_page(sample_topic_page)
        results = wiki_repo.search("supervised unsupervised learning")
        assert len(results) >= 1

    def test_search_by_tag_word(self, wiki_repo, sample_entity_page):
        wiki_repo.save_page(sample_entity_page)
        results = wiki_repo.search("company")
        assert len(results) >= 1

    def test_search_no_results(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        results = wiki_repo.search("quantum computing")
        assert len(results) == 0

    def test_search_limit(
        self, wiki_repo, sample_video_page, sample_topic_page, sample_entity_page
    ):
        wiki_repo.save_page(sample_video_page)
        wiki_repo.save_page(sample_topic_page)
        wiki_repo.save_page(sample_entity_page)
        results = wiki_repo.search("AI", limit=1)
        assert len(results) <= 1


class TestTypedGetters:
    def test_get_video_page(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        vp = wiki_repo.get_video_page("abc123")
        assert vp is not None
        assert vp.video_id == "abc123"

    def test_get_video_page_not_found(self, wiki_repo):
        assert wiki_repo.get_video_page("nonexistent") is None

    def test_get_entity_pages(self, wiki_repo, sample_entity_page):
        wiki_repo.save_page(sample_entity_page)
        pages = wiki_repo.get_entity_pages()
        assert len(pages) == 1
        assert isinstance(pages[0], EntityPage)

    def test_get_topic_pages(self, wiki_repo, sample_topic_page):
        wiki_repo.save_page(sample_topic_page)
        pages = wiki_repo.get_topic_pages()
        assert len(pages) == 1

    def test_get_concept_pages(self, wiki_repo, sample_concept_page):
        wiki_repo.save_page(sample_concept_page)
        pages = wiki_repo.get_concept_pages()
        assert len(pages) == 1


class TestTOC:
    def test_toc_empty(self, wiki_repo):
        assert "empty" in wiki_repo.get_toc().lower()

    def test_toc_with_pages(self, wiki_repo, sample_video_page, sample_entity_page):
        wiki_repo.save_page(sample_video_page)
        wiki_repo.save_page(sample_entity_page)
        toc = wiki_repo.get_toc()
        assert "Test Video" in toc
        assert "OpenAI" in toc
        assert "video-abc123" in toc


class TestVersionHistory:
    def test_no_history_for_new_page(self, wiki_repo, sample_video_page):
        wiki_repo.save_page(sample_video_page)
        history = wiki_repo.get_page_history("video-abc123")
        assert len(history) == 0

    def test_history_saved_on_update(self, wiki_repo, sample_topic_page):
        wiki_repo.save_page(sample_topic_page)
        sample_topic_page.synthesis = "Updated synthesis"
        wiki_repo.save_page(sample_topic_page)
        history = wiki_repo.get_page_history("topic-machine-learning")
        assert len(history) == 1

    def test_multiple_versions(self, wiki_repo, sample_entity_page):
        wiki_repo.save_page(sample_entity_page)
        sample_entity_page.overview = "Update 1"
        wiki_repo.save_page(sample_entity_page)
        sample_entity_page.overview = "Update 2"
        wiki_repo.save_page(sample_entity_page)
        history = wiki_repo.get_page_history("entity-openai")
        assert len(history) == 2


class TestFileStructure:
    def test_directories_created(self, tmp_path):
        repo = FileWikiRepository(wiki_dir=tmp_path / "wiki", db_path=":memory:")
        assert (tmp_path / "wiki" / "video").is_dir()
        assert (tmp_path / "wiki" / "entity").is_dir()
        assert (tmp_path / "wiki" / "topic").is_dir()
        assert (tmp_path / "wiki" / "concept").is_dir()
        assert (tmp_path / "wiki" / "_history").is_dir()

    def test_json_file_written(self, wiki_repo, sample_video_page, tmp_path):
        wiki_repo.save_page(sample_video_page)
        json_path = tmp_path / "wiki" / "video" / "video-abc123.json"
        assert json_path.exists()
        content = json_path.read_text()
        assert "abc123" in content
