"""Tests for wiki updater — merge logic for new extractions into existing wiki."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from mcptube.llm import LLMClient, LLMError
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
from mcptube.wiki.updater import WikiUpdater


@pytest.fixture
def wiki_repo(tmp_path):
    return FileWikiRepository(wiki_dir=tmp_path / "wiki", db_path=":memory:")


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.available = True
    llm._complete = MagicMock(return_value="Updated synthesis combining all perspectives.")
    return llm


@pytest.fixture
def updater(wiki_repo, mock_llm):
    return WikiUpdater(wiki_repo, mock_llm)


@pytest.fixture
def video_page_a():
    return VideoPage(
        slug="video-aaa111",
        title="Video A",
        video_id="aaa111",
        channel="ChannelA",
        summary="Summary of video A.",
    )


@pytest.fixture
def video_page_b():
    return VideoPage(
        slug="video-bbb222",
        title="Video B",
        video_id="bbb222",
        channel="ChannelB",
        summary="Summary of video B.",
    )


@pytest.fixture
def entity_page_from_a():
    return EntityPage(
        slug="entity-openai",
        title="OpenAI",
        category=EntityCategory.COMPANY,
        overview="OpenAI builds GPT models.",
        video_references=[
            VideoContribution(
                video_id="aaa111", title="Video A", channel="ChannelA",
                content="Video A discusses OpenAI's research.", timestamps=["01:00"],
            )
        ],
        tags=["company"],
        related_pages=["video-aaa111"],
    )


@pytest.fixture
def entity_page_from_b():
    return EntityPage(
        slug="entity-openai",
        title="OpenAI",
        category=EntityCategory.COMPANY,
        overview="OpenAI released GPT-4.",
        video_references=[
            VideoContribution(
                video_id="bbb222", title="Video B", channel="ChannelB",
                content="Video B covers GPT-4 launch.", timestamps=["03:00"],
            )
        ],
        tags=["AI"],
        related_pages=["video-bbb222"],
    )


@pytest.fixture
def topic_page_from_a():
    return TopicPage(
        slug="topic-machine-learning",
        title="Machine Learning",
        synthesis="ML is about learning from data.",
        contributions=[
            VideoContribution(
                video_id="aaa111", title="Video A", channel="ChannelA",
                content="Video A covers ML basics.", timestamps=["02:00"],
            )
        ],
        tags=["ML"],
        related_pages=["video-aaa111"],
    )


@pytest.fixture
def topic_page_from_b():
    return TopicPage(
        slug="topic-machine-learning",
        title="Machine Learning",
        synthesis="ML powers modern AI systems.",
        contributions=[
            VideoContribution(
                video_id="bbb222", title="Video B", channel="ChannelB",
                content="Video B covers advanced ML.", timestamps=["04:00"],
            )
        ],
        tags=["AI"],
        related_pages=["video-bbb222"],
    )


@pytest.fixture
def concept_page_from_a():
    return ConceptPage(
        slug="concept-self-attention",
        title="Self-Attention",
        synthesis="Self-attention lets tokens attend to each other.",
        contributions=[
            VideoContribution(
                video_id="aaa111", title="Video A", channel="ChannelA",
                content="Explains the mechanics of self-attention.", timestamps=["05:00"],
            )
        ],
        tags=["transformers"],
        related_pages=["video-aaa111"],
    )


@pytest.fixture
def concept_page_from_b():
    return ConceptPage(
        slug="concept-self-attention",
        title="Self-Attention",
        synthesis="Self-attention is computationally expensive.",
        contributions=[
            VideoContribution(
                video_id="bbb222", title="Video B", channel="ChannelB",
                content="Discusses efficiency tradeoffs in attention.", timestamps=["06:00"],
            )
        ],
        tags=["efficiency"],
        related_pages=["video-bbb222"],
    )


# --- Video Page Tests ---


class TestVideoPageUpdate:
    def test_new_video_page_created(self, updater, wiki_repo, video_page_a):
        extracted = {
            "video_page": video_page_a,
            "entity_pages": [],
            "topic_pages": [],
            "concept_pages": [],
        }
        stats = updater.update_wiki(extracted)
        assert stats["created"] == 1
        assert stats["skipped"] == 0
        assert wiki_repo.exists("video-aaa111")

    def test_duplicate_video_page_skipped(self, updater, wiki_repo, video_page_a):
        wiki_repo.save_page(video_page_a)
        extracted = {
            "video_page": video_page_a,
            "entity_pages": [],
            "topic_pages": [],
            "concept_pages": [],
        }
        stats = updater.update_wiki(extracted)
        assert stats["skipped"] == 1
        assert stats["created"] == 0


# --- Entity Page Tests ---


class TestEntityPageUpdate:
    def test_new_entity_created(self, updater, wiki_repo, entity_page_from_a):
        extracted = {
            "video_page": VideoPage(slug="video-aaa111", title="V", video_id="aaa111"),
            "entity_pages": [entity_page_from_a],
            "topic_pages": [],
            "concept_pages": [],
        }
        stats = updater.update_wiki(extracted)
        assert stats["created"] == 2  # video + entity
        page = wiki_repo.get_page("entity-openai")
        assert isinstance(page, EntityPage)
        assert len(page.video_references) == 1

    def test_entity_appends_new_reference(self, updater, wiki_repo, entity_page_from_a, entity_page_from_b):
        wiki_repo.save_page(entity_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [entity_page_from_b],
            "topic_pages": [],
            "concept_pages": [],
        }
        stats = updater.update_wiki(extracted)
        page = wiki_repo.get_page("entity-openai")
        assert len(page.video_references) == 2
        assert stats["updated"] >= 1

    def test_entity_skips_duplicate_video(self, updater, wiki_repo, entity_page_from_a):
        wiki_repo.save_page(entity_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-aaa111", title="V", video_id="aaa111"),
            "entity_pages": [entity_page_from_a],
            "topic_pages": [],
            "concept_pages": [],
        }
        stats = updater.update_wiki(extracted)
        page = wiki_repo.get_page("entity-openai")
        assert len(page.video_references) == 1  # not duplicated

    def test_entity_merges_tags(self, updater, wiki_repo, entity_page_from_a, entity_page_from_b):
        wiki_repo.save_page(entity_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [entity_page_from_b],
            "topic_pages": [],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("entity-openai")
        assert "company" in page.tags
        assert "AI" in page.tags

    def test_entity_merges_related_pages(self, updater, wiki_repo, entity_page_from_a, entity_page_from_b):
        wiki_repo.save_page(entity_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [entity_page_from_b],
            "topic_pages": [],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("entity-openai")
        assert "video-aaa111" in page.related_pages
        assert "video-bbb222" in page.related_pages

    def test_entity_overview_rewritten(self, updater, wiki_repo, mock_llm, entity_page_from_a, entity_page_from_b):
        wiki_repo.save_page(entity_page_from_a)
        mock_llm._complete = MagicMock(return_value="Rewritten overview of OpenAI.")
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [entity_page_from_b],
            "topic_pages": [],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("entity-openai")
        assert page.overview == "Rewritten overview of OpenAI."


# --- Topic Page Tests ---


class TestTopicPageUpdate:
    def test_new_topic_created(self, updater, wiki_repo, topic_page_from_a):
        extracted = {
            "video_page": VideoPage(slug="video-aaa111", title="V", video_id="aaa111"),
            "entity_pages": [],
            "topic_pages": [topic_page_from_a],
            "concept_pages": [],
        }
        stats = updater.update_wiki(extracted)
        assert wiki_repo.exists("topic-machine-learning")
        page = wiki_repo.get_page("topic-machine-learning")
        assert len(page.contributions) == 1

    def test_topic_appends_contribution(self, updater, wiki_repo, topic_page_from_a, topic_page_from_b):
        wiki_repo.save_page(topic_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [],
            "topic_pages": [topic_page_from_b],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("topic-machine-learning")
        assert len(page.contributions) == 2
        video_ids = {c.video_id for c in page.contributions}
        assert video_ids == {"aaa111", "bbb222"}

    def test_topic_skips_duplicate_contribution(self, updater, wiki_repo, topic_page_from_a):
        wiki_repo.save_page(topic_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-aaa111", title="V", video_id="aaa111"),
            "entity_pages": [],
            "topic_pages": [topic_page_from_a],
            "concept_pages": [],
        }
        stats = updater.update_wiki(extracted)
        page = wiki_repo.get_page("topic-machine-learning")
        assert len(page.contributions) == 1  # not duplicated

    def test_topic_synthesis_rewritten(self, updater, wiki_repo, mock_llm, topic_page_from_a, topic_page_from_b):
        wiki_repo.save_page(topic_page_from_a)
        mock_llm._complete = MagicMock(return_value="Rewritten ML synthesis.")
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [],
            "topic_pages": [topic_page_from_b],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("topic-machine-learning")
        assert page.synthesis == "Rewritten ML synthesis."

    def test_topic_original_contributions_immutable(self, updater, wiki_repo, mock_llm, topic_page_from_a, topic_page_from_b):
        wiki_repo.save_page(topic_page_from_a)
        mock_llm._complete = MagicMock(return_value="New synthesis.")
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [],
            "topic_pages": [topic_page_from_b],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("topic-machine-learning")
        original = next(c for c in page.contributions if c.video_id == "aaa111")
        assert original.content == "Video A covers ML basics."  # unchanged


# --- Concept Page Tests ---


class TestConceptPageUpdate:
    def test_new_concept_created(self, updater, wiki_repo, concept_page_from_a):
        extracted = {
            "video_page": VideoPage(slug="video-aaa111", title="V", video_id="aaa111"),
            "entity_pages": [],
            "topic_pages": [],
            "concept_pages": [concept_page_from_a],
        }
        stats = updater.update_wiki(extracted)
        assert wiki_repo.exists("concept-self-attention")

    def test_concept_appends_contribution(self, updater, wiki_repo, concept_page_from_a, concept_page_from_b):
        wiki_repo.save_page(concept_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [],
            "topic_pages": [],
            "concept_pages": [concept_page_from_b],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("concept-self-attention")
        assert len(page.contributions) == 2

    def test_concept_skips_duplicate_contribution(self, updater, wiki_repo, concept_page_from_a):
        wiki_repo.save_page(concept_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-aaa111", title="V", video_id="aaa111"),
            "entity_pages": [],
            "topic_pages": [],
            "concept_pages": [concept_page_from_a],
        }
        stats = updater.update_wiki(extracted)
        page = wiki_repo.get_page("concept-self-attention")
        assert len(page.contributions) == 1

    def test_concept_synthesis_rewritten(self, updater, wiki_repo, mock_llm, concept_page_from_a, concept_page_from_b):
        wiki_repo.save_page(concept_page_from_a)
        mock_llm._complete = MagicMock(return_value="Rewritten attention synthesis.")
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [],
            "topic_pages": [],
            "concept_pages": [concept_page_from_b],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("concept-self-attention")
        assert page.synthesis == "Rewritten attention synthesis."

    def test_concept_merges_tags(self, updater, wiki_repo, concept_page_from_a, concept_page_from_b):
        wiki_repo.save_page(concept_page_from_a)
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [],
            "topic_pages": [],
            "concept_pages": [concept_page_from_b],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("concept-self-attention")
        assert "transformers" in page.tags
        assert "efficiency" in page.tags


# --- Full Extraction Tests ---


class TestFullExtraction:
    def test_combined_stats(self, updater, wiki_repo, video_page_a, entity_page_from_a, topic_page_from_a, concept_page_from_a):
        extracted = {
            "video_page": video_page_a,
            "entity_pages": [entity_page_from_a],
            "topic_pages": [topic_page_from_a],
            "concept_pages": [concept_page_from_a],
        }
        stats = updater.update_wiki(extracted)
        assert stats["created"] == 4
        assert stats["updated"] == 0
        assert stats["skipped"] == 0

    def test_second_video_updates(
        self, updater, wiki_repo, mock_llm,
        video_page_a, video_page_b,
        entity_page_from_a, entity_page_from_b,
        topic_page_from_a, topic_page_from_b,
        concept_page_from_a, concept_page_from_b,
    ):
        # Ingest first video
        extracted_a = {
            "video_page": video_page_a,
            "entity_pages": [entity_page_from_a],
            "topic_pages": [topic_page_from_a],
            "concept_pages": [concept_page_from_a],
        }
        updater.update_wiki(extracted_a)

        # Ingest second video
        mock_llm._complete = MagicMock(return_value="Synthesized from two videos.")
        extracted_b = {
            "video_page": video_page_b,
            "entity_pages": [entity_page_from_b],
            "topic_pages": [topic_page_from_b],
            "concept_pages": [concept_page_from_b],
        }
        stats = updater.update_wiki(extracted_b)

        assert stats["created"] == 1  # only video_page_b is new
        assert stats["updated"] == 3  # entity, topic, concept all updated

        # Verify entity
        entity = wiki_repo.get_page("entity-openai")
        assert len(entity.video_references) == 2

        # Verify topic
        topic = wiki_repo.get_page("topic-machine-learning")
        assert len(topic.contributions) == 2

        # Verify concept
        concept = wiki_repo.get_page("concept-self-attention")
        assert len(concept.contributions) == 2


# --- LLM Failure Handling ---


class TestLLMFailureHandling:
    def test_entity_overview_preserved_on_llm_failure(self, updater, wiki_repo, mock_llm, entity_page_from_a, entity_page_from_b):
        wiki_repo.save_page(entity_page_from_a)
        mock_llm._complete = MagicMock(side_effect=LLMError("API down"))
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [entity_page_from_b],
            "topic_pages": [],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("entity-openai")
        # References still appended even though overview rewrite failed
        assert len(page.video_references) == 2
        # Original overview preserved
        assert page.overview == "OpenAI builds GPT models."

    def test_topic_synthesis_preserved_on_llm_failure(self, updater, wiki_repo, mock_llm, topic_page_from_a, topic_page_from_b):
        wiki_repo.save_page(topic_page_from_a)
        mock_llm._complete = MagicMock(side_effect=LLMError("API down"))
        extracted = {
            "video_page": VideoPage(slug="video-bbb222", title="V", video_id="bbb222"),
            "entity_pages": [],
            "topic_pages": [topic_page_from_b],
            "concept_pages": [],
        }
        updater.update_wiki(extracted)
        page = wiki_repo.get_page("topic-machine-learning")
        # Contributions still appended
        assert len(page.contributions) == 2
        # Original synthesis preserved
        assert page.synthesis == "ML is about learning from data."
