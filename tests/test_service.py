"""Tests for McpTubeService — updated for wiki engine integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from mcptube.llm import LLMClient, LLMError
from mcptube.models import Video, TranscriptSegment, Chapter
from mcptube.service import (
    AmbiguousVideoError,
    McpTubeService,
    VideoAlreadyExistsError,
    VideoNotFoundError,
)
from mcptube.storage.sqlite import SQLiteVideoRepository
from mcptube.wiki.engine import WikiEngine
from mcptube.wiki.models import (
    ConceptPage,
    EntityPage,
    FrameDescription,
    TopicPage,
    VideoPage,
    WikiPageType,
)
from mcptube.wiki.storage import FileWikiRepository


SAMPLE_LLM_EXTRACTION = """{
    "video_summary": "A video about testing.",
    "key_timestamps": {"00:00": "Intro"},
    "entities": [{"name": "Python", "category": "tool", "context": "Uses Python.", "timestamps": ["00:00"]}],
    "topics": [{"name": "Testing", "content": "Covers unit testing.", "timestamps": ["00:00"], "tags": ["dev"]}],
    "concepts": [{"name": "Mocking", "content": "Explains mock objects.", "timestamps": ["00:00"], "tags": ["testing"]}]
}"""


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.available = True
    llm._complete = MagicMock(return_value=SAMPLE_LLM_EXTRACTION)
    llm.classify = MagicMock(return_value=["testing", "python"])
    llm.answer_question = MagicMock(
        return_value="Mocking is used to isolate units under test."
    )
    return llm


@pytest.fixture
def repo(tmp_path):
    return SQLiteVideoRepository(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def wiki_engine(tmp_path, mock_llm):
    wiki_repo = FileWikiRepository(wiki_dir=tmp_path / "wiki", db_path=":memory:")
    return WikiEngine(repo=wiki_repo, llm=mock_llm)


@pytest.fixture
def mock_extractor():
    ext = MagicMock()
    ext.extract.return_value = Video(
        video_id="test1234567",
        title="Test Video",
        description="A test video",
        channel="TestChannel",
        duration=120.0,
        transcript=[
            TranscriptSegment(start=0.0, duration=5.0, text="Hello world."),
            TranscriptSegment(start=5.0, duration=5.0, text="This is a test."),
        ],
    )
    ext.parse_video_id = MagicMock(return_value="test1234567")
    return ext


@pytest.fixture
def mock_scene_extractor():
    ext = MagicMock()
    ext.extract_scene_frames.return_value = [
        {"path": "/tmp/frame1.jpg", "timestamp": 10.0, "index": 0},
        {"path": "/tmp/frame2.jpg", "timestamp": 20.0, "index": 1},
    ]
    return ext


@pytest.fixture
def mock_vision_describer():
    desc = MagicMock()
    desc.describe_frames.return_value = [
        FrameDescription(
            filename="frame1.jpg", timestamp=10.0, description="A slide about testing"
        ),
        FrameDescription(
            filename="frame2.jpg", timestamp=20.0, description="Code example"
        ),
    ]
    return desc


@pytest.fixture
def service(
    repo,
    mock_extractor,
    wiki_engine,
    mock_llm,
    mock_scene_extractor,
    mock_vision_describer,
):
    return McpTubeService(
        repository=repo,
        extractor=mock_extractor,
        wiki_engine=wiki_engine,
        llm_client=mock_llm,
        scene_extractor=mock_scene_extractor,
        vision_describer=mock_vision_describer,
    )


@pytest.fixture
def sample_video():
    return Video(
        video_id="test1234567",
        title="Test Video",
        description="A test video",
        channel="TestChannel",
        duration=120.0,
        transcript=[
            TranscriptSegment(start=0.0, duration=5.0, text="Hello world."),
            TranscriptSegment(start=5.0, duration=5.0, text="This is a test."),
        ],
    )


# --- Add Video ---


class TestAddVideo:
    def test_add_video_success(self, service):
        video = service.add_video("https://youtube.com/watch?v=test1234567")
        assert video.video_id == "test1234567"
        assert video.title == "Test Video"

    def test_add_video_saves_to_repo(self, service, repo):
        service.add_video("https://youtube.com/watch?v=test1234567")
        assert repo.exists("test1234567")

    def test_add_video_builds_wiki(self, service, wiki_engine):
        service.add_video("https://youtube.com/watch?v=test1234567")
        pages = wiki_engine.list_pages()
        assert len(pages) >= 1

    def test_add_video_auto_classifies(self, service, mock_llm):
        service.add_video("https://youtube.com/watch?v=test1234567")
        mock_llm.classify.assert_called_once()

    def test_add_duplicate_raises(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        with pytest.raises(VideoAlreadyExistsError):
            service.add_video("https://youtube.com/watch?v=test1234567")

    def test_add_video_text_only(self, service, mock_scene_extractor):
        service.add_video("https://youtube.com/watch?v=test1234567", text_only=True)
        mock_scene_extractor.extract_scene_frames.assert_not_called()

    def test_add_video_full_analysis(
        self, service, mock_scene_extractor, mock_vision_describer
    ):
        service.add_video("https://youtube.com/watch?v=test1234567", text_only=False)
        mock_scene_extractor.extract_scene_frames.assert_called_once()
        mock_vision_describer.describe_frames.assert_called_once()

    def test_add_video_frame_stats_populated(
        self, service, mock_scene_extractor, mock_vision_describer
    ):
        video = service.add_video(
            "https://youtube.com/watch?v=test1234567", text_only=False
        )
        assert video.frame_stats["ffmpeg_extracted"] == 2
        assert video.frame_stats["llm_processed"] == 2

    def test_add_video_frame_stats_text_only(self, service, mock_scene_extractor):
        video = service.add_video(
            "https://youtube.com/watch?v=test1234567", text_only=True
        )
        assert video.frame_stats["ffmpeg_extracted"] == 0
        assert video.frame_stats["llm_processed"] == 0


# --- List / Info ---


class TestListAndInfo:
    def test_list_empty(self, service):
        assert service.list_videos() == []

    def test_list_after_add(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        videos = service.list_videos()
        assert len(videos) == 1
        assert videos[0].video_id == "test1234567"

    def test_get_info(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        video = service.get_info("test1234567")
        assert video.title == "Test Video"

    def test_get_info_not_found(self, service):
        with pytest.raises(VideoNotFoundError):
            service.get_info("nonexistent")


# --- Remove Video ---


class TestRemoveVideo:
    def test_remove_video(self, service, repo):
        service.add_video("https://youtube.com/watch?v=test1234567")
        service.remove_video("test1234567")
        assert not repo.exists("test1234567")

    def test_remove_cleans_wiki(self, service, wiki_engine):
        service.add_video("https://youtube.com/watch?v=test1234567")
        service.remove_video("test1234567")
        vp = wiki_engine.get_page("video-test1234567")
        assert vp is None

    def test_remove_not_found(self, service):
        with pytest.raises(VideoNotFoundError):
            service.remove_video("nonexistent")


# --- Wiki Operations ---


class TestWikiSearch:
    def test_wiki_search(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        results = service.wiki_search("testing")
        assert len(results) >= 1

    def test_wiki_search_no_results(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        results = service.wiki_search("quantum physics")
        assert len(results) == 0

    def test_wiki_search_no_engine_raises(self, repo, mock_extractor, mock_llm):
        svc = McpTubeService(
            repository=repo, extractor=mock_extractor, llm_client=mock_llm
        )
        with pytest.raises(RuntimeError):
            svc.wiki_search("test")


class TestWikiAsk:
    def test_wiki_ask(self, service, mock_llm):
        service.add_video("https://youtube.com/watch?v=test1234567")
        mock_llm._complete = MagicMock(
            return_value="Mocking replaces real objects with fakes."
        )
        answer = service.wiki_ask("What is mocking?")
        assert len(answer) > 0

    def test_wiki_ask_no_engine_raises(self, repo, mock_extractor, mock_llm):
        svc = McpTubeService(
            repository=repo, extractor=mock_extractor, llm_client=mock_llm
        )
        with pytest.raises(RuntimeError):
            svc.wiki_ask("test")


class TestWikiList:
    def test_wiki_list_all(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        pages = service.wiki_list()
        assert len(pages) >= 1

    def test_wiki_list_by_type(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        entities = service.wiki_list(page_type=WikiPageType.ENTITY)
        assert all(isinstance(p, EntityPage) for p in entities)

    def test_wiki_list_empty(self, service):
        pages = service.wiki_list()
        assert pages == []


class TestWikiShow:
    def test_wiki_show(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        page = service.wiki_show("video-test1234567")
        assert page is not None
        assert isinstance(page, VideoPage)

    def test_wiki_show_not_found(self, service):
        assert service.wiki_show("nonexistent") is None


class TestWikiToc:
    def test_wiki_toc_empty(self, service):
        toc = service.wiki_toc()
        assert "empty" in toc.lower()

    def test_wiki_toc_after_add(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        toc = service.wiki_toc()
        assert "Test Video" in toc


class TestWikiHistory:
    def test_wiki_history_empty(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        history = service.wiki_history("video-test1234567")
        assert len(history) == 0


# --- Ask Video (direct transcript Q&A) ---


class TestAskVideo:
    def test_ask_single_video(self, service, mock_llm):
        service.add_video("https://youtube.com/watch?v=test1234567")
        answer = service.ask_video("test1234567", "What is this about?")
        mock_llm.answer_question.assert_called_once()
        assert len(answer) > 0

    def test_ask_video_not_found(self, service):
        with pytest.raises(VideoNotFoundError):
            service.ask_video("nonexistent", "question")

    def test_ask_videos_multiple(self, service, mock_llm, mock_extractor):
        service.add_video("https://youtube.com/watch?v=test1234567")
        mock_extractor.extract.return_value = Video(
            video_id="test4567890",
            title="V2",
            channel="C",
            duration=60.0,
            transcript=[
                TranscriptSegment(start=0.0, duration=5.0, text="Second video.")
            ],
        )
        mock_extractor.parse_video_id = MagicMock(return_value="test4567890")
        service.add_video("https://youtube.com/watch?v=test4567890")
        answer = service.ask_videos(["test1234567", "test4567890"], "Compare them")
        assert len(answer) > 0


# --- Classification ---


class TestClassify:
    def test_classify_video(self, service, mock_llm):
        service.add_video("https://youtube.com/watch?v=test1234567")
        mock_llm.classify = MagicMock(return_value=["new-tag"])
        tags = service.classify_video("test1234567")
        assert tags == ["new-tag"]

    def test_classify_not_found(self, service):
        with pytest.raises(VideoNotFoundError):
            service.classify_video("nonexistent")


# --- Resolve Video ---


class TestResolveVideo:
    def test_resolve_by_id(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        video = service.resolve_video("test1234567")
        assert video.video_id == "test1234567"

    def test_resolve_by_index(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        video = service.resolve_video("1")
        assert video.video_id == "test1234567"

    def test_resolve_by_title_substring(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        video = service.resolve_video("Test")
        assert video.video_id == "test1234567"

    def test_resolve_not_found(self, service):
        with pytest.raises(VideoNotFoundError):
            service.resolve_video("nonexistent")

    def test_resolve_index_out_of_range(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        with pytest.raises(VideoNotFoundError):
            service.resolve_video("99")


# --- Frame Extraction ---


class TestFrames:
    def test_get_frame_not_found(self, service):
        with pytest.raises(VideoNotFoundError):
            service.get_frame("nonexistent", 10.0)

    def test_get_frame_by_query_not_found(self, service):
        with pytest.raises(VideoNotFoundError):
            service.get_frame_by_query("nonexistent", "test")

    def test_get_frame_by_query_matches(self, service):
        service.add_video("https://youtube.com/watch?v=test1234567")
        with patch.object(
            service._frame_extractor, "extract_frame", return_value="/tmp/frame.jpg"
        ):
            result = service.get_frame_by_query("test1234567", "hello")
            assert result["text"] == "Hello world."
