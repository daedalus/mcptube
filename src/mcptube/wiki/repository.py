"""Abstract repository interface for wiki page storage."""

from abc import ABC, abstractmethod

from mcptube.wiki.models import (
    ConceptPage,
    EntityPage,
    TopicPage,
    VideoPage,
    WikiPageBase,
    WikiPageType,
)


class WikiRepository(ABC):
    """Abstract base class defining the wiki storage contract.

    Concrete implementations (markdown files, database-backed, etc.)
    must implement this interface. Keeps the wiki engine decoupled
    from any specific storage backend (DIP).
    """

    # --- CRUD operations ---

    @abstractmethod
    def save_page(self, page: WikiPageBase) -> None:
        """Persist a wiki page to storage. Upserts if slug already exists.

        Args:
            page: Any wiki page model (VideoPage, EntityPage, TopicPage, ConceptPage).
        """

    @abstractmethod
    def get_page(self, slug: str) -> WikiPageBase | None:
        """Retrieve a wiki page by slug.

        Args:
            slug: URL/filename-safe identifier.

        Returns:
            The wiki page, or None if not found.
        """

    @abstractmethod
    def delete_page(self, slug: str) -> None:
        """Remove a wiki page from storage. No-op if slug does not exist.

        Args:
            slug: URL/filename-safe identifier.
        """

    @abstractmethod
    def list_pages(
        self,
        page_type: WikiPageType | None = None,
        tag: str | None = None,
    ) -> list[WikiPageBase]:
        """List wiki pages, optionally filtered by type and/or tag.

        Args:
            page_type: If provided, only return pages of this type.
            tag: If provided, only return pages with this tag.

        Returns:
            List of wiki pages (metadata only — synthesis/contributions may be truncated).
        """

    @abstractmethod
    def exists(self, slug: str) -> bool:
        """Check whether a wiki page with the given slug exists.

        Args:
            slug: URL/filename-safe identifier.
        """

    # --- Search ---

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[WikiPageBase]:
        """Full-text search across all wiki pages via FTS5.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching wiki pages ordered by relevance.
        """

    # --- Typed convenience getters ---

    @abstractmethod
    def get_video_page(self, video_id: str) -> VideoPage | None:
        """Retrieve a video page by YouTube video ID.

        Args:
            video_id: YouTube video ID (not the slug).

        Returns:
            VideoPage or None if not found.
        """

    @abstractmethod
    def get_entity_pages(self) -> list[EntityPage]:
        """List all entity pages."""

    @abstractmethod
    def get_topic_pages(self) -> list[TopicPage]:
        """List all topic pages."""

    @abstractmethod
    def get_concept_pages(self) -> list[ConceptPage]:
        """List all concept pages."""

    # --- Table of Contents ---

    @abstractmethod
    def get_toc(self) -> str:
        """Generate a compact table of contents for agent context.

        Returns a concise summary of all wiki pages — titles, types,
        tags, and short descriptions — suitable for inclusion in an
        LLM prompt to help the agent decide which pages to read.

        Returns:
            Formatted TOC string.
        """

    # --- Version history ---

    @abstractmethod
    def get_page_history(self, slug: str) -> list[WikiPageBase]:
        """Retrieve version history for a wiki page.

        Args:
            slug: URL/filename-safe identifier.

        Returns:
            List of previous versions, most recent first.
        """
