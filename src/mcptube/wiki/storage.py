"""Concrete wiki storage — JSON files on disk + SQLite FTS5 index."""

import json
import re
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mcptube.config import settings
from mcptube.wiki.models import (
    ConceptPage,
    EntityPage,
    TopicPage,
    VideoPage,
    WikiPageBase,
    WikiPageType,
)
from mcptube.wiki.repository import WikiRepository

logger = logging.getLogger(__name__)

# Map page types to their Pydantic model classes
_PAGE_TYPE_MAP: dict[WikiPageType, type[WikiPageBase]] = {
    WikiPageType.VIDEO: VideoPage,
    WikiPageType.ENTITY: EntityPage,
    WikiPageType.TOPIC: TopicPage,
    WikiPageType.CONCEPT: ConceptPage,
}


class FileWikiRepository(WikiRepository):
    """Wiki storage backed by JSON files on disk + SQLite FTS5 index.

    Pages are stored as JSON files (reliable Pydantic round-trip) in a
    directory structure organized by page type. A SQLite FTS5 virtual
    table provides full-text search. Version history is maintained as
    timestamped copies in a _history subdirectory.

    Directory layout:
        wiki_dir/
            video/
                <slug>.json
            entity/
                <slug>.json
            topic/
                <slug>.json
            concept/
                <slug>.json
            _history/
                <slug>/
                    <timestamp>.json
    """

    _FTS_CREATE = """
        CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
            slug,
            title,
            page_type,
            tags,
            content,
            tokenize='porter unicode61'
        )
    """

    _METADATA_CREATE = """
        CREATE TABLE IF NOT EXISTS wiki_meta (
            slug        TEXT PRIMARY KEY,
            page_type   TEXT NOT NULL,
            title       TEXT NOT NULL,
            tags        TEXT DEFAULT '[]',
            summary     TEXT DEFAULT '',
            updated_at  TEXT NOT NULL
        )
    """

    def __init__(
        self,
        wiki_dir: Path | None = None,
        db_path: str | None = None,
    ) -> None:
        """Initialize wiki storage.

        Args:
            wiki_dir: Root directory for wiki JSON files.
                      Defaults to settings.data_dir / "wiki".
            db_path: Path to SQLite database for FTS5 index.
                     Defaults to settings.data_dir / "wiki.db".
                     Use ":memory:" for testing.
        """
        self._wiki_dir = wiki_dir or (settings.data_dir / "wiki")
        self._db_path = db_path or str(settings.data_dir / "wiki.db")

        # Create directory structure
        for page_type in WikiPageType:
            (self._wiki_dir / page_type.value).mkdir(parents=True, exist_ok=True)
        (self._wiki_dir / "_history").mkdir(parents=True, exist_ok=True)

        # Initialize SQLite FTS5
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)

        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._METADATA_CREATE)
        self._conn.execute(self._FTS_CREATE)
        self._conn.commit()

    # --- CRUD ---

    def save_page(self, page: WikiPageBase) -> None:
        """Persist a wiki page. Saves version history before overwriting."""
        # Save version history if page already exists
        if self.exists(page.slug):
            self._save_history(page.slug)

        # Update timestamp
        page.updated_at = datetime.now(timezone.utc)

        # Write JSON file
        path = self._page_path(page.slug, page.page_type)
        path.write_text(
            page.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Update FTS5 index
        self._index_page(page)

        logger.info("Wiki page saved: %s (%s)", page.slug, page.page_type.value)

    def get_page(self, slug: str) -> WikiPageBase | None:
        """Retrieve a wiki page by slug."""
        # Look up page type from metadata
        row = self._conn.execute(
            "SELECT page_type FROM wiki_meta WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None

        page_type = WikiPageType(row["page_type"])
        path = self._page_path(slug, page_type)
        if not path.exists():
            return None

        return self._load_page(path, page_type)

    def delete_page(self, slug: str) -> None:
        """Remove a wiki page and its FTS5 entry."""
        row = self._conn.execute(
            "SELECT page_type FROM wiki_meta WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return

        page_type = WikiPageType(row["page_type"])
        path = self._page_path(slug, page_type)
        if path.exists():
            path.unlink()

        # Remove from FTS and metadata
        self._conn.execute("DELETE FROM wiki_fts WHERE slug = ?", (slug,))
        self._conn.execute("DELETE FROM wiki_meta WHERE slug = ?", (slug,))
        self._conn.commit()

        # Remove history
        history_dir = self._wiki_dir / "_history" / slug
        if history_dir.exists():
            shutil.rmtree(history_dir)

        logger.info("Wiki page deleted: %s", slug)

    def list_pages(
        self,
        page_type: WikiPageType | None = None,
        tag: str | None = None,
    ) -> list[WikiPageBase]:
        """List wiki pages, optionally filtered."""
        sql = "SELECT slug, page_type FROM wiki_meta WHERE 1=1"
        params: list = []

        if page_type:
            sql += " AND page_type = ?"
            params.append(page_type.value)

        if tag:
            sql += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        sql += " ORDER BY updated_at DESC"

        rows = self._conn.execute(sql, params).fetchall()
        pages = []
        for row in rows:
            pt = WikiPageType(row["page_type"])
            path = self._page_path(row["slug"], pt)
            if path.exists():
                page = self._load_page(path, pt)
                if page:
                    pages.append(page)
        return pages

    def exists(self, slug: str) -> bool:
        """Check whether a wiki page exists."""
        return (
            self._conn.execute(
                "SELECT 1 FROM wiki_meta WHERE slug = ? LIMIT 1", (slug,)
            ).fetchone()
            is not None
        )

    # --- Search ---

    def search(self, query: str, limit: int = 10) -> list[WikiPageBase]:
        """Full-text search via FTS5."""
        # FTS5 query — match against title, tags, and content
        sql = """
            SELECT slug, page_type, rank
            FROM wiki_fts
            WHERE wiki_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        query = re.sub(r"[^\w\s]", " ", query).strip()

        try:
            rows = self._conn.execute(sql, (query, limit)).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 search failed: %s", e)
            return []

        pages = []
        for row in rows:
            pt = WikiPageType(row["page_type"])
            path = self._page_path(row["slug"], pt)
            if path.exists():
                page = self._load_page(path, pt)
                if page:
                    pages.append(page)
        return pages

    # --- Typed convenience getters ---

    def get_video_page(self, video_id: str) -> VideoPage | None:
        """Retrieve a video page by YouTube video ID."""
        slug = f"video-{video_id}"
        page = self.get_page(slug)
        if isinstance(page, VideoPage):
            return page

        # Fallback: scan video pages for matching video_id
        for page in self.list_pages(page_type=WikiPageType.VIDEO):
            if isinstance(page, VideoPage) and page.video_id == video_id:
                return page
        return None

    def get_entity_pages(self) -> list[EntityPage]:
        """List all entity pages."""
        return [
            p
            for p in self.list_pages(page_type=WikiPageType.ENTITY)
            if isinstance(p, EntityPage)
        ]

    def get_topic_pages(self) -> list[TopicPage]:
        """List all topic pages."""
        return [
            p
            for p in self.list_pages(page_type=WikiPageType.TOPIC)
            if isinstance(p, TopicPage)
        ]

    def get_concept_pages(self) -> list[ConceptPage]:
        """List all concept pages."""
        return [
            p
            for p in self.list_pages(page_type=WikiPageType.CONCEPT)
            if isinstance(p, ConceptPage)
        ]

    # --- Table of Contents ---

    def get_toc(self) -> str:
        """Generate a compact TOC for agent context."""
        sql = """
            SELECT slug, page_type, title, tags, summary
            FROM wiki_meta
            ORDER BY page_type, title
        """
        rows = self._conn.execute(sql).fetchall()
        if not rows:
            return "Wiki is empty."

        lines = ["# Wiki Table of Contents", ""]
        current_type = None

        for row in rows:
            pt = row["page_type"]
            if pt != current_type:
                current_type = pt
                lines.append(f"## {pt.title()}s")

            tags = json.loads(row["tags"]) if row["tags"] else []
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            summary = row["summary"]
            summary_str = f" — {summary}" if summary else ""
            lines.append(
                f"- **{row['title']}** (`{row['slug']}`){tag_str}{summary_str}"
            )

        return "\n".join(lines)

    # --- Version history ---

    def get_page_history(self, slug: str) -> list[WikiPageBase]:
        """Retrieve version history, most recent first."""
        history_dir = self._wiki_dir / "_history" / slug
        if not history_dir.exists():
            return []

        # Look up page type
        row = self._conn.execute(
            "SELECT page_type FROM wiki_meta WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return []

        page_type = WikiPageType(row["page_type"])

        versions = []
        for path in sorted(history_dir.glob("*.json"), reverse=True):
            page = self._load_page(path, page_type)
            if page:
                versions.append(page)
        return versions

    # --- Internal helpers ---

    def _page_path(self, slug: str, page_type: WikiPageType) -> Path:
        """Get the file path for a wiki page."""
        return self._wiki_dir / page_type.value / f"{slug}.json"

    def _load_page(self, path: Path, page_type: WikiPageType) -> WikiPageBase | None:
        """Load a wiki page from a JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            model_class = _PAGE_TYPE_MAP[page_type]
            return model_class.model_validate(data)
        except Exception as e:
            logger.warning("Failed to load wiki page %s: %s", path, e)
            return None

    def _index_page(self, page: WikiPageBase) -> None:
        """Update the FTS5 index and metadata table for a page."""
        # Build searchable content based on page type
        content = self._extract_searchable_content(page)
        summary = self._extract_summary(page)

        tags_json = json.dumps(page.tags)

        # Upsert metadata
        self._conn.execute(
            """INSERT INTO wiki_meta (slug, page_type, title, tags, summary, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   title = excluded.title,
                   tags = excluded.tags,
                   summary = excluded.summary,
                   updated_at = excluded.updated_at
            """,
            (
                page.slug,
                page.page_type.value,
                page.title,
                tags_json,
                summary,
                page.updated_at.isoformat(),
            ),
        )

        # Delete old FTS entry and insert new one
        self._conn.execute("DELETE FROM wiki_fts WHERE slug = ?", (page.slug,))
        self._conn.execute(
            "INSERT INTO wiki_fts (slug, title, page_type, tags, content) VALUES (?, ?, ?, ?, ?)",
            (page.slug, page.title, page.page_type.value, " ".join(page.tags), content),
        )
        self._conn.commit()

    def _save_history(self, slug: str) -> None:
        """Save the current version of a page to history before overwriting."""
        page = self.get_page(slug)
        if page is None:
            return

        history_dir = self._wiki_dir / "_history" / slug
        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

        history_path = history_dir / f"{timestamp}.json"
        history_path.write_text(
            page.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Version saved: %s → %s", slug, history_path.name)

    @staticmethod
    def _extract_searchable_content(page: WikiPageBase) -> str:
        """Extract text content from a page for FTS5 indexing."""
        parts = [page.title]

        if isinstance(page, VideoPage):
            parts.extend([page.summary, page.transcript])
        elif isinstance(page, EntityPage):
            parts.append(page.overview)
            for ref in page.video_references:
                parts.append(ref.content)
        elif isinstance(page, (TopicPage, ConceptPage)):
            parts.append(page.synthesis)
            for contrib in page.contributions:
                parts.append(contrib.content)

        return " ".join(p for p in parts if p)

    @staticmethod
    def _extract_summary(page: WikiPageBase) -> str:
        """Extract a short summary for the TOC metadata table."""
        if isinstance(page, VideoPage):
            return page.summary[:200]
        elif isinstance(page, EntityPage):
            return page.overview[:200]
        elif isinstance(page, (TopicPage, ConceptPage)):
            return page.synthesis[:200]
        return ""
