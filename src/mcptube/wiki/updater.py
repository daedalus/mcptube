"""Wiki updater — merges new extractions into existing wiki pages."""

import logging
from datetime import datetime, timezone

from mcptube.llm import LLMClient, LLMError
from mcptube.wiki.models import (
    ConceptPage,
    EntityPage,
    TopicPage,
    VideoContribution,
    VideoPage,
    WikiPageBase,
)
from mcptube.wiki.repository import WikiRepository

logger = logging.getLogger(__name__)


class WikiUpdater:
    """Merges newly extracted wiki pages into the existing wiki.

    Update policies:
    - VideoPage: write-once, never updated (skip if exists)
    - EntityPage: append-only — new video references are added, overview updated
    - TopicPage: synthesis rewritten by LLM, per-video contributions are immutable
    - ConceptPage: synthesis rewritten by LLM, per-video contributions are immutable
    """

    _SYNTHESIS_PROMPT = """You are a knowledge synthesis system. Given multiple video perspectives 
on a {page_type}, write an updated synthesis that integrates ALL perspectives.

{page_type_upper}: {title}

EXISTING CONTRIBUTIONS:
{contributions}

Write a concise synthesis (3-6 sentences) that:
- Integrates all perspectives fairly
- Notes agreements and disagreements between sources
- Highlights how understanding of this {page_type} has evolved across videos
- Does NOT favour any single source
- Always differentiate factual claims from non-factual content, fiction from non-fiction, and speculation from well-grounded truth

Return ONLY the synthesis text. No markdown formatting, no headers."""

    _ENTITY_OVERVIEW_PROMPT = """You are a knowledge synthesis system. Given multiple video references 
to an entity, write an updated overview.

ENTITY: {name} ({category})

VIDEO REFERENCES:
{references}

Write a concise overview (2-4 sentences) that:
- Summarises who/what this entity is based on the video references
- Notes different contexts in which this entity appears
- Captures the most important facts mentioned
- Always differentiate factual claims from non-factual content, fiction from non-fiction, and speculation from well-grounded truth

Return ONLY the overview text. No markdown formatting, no headers."""

    def __init__(self, repo: WikiRepository, llm: LLMClient) -> None:
        self._repo = repo
        self._llm = llm

    def update_wiki(self, extracted: dict) -> dict:
        """Merge extracted pages into the wiki.

        Args:
            extracted: Output from KnowledgeExtractor.extract() with keys:
                       "video_page", "entity_pages", "topic_pages", "concept_pages"

        Returns:
            Dict with counts: {"created": int, "updated": int, "skipped": int}
        """
        stats = {"created": 0, "updated": 0, "skipped": 0}

        # Video page — write-once
        video_page = extracted["video_page"]
        self._handle_video_page(video_page, stats)

        # Entity pages — append-only
        for page in extracted["entity_pages"]:
            self._handle_entity_page(page, stats)

        # Topic pages — synthesis rewritten
        for page in extracted["topic_pages"]:
            self._handle_topic_page(page, stats)

        # Concept pages — synthesis rewritten
        for page in extracted["concept_pages"]:
            self._handle_concept_page(page, stats)

        logger.info(
            "Wiki update complete: %d created, %d updated, %d skipped",
            stats["created"],
            stats["updated"],
            stats["skipped"],
        )
        return stats

    def _handle_video_page(self, page: VideoPage, stats: dict) -> None:
        """Video pages are write-once — skip if exists."""
        if self._repo.exists(page.slug):
            logger.info("Video page already exists, skipping: %s", page.slug)
            stats["skipped"] += 1
            return

        self._repo.save_page(page)
        stats["created"] += 1

    def _handle_entity_page(self, new_page: EntityPage, stats: dict) -> None:
        """Entity pages are append-only — add new references, update overview."""
        existing = self._repo.get_page(new_page.slug)

        if existing is None or not isinstance(existing, EntityPage):
            # New entity — save as-is
            self._repo.save_page(new_page)
            stats["created"] += 1
            return

        # Check if this video already contributed
        existing_video_ids = {ref.video_id for ref in existing.video_references}
        new_refs = [
            ref
            for ref in new_page.video_references
            if ref.video_id not in existing_video_ids
        ]

        if not new_refs:
            logger.info(
                "Entity already has this video's references, skipping: %s",
                new_page.slug,
            )
            stats["skipped"] += 1
            return

        # Append new references
        existing.video_references.extend(new_refs)

        # Update related pages
        for rp in new_page.related_pages:
            if rp not in existing.related_pages:
                existing.related_pages.append(rp)

        # Merge tags
        for tag in new_page.tags:
            if tag not in existing.tags:
                existing.tags.append(tag)

        # Rewrite overview if multiple sources now
        if len(existing.video_references) > 1 and self._llm.available:
            try:
                existing.overview = self._rewrite_entity_overview(existing)
            except LLMError as e:
                logger.warning("Failed to rewrite entity overview: %s", e)

        self._repo.save_page(existing)
        stats["updated"] += 1

    def _handle_topic_page(self, new_page: TopicPage, stats: dict) -> None:
        """Topic pages — append contribution, rewrite synthesis."""
        existing = self._repo.get_page(new_page.slug)

        if existing is None or not isinstance(existing, TopicPage):
            self._repo.save_page(new_page)
            stats["created"] += 1
            return

        # Check for duplicate contribution
        existing_video_ids = {c.video_id for c in existing.contributions}
        new_contribs = [
            c for c in new_page.contributions if c.video_id not in existing_video_ids
        ]

        if not new_contribs:
            logger.info(
                "Topic already has this video's contribution, skipping: %s",
                new_page.slug,
            )
            stats["skipped"] += 1
            return

        # Append immutable contributions
        existing.contributions.extend(new_contribs)

        # Update related pages and tags
        for rp in new_page.related_pages:
            if rp not in existing.related_pages:
                existing.related_pages.append(rp)
        for tag in new_page.tags:
            if tag not in existing.tags:
                existing.tags.append(tag)

        # Rewrite synthesis
        if self._llm.available:
            try:
                existing.synthesis = self._rewrite_synthesis(existing, "topic")
            except LLMError as e:
                logger.warning("Failed to rewrite topic synthesis: %s", e)

        self._repo.save_page(existing)
        stats["updated"] += 1

    def _handle_concept_page(self, new_page: ConceptPage, stats: dict) -> None:
        """Concept pages — append contribution, rewrite synthesis."""
        existing = self._repo.get_page(new_page.slug)

        if existing is None or not isinstance(existing, ConceptPage):
            self._repo.save_page(new_page)
            stats["created"] += 1
            return

        # Check for duplicate contribution
        existing_video_ids = {c.video_id for c in existing.contributions}
        new_contribs = [
            c for c in new_page.contributions if c.video_id not in existing_video_ids
        ]

        if not new_contribs:
            logger.info(
                "Concept already has this video's contribution, skipping: %s",
                new_page.slug,
            )
            stats["skipped"] += 1
            return

        # Append immutable contributions
        existing.contributions.extend(new_contribs)

        # Update related pages and tags
        for rp in new_page.related_pages:
            if rp not in existing.related_pages:
                existing.related_pages.append(rp)
        for tag in new_page.tags:
            if tag not in existing.tags:
                existing.tags.append(tag)

        # Rewrite synthesis
        if self._llm.available:
            try:
                existing.synthesis = self._rewrite_synthesis(existing, "concept")
            except LLMError as e:
                logger.warning("Failed to rewrite concept synthesis: %s", e)

        self._repo.save_page(existing)
        stats["updated"] += 1

    def _rewrite_synthesis(self, page: TopicPage | ConceptPage, page_type: str) -> str:
        """Use LLM to rewrite synthesis from all contributions."""
        contributions_text = "\n\n".join(
            f"### From: {c.title} ({c.video_id}) by {c.channel}\n{c.content}"
            for c in page.contributions
        )

        prompt = self._SYNTHESIS_PROMPT.format(
            page_type=page_type,
            page_type_upper=page_type.upper(),
            title=page.title,
            contributions=contributions_text,
        )

        return self._llm._complete(prompt, max_tokens=1024)

    def _rewrite_entity_overview(self, page: EntityPage) -> str:
        """Use LLM to rewrite entity overview from all references."""
        references_text = "\n\n".join(
            f"### From: {ref.title} ({ref.video_id}) by {ref.channel}\n{ref.content}"
            for ref in page.video_references
        )

        prompt = self._ENTITY_OVERVIEW_PROMPT.format(
            name=page.title,
            category=page.category.value,
            references=references_text,
        )

        return self._llm._complete(prompt, max_tokens=512)
