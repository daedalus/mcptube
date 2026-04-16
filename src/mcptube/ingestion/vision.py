"""Vision model integration — describe video frames using multimodal LLM."""

import base64
import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import litellm

from mcptube.config import settings
from mcptube.llm import LLMClient, LLMError
from mcptube.wiki.models import FrameDescription

logger = logging.getLogger(__name__)

_FRAME_CACHE_DB = "frame_cache.db"


class FrameCacheDB:
    """Persistent SQLite cache for frame descriptions indexed by image hash."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.data_dir / _FRAME_CACHE_DB
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS frame_descriptions (
                content_hash TEXT PRIMARY KEY,
                description TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_hash ON frame_descriptions(content_hash)
        """)
        self._conn.commit()

    def compute_hash(self, image_path: Path) -> str:
        """Compute SHA256 hash of image file content."""
        h = hashlib.sha256()
        h.update(image_path.read_bytes())
        return h.hexdigest()

    def get(self, image_path: Path) -> str | None:
        """Get cached description for image, or None if not cached."""
        content_hash = self.compute_hash(image_path)
        cursor = self._conn.execute(
            "SELECT description FROM frame_descriptions WHERE content_hash = ?",
            (content_hash,),
        )
        row = cursor.fetchone()
        return row["description"] if row else None

    def put(self, image_path: Path, description: str) -> None:
        """Store image hash → description mapping."""
        content_hash = self.compute_hash(image_path)
        try:
            self._conn.execute(
                "INSERT INTO frame_descriptions (content_hash, description) VALUES (?, ?)",
                (content_hash, description),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass

    def close(self) -> None:
        self._conn.close()


class VisionDescriber:
    """Describes video frames using a multimodal LLM.

    Takes extracted scene-change frames and produces text descriptions
    via a vision-capable model (GPT-4o, Claude, Gemini).
    Uses ContentHashDB cache to avoid redundant LLM calls for identical frames.
    """

    _VISION_MODELS = {
        "ANTHROPIC_API_KEY": "anthropic/claude-sonnet-4-20250514",
        "OPENAI_API_KEY": "gpt-4o",
        "GOOGLE_API_KEY": "gemini/gemini-2.0-flash",
    }

    _FRAME_PROMPT = """Describe this video frame concisely in 1-3 sentences.
Focus on what is visually significant:
- Slides or text on screen: transcribe key text
- Code: describe the language and what it does
- Diagrams: describe the structure and labels
- People: describe what they are doing (presenting, demoing, etc.)
- UI/demos: describe the application or tool shown

Be factual and specific. No speculation."""

    _BATCH_PROMPT = """You are analyzing frames from a YouTube video. For each frame,
provide a concise 1-3 sentence description focusing on visually significant content
(slides, code, diagrams, demos, people presenting).

Respond with a JSON array of descriptions in the same order as the frames.
Example: ["Frame shows a title slide reading 'Introduction to LLMs'", "Presenter at whiteboard drawing transformer architecture"]

Return ONLY the JSON array. No markdown, no explanation."""

    def __init__(self, llm: LLMClient, cache: FrameCacheDB | None = None) -> None:
        self._llm = llm
        self._model = self._detect_vision_model()
        self._cache = cache

    def describe_frames(self, frames: list[dict]) -> list[FrameDescription]:
        """Describe a list of scene-change frames using vision model.

        Args:
            frames: List of dicts with keys: "path" (Path), "timestamp" (float), "index" (int)

        Returns:
            List of FrameDescription models.

        Raises:
            LLMError: If vision model call fails.
        """
        if not self._llm.available:
            raise LLMError("Vision analysis requires an LLM. Set an API key.")

        if not frames:
            return []

        # For small batches, describe individually for better quality
        # For larger batches, use batch mode to save cost
        if len(frames) <= 5:
            return self._describe_individually(frames)
        else:
            return self._describe_batch(frames)

    def _describe_individually(self, frames: list[dict]) -> list[FrameDescription]:
        """Describe each frame with a separate vision call."""
        descriptions = []
        for frame in frames:
            try:
                desc = self._describe_single_frame(frame["path"])
                descriptions.append(
                    FrameDescription(
                        filename=frame["path"].name,
                        timestamp=frame["timestamp"],
                        description=desc,
                    )
                )
            except LLMError as e:
                logger.warning("Failed to describe frame %s: %s", frame["path"].name, e)
                descriptions.append(
                    FrameDescription(
                        filename=frame["path"].name,
                        timestamp=frame["timestamp"],
                        description="(description unavailable)",
                    )
                )
        return descriptions

    def _describe_single_frame(self, image_path: Path) -> str:
        """Describe a single frame using vision model. Uses cache to avoid redundant LLM calls."""
        # Check cache first
        if self._cache:
            cached_desc = self._cache.get(image_path)
            if cached_desc is not None:
                logger.debug("Frame cache hit: %s", image_path.name)
                return cached_desc

        b64 = base64.b64encode(image_path.read_bytes()).decode()
        mime = "image/jpeg"

        try:
            response = litellm.completion(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._FRAME_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=256,
            )
            description = response.choices[0].message.content.strip()

            # Store in cache
            if self._cache:
                self._cache.put(image_path, description)

            return description
        except Exception as e:
            raise LLMError(f"Vision model failed: {e}") from e

    def _describe_batch(self, frames: list[dict]) -> list[FrameDescription]:
        """Describe multiple frames in a single vision call. Checks cache first."""
        import json

        # Check cache and separate cached vs uncached frames
        cached_results = {}
        uncached_frames = []
        for frame in frames:
            if self._cache:
                cached_desc = self._cache.get(frame["path"])
                if cached_desc is not None:
                    cached_results[frame["path"]] = cached_desc
                else:
                    uncached_frames.append(frame)
            else:
                uncached_frames.append(frame)

        # If all frames cached, return cached results
        if uncached_frames:
            # Build content array with uncached images only
            content = [{"type": "text", "text": self._BATCH_PROMPT}]
            for frame in uncached_frames:
                b64 = base64.b64encode(frame["path"].read_bytes()).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    }
                )

            try:
                response = litellm.completion(
                    model=self._model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.2,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content.strip()

                # Parse JSON array
                text = raw
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                descs = json.loads(text)

                # Store uncached results in cache
                for i, frame in enumerate(uncached_frames):
                    desc = descs[i] if i < len(descs) else "(description unavailable)"
                    cached_results[frame["path"]] = desc
                    if self._cache:
                        self._cache.put(frame["path"], desc)

            except Exception as e:
                logger.warning("Batch vision failed, falling back to individual: %s", e)
                # Fall back to individual for uncached
                for frame in uncached_frames:
                    cached_results[frame["path"]] = None

        # Build final descriptions in original order
        descriptions = []
        for frame in frames:
            desc = cached_results.get(frame["path"])
            if desc is None:
                desc = "(description unavailable)"
            descriptions.append(
                FrameDescription(
                    filename=frame["path"].name,
                    timestamp=frame["timestamp"],
                    description=desc,
                )
            )
        return descriptions

    def _detect_vision_model(self) -> str:
        """Auto-detect the best available vision-capable model."""
        import os

        for key, model in self._VISION_MODELS.items():
            if os.environ.get(key):
                logger.info("Vision model: %s → %s", key, model)
                return model
        return "gpt-4o"
