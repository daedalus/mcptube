"""Vision model integration — describe video frames using multimodal LLM."""

import base64
import logging
import time
from pathlib import Path

import litellm

from mcptube.llm import LLMClient, LLMError
from mcptube.storage.cache import FrameCacheDB
from mcptube.wiki.models import FrameDescription

logger = logging.getLogger(__name__)

# Free-tier providers (e.g. Google AI Studio via OpenRouter) frequently
# return transient 429s on vision calls; retry with a short delay.
_RETRIES_429 = 3
_RETRY_429_DELAY_S = 5

# Keep batch vision calls small: free models truncate long JSON outputs,
# which silently drops descriptions for trailing frames.
_BATCH_CHUNK = 10


def _is_rate_limit_error(e: Exception) -> bool:
    """True if the exception looks like a transient rate limit."""
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "ratelimit" in msg


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
        "OPENROUTER_API_KEY": "openrouter/openrouter/free",
    }

    _FRAME_PROMPT = """Describe this video frame concisely in 1-3 sentences.
Focus on what is visually significant:
- Slides or text on screen: transcribe key text
- Code: describe the language and what it does
- Diagrams: describe the structure and labels
- People: describe what they are doing (presenting, demoing, etc.)
- UI/demos: describe the application or tool shown

Be factual and specific. No speculation.
Always differentiate factual content from speculation."""

    _BATCH_PROMPT = """You are analyzing frames from a YouTube video. For each frame,
provide a concise 1-3 sentence description focusing on visually significant content
(slides, code, diagrams, demos, people presenting).

Respond with a JSON array of descriptions in the same order as the frames.
Example: ["Frame shows a title slide reading 'Introduction to LLMs'", "Presenter at whiteboard drawing transformer architecture"]

Return ONLY the JSON array. No markdown, no explanation."""

    def __init__(
        self,
        llm: LLMClient,
        cache: FrameCacheDB | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._model = model or self._detect_vision_model()
        self._cache = cache
        candidates = [self._model, *(fallback_models or [])]
        # Keep only vision-capable candidates, preserving order, deduped.
        self._candidates = [
            m
            for m in dict.fromkeys(m for m in candidates if m)
            if self._is_vision_capable(m)
        ]

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

        # Skip vision if no vision-capable model is available
        if not self._candidates:
            logger.warning(
                "No vision-capable model available (model: %s), skipping frame descriptions",
                self._model,
            )
            return [
                FrameDescription(
                    filename=frame["path"].name,
                    timestamp=frame["timestamp"],
                    description="(vision model not available)",
                )
                for frame in frames
            ]

        # For small batches, describe individually for better quality
        # For larger batches, use batch mode to save cost
        if len(frames) <= 5:
            results = self._describe_individually(frames)
        else:
            results = self._describe_batch(frames)

        # Log cache statistics
        if self._cache:
            stats = self._cache.stats
            total = stats["hits"] + stats["misses"]
            if total > 0:
                hit_rate = stats["hits"] / total * 100
                logger.info(
                    "Frame cache: %d/%d hits (%.1f%%)",
                    stats["hits"],
                    total,
                    hit_rate,
                )

        return results

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

        last_error: Exception | None = None
        for idx, model in enumerate(self._candidates):
            if idx > 0:
                logger.warning(
                    "Vision model %s failed, trying fallback %s",
                    self._candidates[idx - 1],
                    model,
                )
            for attempt in range(1, _RETRIES_429 + 1):
                try:
                    response = litellm.completion(
                        model=model,
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
                    description = response.choices[0].message.content or ""
                    description = description.strip()

                    # Store in cache
                    if self._cache:
                        self._cache.put(image_path, description)

                    return description
                except Exception as e:
                    last_error = e
                    if attempt < _RETRIES_429 and _is_rate_limit_error(e):
                        logger.warning(
                            "Vision call rate-limited (attempt %d/%d), retrying in %ds",
                            attempt,
                            _RETRIES_429,
                            _RETRY_429_DELAY_S,
                        )
                        time.sleep(_RETRY_429_DELAY_S)
                        continue
                    break
        raise LLMError(f"Vision model failed: {last_error}") from last_error

    def _partition_cached(self, frames: list[dict]) -> tuple[dict, list[dict]]:
        """Split frames into already-described (by path) vs uncached."""
        cached_results: dict = {}
        uncached_frames: list[dict] = []
        for frame in frames:
            cached_desc = self._cache.get(frame["path"]) if self._cache else None
            if cached_desc is not None:
                cached_results[frame["path"]] = cached_desc
            else:
                uncached_frames.append(frame)
        return cached_results, uncached_frames

    def _batch_call(self, content: list[dict]) -> list:
        """Run one batched vision completion across candidate models.

        Returns the parsed description array, or raises LLMError if every
        candidate model fails.
        """
        import json

        last_error: Exception | None = None
        for idx, model in enumerate(self._candidates):
            if idx > 0:
                logger.warning(
                    "Vision model %s failed, trying fallback %s",
                    self._candidates[idx - 1],
                    model,
                )
            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.2,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                return json.loads(raw)
            except Exception as e:
                last_error = e
                logger.warning("Vision batch call failed on %s: %s", model, e)
        raise LLMError(f"Vision batch failed: {last_error}") from last_error

    def _request_batch(
        self, uncached_frames: list[dict], cached_results: dict
    ) -> list[dict]:
        """Single vision call over uncached frames; store into cached_results.

        Returns the frames that still lack a real description (e.g. a
        truncated JSON array); those should be described individually.
        """
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

        descs = self._batch_call(content)

        # Store real descriptions in cache; placeholders are NOT cached so
        # future runs retry them, and are returned for individual fallback.
        missing: list[dict] = []
        for i, frame in enumerate(uncached_frames):
            desc = descs[i] if i < len(descs) else "(description unavailable)"
            cached_results[frame["path"]] = desc
            if desc == "(description unavailable)":
                missing.append(frame)
            elif self._cache:
                self._cache.put(frame["path"], desc)
        return missing

    def _describe_batch(self, frames: list[dict]) -> list[FrameDescription]:
        """Describe frames in batched vision calls, chunked to avoid truncation."""
        cached_results, uncached_frames = self._partition_cached(frames)

        if uncached_frames:
            missing: list[dict] = []
            chunks = [
                uncached_frames[i : i + _BATCH_CHUNK]
                for i in range(0, len(uncached_frames), _BATCH_CHUNK)
            ]
            for chunk in chunks:
                try:
                    missing.extend(self._request_batch(chunk, cached_results))
                except LLMError as e:
                    logger.warning(
                        "Batch vision failed, falling back to individual: %s", e
                    )
                    # Fall back to individual for this chunk
                    individual = self._describe_individually(chunk)
                    for frame, desc in zip(chunk, individual):
                        cached_results[frame["path"]] = desc.description
            if missing:
                logger.info(
                    "Describing %d frame(s) with truncated batch output individually",
                    len(missing),
                )
                individual = self._describe_individually(missing)
                for frame, desc in zip(missing, individual):
                    cached_results[frame["path"]] = desc.description

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

    def _detect_vision_model(self) -> str | None:
        """Auto-detect the best available vision-capable model.

        Returns None if no vision-capable model is available.
        """
        import os

        for key, model in self._VISION_MODELS.items():
            if os.environ.get(key):
                logger.info("Vision model: %s → %s", key, model)
                return model
        return None

    @staticmethod
    def _is_vision_capable(model: str) -> bool:
        """Check if a model supports vision (multimodal) inputs."""
        vision_keywords = [
            "gpt-4o",
            "gpt-4v",
            "claude-3",
            "claude-sonnet-4",
            "gemini",
            "llava",
            "vision",
            # OpenRouter free multimodal models (verified via /api/v1/models)
            "openrouter/free",
            "gemma-4",
            "nemotron-3-nano-omni",
            "inkling",
            "dots-3",
        ]
        return any(kw in model.lower() for kw in vision_keywords)
