"""mcptube — YouTube video knowledge engine."""

from typing import TYPE_CHECKING

__version__ = "0.2.1"
__all__ = [
    "__version__",
]

if TYPE_CHECKING:
    from mcptube.llm import LLMClient
    from mcptube.models import Chapter, TranscriptSegment, Video
    from mcptube.service import McpTubeService
