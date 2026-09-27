"""Web access: one tool, several interchangeable providers.

``from .tool import WebTool`` — everything else is plumbing.
"""

from __future__ import annotations

from .providers.base import Page, SearchHit, SearchResult, WebProvider
from .tool import WebTool, build_providers

__all__ = ["WebTool", "build_providers", "WebProvider", "SearchResult", "SearchHit", "Page"]
