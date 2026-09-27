"""Web providers.

Each module implements :class:`~lumi.tools.web.providers.base.WebProvider` and is
registered in :func:`lumi.tools.web.tool.build_providers`. Adding a new one means
writing a single file with two methods; nothing else changes.
"""

from __future__ import annotations

from .base import Page, SearchHit, SearchResult, WebProvider

__all__ = ["WebProvider", "SearchResult", "SearchHit", "Page"]
