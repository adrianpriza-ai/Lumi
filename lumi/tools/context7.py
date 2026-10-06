"""The ``context7`` tool: up-to-date library documentation on demand.

Context7 (https://context7.com) keeps an index of library docs and code
examples. Calling it is two round-trips:

1. ``resolve-library-id`` — turn a name like ``react`` into a Context7 ID
   such as ``/facebook/react``. Optional but worth it: it stops the next call
   from picking the wrong library when names collide.
2. ``query-docs`` — given a Context7 library ID and a question, return the
   best snippets.

The HTTP API is documented at https://context7.com/docs/api-guide; the two
endpoints we hit are ``GET /api/v2/libs/search`` and ``GET /api/v2/context``,
both authenticated with ``Authorization: Bearer <CONTEXT7_API_KEY>``.

The tool is auto-validated: when ``CONTEXT7_API_KEY`` is unset, ``available()``
returns ``(False, "CONTEXT7_API_KEY is not set")`` and the registry hides it
from the model. Same goes for a missing ``httpx`` — that import is already in
the dependency tree through ``openai``, but the import is defensive so a future
swap of the HTTP library cannot crash the bot at startup.

Multiple keys are supported (comma-separated, ``key_strategy`` picks how they
are spent) and the rotation reuses :class:`lumi.llm.keypool.KeyPool` — the
exact same mechanism the LLM client uses. A failed key is parked for a minute
before it is retried, so a Context7 rate limit clears itself without operator
intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Context7Config
from ..llm.keypool import KeyPool, is_retryable, mask
from ..util.log import get_logger
from ..util.text import truncate
from .base import Tool, ToolContext, ToolError, ToolResult

log = get_logger(__name__)

#: Characters of each snippet handed to the model. 0 (the default) sends the
#: snippet whole: documentation is the entire point of this tool, and a snippet
#: cut mid-example is the one thing guaranteed to be re-fetched.
_DEFAULT_MAX_CHARS = 0

#: Hard ceiling on result-count requests; Context7 ignores anything higher.
_MAX_RESULTS_CEILING = 20


@dataclass(slots=True)
class LibraryMatch:
    """One entry from ``GET /api/v2/libs/search``."""

    id: str
    title: str = ""
    description: str = ""
    score: float | None = None


@dataclass(slots=True)
class DocSnippet:
    """One code or prose snippet returned by ``GET /api/v2/context``."""

    library_id: str = ""
    title: str = ""
    body: str = ""
    #: "code" or "info" — Context7 splits results into codeList and infoSnippets.
    kind: str = "info"


class Context7Client:
    """Thin async wrapper over the two Context7 endpoints.

    Kept separate from the tool class so it can be exercised by tests without
    touching the registry, and so the rotation logic is reusable: the LLM
    client has the same problem and the same fix, so the two implementations
    share :class:`lumi.llm.keypool.KeyPool`.
    """

    def __init__(self, config: Context7Config) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.key_strategy_of())
        self._clients: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(self._pool)

    # -- HTTP plumbing ---------------------------------------------------- #

    def _client_for(self, key: str) -> Any:
        """One ``httpx.AsyncClient`` per key, built on first use."""
        client = self._clients.get(key)
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - openai pulls it in
                raise ToolError("httpx is not installed; the context7 tool needs it") from exc
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={"Authorization": f"Bearer {key}"},
            )
            self._clients[key] = client
            log.debug("context7 client created for %s", mask(key))
        return client

    async def aclose(self) -> None:
        """Close every cached client. Called when the tool is garbage-collected."""
        import contextlib

        for client in self._clients.values():
            with contextlib.suppress(Exception):  # shutdown is best-effort
                await client.aclose()

    # -- endpoints -------------------------------------------------------- #

    async def search_libraries(
        self, library_name: str, query: str, *, limit: int = 5
    ) -> list[LibraryMatch]:
        """``GET /api/v2/libs/search`` — find library IDs for a name + task."""
        if not self._pool:
            raise ToolError(
                f"{self.config.api_key_env} is not set. Put it in .env to use context7."
            )

        params: dict[str, Any] = {"libraryName": library_name}
        if query:
            params["query"] = query
        if limit:
            params["limit"] = max(1, min(limit, _MAX_RESULTS_CEILING))

        data = await self._call("libs/search", params)
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        out: list[LibraryMatch] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            lib_id = str(item.get("id") or "").strip()
            if not lib_id:
                continue
            out.append(
                LibraryMatch(
                    id=lib_id,
                    title=str(item.get("title") or ""),
                    description=str(item.get("description") or ""),
                    score=item.get("score"),
                )
            )
        return out

    async def get_context(
        self,
        library_id: str,
        query: str,
        *,
        max_tokens: int | None = None,
    ) -> list[DocSnippet]:
        """``GET /api/v2/context`` — fetch snippets for *library_id*."""
        if not self._pool:
            raise ToolError(
                f"{self.config.api_key_env} is not set. Put it in .env to use context7."
            )
        lib_id = library_id.strip()
        if not lib_id.startswith("/"):
            # The API is picky: a bare ``react`` is interpreted as a library
            # name, not an ID. Reject rather than silently misroute.
            raise ToolError(
                f"library_id {library_id!r} is not a Context7 ID — it must start with '/' "
                "(e.g. '/vercel/next.js'). Resolve it first with action='resolve_library_id'."
            )

        params: dict[str, Any] = {"libraryId": lib_id, "type": "json"}
        if query:
            params["query"] = query
        if max_tokens:
            params["maxTokens"] = max(1, max_tokens)

        data = await self._call("context", params)
        snippets: list[DocSnippet] = []
        for raw in data.get("codeSnippets") or []:
            if not isinstance(raw, dict):
                continue
            for code in raw.get("codeList") or []:
                body = str(code.get("code") or "")
                if body.strip():
                    snippets.append(
                        DocSnippet(
                            library_id=str(raw.get("libraryId") or lib_id),
                            title=str(raw.get("codeTitle") or ""),
                            body=body,
                            kind="code",
                        )
                    )
        for raw in data.get("infoSnippets") or []:
            if not isinstance(raw, dict):
                continue
            body = str(raw.get("content") or "")
            if body.strip():
                snippets.append(
                    DocSnippet(
                        library_id=lib_id,
                        title=str(raw.get("title") or ""),
                        body=body,
                        kind="info",
                    )
                )
        return snippets

    # -- call with key rotation ------------------------------------------- #

    async def _call(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET one endpoint, walking the key pool on retryable errors.

        Returns the parsed JSON dict on success. Raises :class:`ToolError`
        when no key answers.
        """
        import json

        tried: list[str] = []
        failure: BaseException | None = None
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            client = self._client_for(key)
            url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
            try:
                response = await client.get(url, params=params)
            except Exception as exc:  # noqa: BLE001 - normalised for the owner
                failure = exc
                retryable = is_retryable(exc)
                self._pool.report(key, ok=False, retryable=retryable)
                log.warning(
                    "context7 call to %s failed on key %s: %s",
                    path, mask(key), type(exc).__name__,
                )
                if not retryable:
                    break
                continue

            if response.status_code == 200:
                self._pool.report(key, ok=True)
                try:
                    data = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ToolError(
                        f"context7 returned non-JSON for {path}: {exc}"
                    ) from exc
                if not isinstance(data, dict):
                    raise ToolError(
                        f"context7 returned an unexpected shape for {path}: "
                        f"{type(data).__name__}"
                    )
                return data

            # HTTP error path. 404 / 422 are request-shaped, not key-shaped,
            # so they short-circuit out of the pool — same reasoning as the
            # LLM client's retryable detection.
            retryable = response.status_code in {401, 403, 408, 409, 429} or response.status_code >= 500
            self._pool.report(key, ok=False, retryable=retryable)
            log.warning(
                "context7 %s on key %s -> HTTP %d",
                path, mask(key), response.status_code,
            )
            if not retryable:
                raise ToolError(
                    f"context7 {path} failed: HTTP {response.status_code} — {truncate(response.text, 200, '…')}"
                )

        raise ToolError(self._explain(tried, failure))

    def _explain(self, tried: list[str], failure: BaseException | None) -> str:
        """The owner-facing message when every key has failed."""
        if not tried:
            return f"{self.config.api_key_env} is not set."
        suffix = f"; last error: {type(failure).__name__}: {failure}" if failure else ""
        if len(tried) == 1:
            return (
                f"context7 call failed for key {mask(tried[0])}{suffix}. "
                f"Check {self.config.api_key_env} and Context7's status page."
            )
        return (
            f"context7: all {len(tried)} keys in {self.config.api_key_env} failed "
            f"({self._pool.strategy}){suffix}. Check the keys and Context7's status page."
        )


class Context7Tool(Tool):
    """One tool, two actions: resolve a library ID, then fetch its docs."""

    name = "context7"
    description = """
Look up up-to-date documentation and code examples for a third-party library,
directly from the source. Use this whenever a question depends on the API of a
specific package — a function signature, a config option, an error message —
because your training data may be older than the library.

Two actions, used in order:

- `resolve_library_id`: turn a human name ("react", "next.js", "tavily") into
  a Context7 ID like "/facebook/react". Pass `library_name` and, optionally,
  `query` to rank the candidates by relevance to the task.
- `query_docs`: fetch snippets for a Context7 library ID. Always pass the ID
  that `resolve_library_id` returned, prefixed with "/". Add `query` to focus
  on a specific question.

Skip this tool for general web search, current events, or anything that is not
about a specific library's API — use the `web` tool for that.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["resolve_library_id", "query_docs"],
                "description": (
                    "`resolve_library_id` finds a Context7 ID for a library name. "
                    "`query_docs` fetches snippets for a Context7 library ID."
                ),
            },
            "library_name": {
                "type": "string",
                "description": (
                    "Library name to search for. Required for `resolve_library_id`. "
                    "Be specific when ambiguous (e.g. 'next.js' rather than 'next')."
                ),
            },
            "library_id": {
                "type": "string",
                "description": (
                    "Context7 library ID, e.g. '/vercel/next.js'. Must start with '/'. "
                    "Required for `query_docs`. Get one from `resolve_library_id` first."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional focus: for `resolve_library_id`, the task or question "
                    "used to rank the candidates; for `query_docs`, the specific "
                    "question to look up. Be specific — natural language works well."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "For `resolve_library_id`: cap on this match count (default 5).",
                "minimum": 1,
                "maximum": _MAX_RESULTS_CEILING,
            },
            "max_tokens": {
                "type": "integer",
                "description": "For `query_docs`: cap on response size in tokens (Context7 default).",
                "minimum": 100,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self, config) -> None:
        # Same shape as ``ShellTool`` and ``WebTool``: keep the top-level
        # ``Config`` around (so a future field can reach siblings without a
        # wiring change) and read the inner dataclass where we need it.
        self.config = config
        self.settings = config.tools.context7
        self._client = Context7Client(self.settings)

    # -- availability ----------------------------------------------------- #

    def available(self) -> tuple[bool, str]:
        if not self.settings.enabled:
            return False, "disabled in config (tools.context7.enabled = false)"
        if not self.settings.api_key():
            return (
                False,
                f"{self.settings.api_key_env} is not set — add it to .env to enable context7.",
            )
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False, "httpx is not installed; the context7 tool needs it"
        return True, ""

    def summary_line(self) -> str:
        keys = self.settings.api_keys()
        if not keys:
            return f"`{self.name}` _(unavailable: {self.settings.api_key_env} is not set)_"
        noun = "key" if len(keys) == 1 else "keys"
        return f"`{self.name}` — {len(keys)} {noun} ({self.settings.key_strategy_of()}) via {self.settings.base_url}"

    # -- dispatch --------------------------------------------------------- #

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(arguments.get("action", "")).strip().lower()
        if action == "resolve_library_id":
            return await self._resolve(arguments)
        if action == "query_docs":
            return await self._docs(arguments)
        raise ToolError(f"unknown action {action!r}; use resolve_library_id or query_docs")

    async def _resolve(self, arguments: dict[str, Any]) -> ToolResult:
        library_name = str(arguments.get("library_name") or "").strip()
        if not library_name:
            raise ToolError("library_name is required for resolve_library_id")
        query = str(arguments.get("query") or "").strip()
        try:
            limit = int(arguments.get("max_results") or 5)
        except (TypeError, ValueError):
            limit = 5

        try:
            matches = await self._client.search_libraries(library_name, query, limit=limit)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed result
            log.exception("context7 search_libraries failed")
            return ToolResult.failure(f"context7 search failed: {type(exc).__name__}: {exc}")

        return ToolResult(
            text=_render_library_matches(library_name, matches),
            data={
                "library_name": library_name,
                "matches": [
                    {"id": m.id, "title": m.title, "description": m.description, "score": m.score}
                    for m in matches
                ],
            },
            summary=f"context7: {len(matches)} match(es) for {library_name!r}",
        )

    async def _docs(self, arguments: dict[str, Any]) -> ToolResult:
        library_id = str(arguments.get("library_id") or "").strip()
        if not library_id:
            raise ToolError("library_id is required for query_docs")
        query = str(arguments.get("query") or "").strip()
        try:
            max_tokens = int(arguments["max_tokens"]) if arguments.get("max_tokens") else None
        except (TypeError, ValueError):
            max_tokens = None

        try:
            snippets = await self._client.get_context(library_id, query, max_tokens=max_tokens)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("context7 get_context failed")
            return ToolResult.failure(f"context7 query failed: {type(exc).__name__}: {exc}")

        body = _render_snippets(library_id, query, snippets)
        return ToolResult(
            text=body,
            data={
                "library_id": library_id,
                "query": query,
                "snippet_count": len(snippets),
                "code_count": sum(1 for s in snippets if s.kind == "code"),
                "info_count": sum(1 for s in snippets if s.kind == "info"),
            },
            summary=(
                f"context7: {len(snippets)} snippet(s) for {library_id}"
                + (f" / {query!r}" if query else "")
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_library_matches(library_name: str, matches: list[LibraryMatch]) -> str:
    if not matches:
        return (
            f"context7: no libraries for {library_name!r}. "
            "Try a more specific name, or call query_docs directly with a known ID."
        )
    lines = [f"context7 matches for {library_name!r}:"]
    for index, match in enumerate(matches, start=1):
        head = f"[{index}] {match.title or match.id}  ({match.id})"
        if match.score is not None:
            head += f"  score={match.score:.2f}"
        lines.append(head)
        if match.description:
            lines.append(truncate(match.description, 240, "…"))
    lines.append("")
    lines.append(f"Use `library_id` (e.g. '{matches[0].id}') with action='query_docs'.")
    return "\n".join(lines)


def _render_snippets(
    library_id: str, query: str, snippets: list[DocSnippet], *, max_chars: int = _DEFAULT_MAX_CHARS
) -> str:
    if not snippets:
        if query:
            return (
                f"context7: no snippets found for {library_id} / {query!r}. "
                "Try a different query or a different library."
            )
        return f"context7: no snippets returned for {library_id}."

    head = f"context7 docs for {library_id}"
    if query:
        head += f" / {query!r}"
    body = [head + ":"]
    for index, snippet in enumerate(snippets, start=1):
        title = snippet.title or (f"snippet {index}" if snippet.kind == "info" else f"code {index}")
        body.append("")
        body.append(f"[{index}] {title}  ({snippet.kind})")
        body.append(truncate(snippet.body.strip(), max_chars, marker="\n… [truncated]"))
    return "\n".join(body)


__all__ = [
    "Context7Tool",
    "Context7Client",
    "LibraryMatch",
    "DocSnippet",
]