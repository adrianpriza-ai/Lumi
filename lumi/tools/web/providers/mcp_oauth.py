"""OAuth for the MCP servers that want the owner's own authorization.

Header-token servers (``headers: {"Authorization": "Bearer ${PAT}"}``) cover
most remote MCP endpoints, but some — Linear, Notion, Atlassian — only speak
OAuth: the token belongs to a *person*, minted by that person logging in. The
MCP spec standardizes exactly one way to get one: OAuth 2.1 with PKCE and
dynamic client registration, where the client registers itself with the
server's authorization server on first use — no manual app signup, no
client_id to copy into a config file.

The installed ``mcp`` package ships the whole protocol
(:class:`mcp.client.auth.OAuthClientProvider`); what it does *not* ship is a
user. This module is the user-facing half, in three pieces:

* :class:`FileTokenStorage` — tokens per server under ``.mcp-tokens/`` in the
  project root, mode 0600. Access tokens expire and refresh silently; the
  persisted refresh token is what makes "login once" true.
* :class:`CallbackServer` — a throwaway localhost HTTP server that catches
  the browser redirect after the owner approves. This is why OAuth here needs
  the owner's browser and the bot to share a machine; a bot on a VPS should
  keep using header tokens.
* :meth:`authorize_server` — one login: start the callback server, register
  (or reuse the stored registration), show the owner the link, wait for the
  redirect, exchange the code, store the tokens.

The model's own calls never trigger a login. Sessions built through
:meth:`oauth_provider_for` carry the stored tokens, refresh them when they
expire, and if a fresh *login* is ever genuinely needed they fail with a
message telling the owner to run ``/mcp auth <name>`` — a login pops a
browser link, which is the owner's decision, not the model's.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from ....util.log import get_logger

log = get_logger(__name__)

#: How long one login may take, seconds: the owner has to see the link, open
#: it, maybe enter credentials and approve. Nothing else in Lumi waits on a
#: human; this deliberately does.
AUTHORIZE_TIMEOUT = 300.0

#: Where the browser is told to land after the owner approves. Ephemeral
#: port, loopback only: nothing outside the machine can answer for the owner.
CALLBACK_PATH = "/callback"

#: Reply body for the browser tab once the code is caught. Short and final —
#: the conversation continues in Telegram, not here.
CALLBACK_PAGE = (
    "<html><body><h2>Authorized.</h2>"
    "<p>You can close this tab and return to Telegram.</p></body></html>"
)


def tokens_dir(root: Path) -> Path:
    """The directory token files live under, one file per server."""
    return Path(root) / ".mcp-tokens"


def _tokens_path(root: Path, server: str) -> Path:
    # A server name is config-authored, but it becomes a filename: confine it
    # to one directory entry no matter what the file says.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", server) or "server"
    return tokens_dir(root) / f"{safe}.json"


class FileTokenStorage:
    """The SDK's :class:`~mcp.client.auth.TokenStorage` protocol, on disk.

    One JSON file per server holds both halves the protocol wants stored:
    the tokens, and the dynamic registration (``client_info``) — persisting
    the registration is what makes a second login reuse the same client_id
    instead of re-registering. Missing or corrupt file reads as "nothing
    stored", which lands the owner back in a fresh login rather than a crash.
    """

    def __init__(self, root: Path, server: str) -> None:
        self.path = _tokens_path(root, server)
        self.server = server

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        # 0600 on the file: the refresh token is a bearer credential for
        # everything the owner granted this server. Written beside the target
        # and renamed over it, so an interrupted write can never replace good
        # tokens with a half-written file — which _read would take for
        # "nothing stored" and silently demote to a fresh login.
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, self.path)

    # -- TokenStorage ------------------------------------------------------ #

    async def get_tokens(self) -> Any | None:
        from mcp.shared.auth import OAuthToken

        raw = self._read().get("tokens")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:  # noqa: BLE001 - unreadable tokens mean "log in again"
            log.warning("stored oauth tokens for %r are unreadable; ignoring", self.server)
            return None

    async def set_tokens(self, tokens: Any) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        self._write(data)

    async def get_client_info(self) -> Any | None:
        from mcp.shared.auth import OAuthClientInformationFull

        raw = self._read().get("client_info")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:  # noqa: BLE001 - a stale registration re-registers
            log.warning("stored oauth registration for %r is unreadable; re-registering", self.server)
            return None

    async def set_client_info(self, client_info: Any) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)

    def forget(self) -> None:
        """Drop the stored tokens and registration (the owner re-logs in)."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            return


class CallbackServer:
    """A one-shot localhost server that waits for the browser to come back.

    Started on a loopback port before the flow begins — the port goes into
    the redirect URI the server registers — it answers exactly one GET: the
    redirect carrying ``?code=…&state=…``. :meth:`wait` is what the OAuth
    flow awaits. Use it as a context manager: ``with CallbackServer() as cb:``.

    The port is ephemeral on a first login but *fixed* on a re-login: the
    stored registration reuses the redirect URI it registered the first
    time, so the same port must be listening again (it was released when the
    first callback server closed). A re-bind that fails means something else
    took the port; the caller falls back to a fresh registration.
    """

    def __init__(self, port: int = 0) -> None:
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[dict[str, str]] = self._loop.create_future()
        self._wanted_port = port
        self._httpd: ThreadingHTTPServer | None = None
        self.port = 0

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self.port}{CALLBACK_PATH}"

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # -- the two halves ---------------------------------------------------- #

    async def wait(self) -> Any:
        """What the SDK's ``callback_handler`` awaits: the redirect parameters.

        The SDK renamed the answer between versions — 1.x wants a
        ``(code, state)`` tuple, 2.x wants an ``AuthorizationCodeResult`` —
        so the shape is decided here from what the installed SDK defines,
        from the same mapping either way. A redirect that carries ``error``
        (the owner declined, or the server refused the request) raises
        instead: the caller turns that into chat text, not an empty code.
        """
        params = await self._future
        if params.get("error"):
            # The code is RFC 6749's; the description is the server's own
            # words, and against a real provider that is usually the only
            # part that says *why* (invalid_scope, invalid_client, …).
            detail = params.get("error_description") or ""
            raise RuntimeError(
                f"the login was not approved: {params['error']}"
                + (f" — {detail}" if detail else "")
            )
        try:
            from mcp.shared.auth import AuthorizationCodeResult
        except ImportError:
            return (params["code"], params.get("state"))
        return AuthorizationCodeResult(
            code=params["code"], state=params.get("state"), iss=params.get("iss")
        )

    def _deliver(self, params: dict[str, str]) -> None:
        if not self._future.done():
            self._loop.call_soon_threadsafe(self._future.set_result, params)

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                url = urlparse(self.path)
                if url.path != CALLBACK_PATH:
                    self.send_error(404)
                    return
                query = {key: values[0] for key, values in parse_qs(url.query).items()}
                body = CALLBACK_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                if "error" in query:
                    server._deliver(
                        {
                            "code": "",
                            "error": query["error"],
                            "error_description": query.get("error_description", ""),
                        }
                    )
                elif "code" in query:
                    server._deliver(
                        {"code": query["code"], "state": query.get("state", ""), "iss": query.get("iss", "")}
                    )
                else:
                    server._deliver({"code": "", "error": "no code in the redirect"})

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                log.debug("oauth callback: " + format, *args)

        return Handler

    def __enter__(self) -> CallbackServer:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self._wanted_port), self._make_handler())
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]  # type: ignore[index]
        import threading

        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _client_metadata(client_name: str, redirect_uris: list[str], scope: str = "") -> Any:
    from mcp.shared.auth import OAuthClientMetadata

    return OAuthClientMetadata(
        client_name=client_name,
        redirect_uris=redirect_uris,  # type: ignore[arg-type]
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",  # public client: PKCE, no secret
        scope=scope or None,
    )


def _requested_scope(spec: Any) -> str:
    """The scopes *spec*'s config asks the authorization request to name.

    Empty for the plain ``"auth": "oauth"`` string form. Most servers grant
    everything the account can reach anyway, but some only issue a *refresh*
    token when the request carries scopes — so for those, this is the
    difference between logging in once and once an hour.
    """
    return str(getattr(spec, "oauth_scope", "") or "").strip()


def _with_scope(url: str, scope: str) -> str:
    """Re-apply the configured scopes to a built authorization URL.

    The SDK follows the MCP spec's scope-selection strategy: on the 401
    challenge it *overwrites* ``client_metadata.scope`` with what the server
    advertises (WWW-Authenticate's scope, else the metadata's
    ``scopes_supported``) and drops the parameter when the server advertises
    none. That default is right for a client picking scopes on its own —
    wrong here, where the config names them: some providers (Atlassian above
    all) only mint a *refresh* token for the exact scopes requested, and the
    union of everything the server lists is not that request. So the owner's
    choice wins, applied to the URL the redirect handler is shown; the code
    exchange and refresh never carry a scope, so this is the only place the
    grant is decided.
    """
    if not scope:
        return url
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if query.get("scope") == [scope]:
        return url
    query["scope"] = [scope]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _is_1x(provider_cls: type) -> bool:
    """Whether the installed ``OAuthClientProvider`` still takes ``timeout=``."""
    try:
        return "timeout" in inspect.signature(provider_cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - an opaque callable
        return False


def _http_client_for(provider: Any) -> Any:
    """An HTTP client with *provider* attached as its auth handler.

    The SDK's own factory (``create_mcp_http_client``) only exists from
    1.26/2.3 — but the flow needs nothing more than an httpx client that
    consults the auth handler, and ``OAuthClientProvider`` *is* an
    ``httpx.Auth`` on every SDK major. So an older SDK gets a plain client
    instead of an ImportError that would read as "login broken" even though
    the model's own sessions (which fall back to the legacy transport) work
    fine — the same cross-version stance :func:`_http_transport` takes.
    """
    try:
        from mcp.shared._httpx_utils import create_mcp_http_client
    except ImportError:  # pragma: no cover - only an SDK older than the factory
        import httpx

        return httpx.AsyncClient(auth=provider)
    return create_mcp_http_client(auth=provider)


def _stored_redirect_port(client_info: Any) -> int:
    """The callback port the stored registration pinned, or 0 for ephemeral."""
    uris = getattr(client_info, "redirect_uris", None) or []
    for uri in uris:
        parsed = urlparse(str(uri))
        if parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path == CALLBACK_PATH:
            return parsed.port or 0
    return 0


async def authorize_server(
    spec: Any,
    root: Path,
    on_redirect: Callable[[str], Awaitable[None]],
    budget: float = AUTHORIZE_TIMEOUT,
) -> dict[str, Any]:
    """Run one login for *spec* and store what it yields.

    The owner-facing sequence: the callback server starts, the SDK registers
    (or reuses) this bot as a client, the authorization URL is handed to
    *on_redirect* — the ``/mcp auth`` command turns that into a Telegram
    message — and everything waits until the browser redirect carries the
    code back. The final request through the auth'd client is what drives
    the SDK's state machine: its 401 handling does discovery, registration,
    the redirect and the exchange, then retries with the stored token.

    A re-login reuses the stored registration — same client_id, and the
    callback server re-binds the port that registration pinned (it was
    released when the first login's server closed). If that port has since
    been taken, the stale registration is dropped and a fresh one is made.

    Returns a chat-safe summary (no token material): scopes, lifetime, and
    whether the registration was fresh.
    """
    from mcp.client.auth import OAuthClientProvider

    storage = FileTokenStorage(root, spec.name)
    client_info = await storage.get_client_info()
    had_registration = client_info is not None
    scope = _requested_scope(spec)

    async def show_link(url: str) -> None:
        # The SDK's own scope-selection strategy may have replaced the
        # configured scopes by then; the owner's config wins (see _with_scope).
        await on_redirect(_with_scope(url, scope))

    try:
        callback = CallbackServer(_stored_redirect_port(client_info) if client_info else 0)
        callback.__enter__()
    except OSError:
        if not had_registration:
            raise
        log.warning(
            "the callback port %d from %r's stored oauth registration is taken; re-registering",
            _stored_redirect_port(client_info), spec.name,
        )
        storage.forget()
        had_registration = False
        callback = CallbackServer(0)
        callback.__enter__()

    try:
        provider = OAuthClientProvider(
            spec.url,
            _client_metadata(f"Lumi ({spec.name})", [callback.redirect_uri], scope),
            storage,
            redirect_handler=show_link,
            callback_handler=callback.wait,
            **({"timeout": budget} if _is_1x(OAuthClientProvider) else {}),
        )
        client = _http_client_for(provider)
        # The whole flow — including the wait for a human to click — is
        # bounded; an owner who never comes back must not hang the command
        # forever (the 1.x SDK's own timeout kwarg only covers its requests).
        async with asyncio.timeout(budget):
            async with client:
                # Any answer that is not another 401-challenge means the token
                # stuck — servers answer a bare GET to the MCP endpoint all
                # sorts of ways, and the auth outcome is not their business.
                await client.get(spec.url)
    finally:
        callback.close()

    tokens = await storage.get_tokens()
    if tokens is None:
        raise RuntimeError(
            f"the login for {spec.name!r} did not produce a token; try /mcp auth again"
        )
    return {
        "server": spec.name,
        "scopes": tokens.scope or "",
        "expires_in": tokens.expires_in,
        "refreshable": bool(tokens.refresh_token),
        "registered": not had_registration,
    }


def oauth_provider_for(spec: Any, root: Path) -> Any:
    """A non-interactive auth handler for the sessions the model drives.

    Carries the stored tokens and refreshes them on expiry. If a *login*
    would be needed — no tokens, or the refresh token itself expired — the
    handlers here fail the call instead of opening a browser: the link is
    the owner's to click, via ``/mcp auth <name>``.
    """
    from mcp.client.auth import OAuthClientProvider

    storage = FileTokenStorage(root, spec.name)

    async def demand_login(*_args: Any) -> None:
        # The redirect handler is called with the URL, the callback handler
        # with nothing — either shape means the same thing here.
        raise RuntimeError(
            f"the oauth login for {spec.name!r} has expired; the owner runs "
            f"/mcp auth {spec.name} to authorize again"
        )

    return OAuthClientProvider(
        spec.url,
        _client_metadata(
            f"Lumi ({spec.name})",
            [f"http://localhost:0{CALLBACK_PATH}"],
            _requested_scope(spec),
        ),
        storage,
        redirect_handler=demand_login,
        callback_handler=demand_login,
    )


def oauth_supported() -> bool:
    """Whether the installed SDK ships the OAuth client at all."""
    try:
        from mcp.client.auth import OAuthClientProvider  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "AUTHORIZE_TIMEOUT",
    "CallbackServer",
    "FileTokenStorage",
    "authorize_server",
    "oauth_provider_for",
    "oauth_supported",
    "tokens_dir",
]
