"""OAuth for MCP servers: token storage, the browser login, and the wiring.

The SDK ships the protocol; what is under test here is the part around it —
that tokens land on disk mode 0600 keyed per server, that a full login
against a *fake OAuth server* (a real HTTP server speaking the discovery,
registration, authorization and token endpoints) completes through the
installed SDK with the loopback callback actually receiving the browser's
redirect, and that the model's own sessions fail with "run /mcp auth"
instead of ever popping a browser.
"""

from __future__ import annotations

import asyncio
import json
import stat
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from mcp.client.auth import OAuthClientProvider

from lumi.tools.web.providers.mcp_oauth import (
    CallbackServer,
    FileTokenStorage,
    MCPOAuthLoginRequired,
    authorize_server,
    oauth_provider_for,
    tokens_dir,
)
from lumi.tools.web.providers.mcp_provider import ServerSpec


def remote_spec(url: str, scope: str = "") -> ServerSpec:
    return ServerSpec(name="fake", kind="remote", url=url, oauth=True, oauth_scope=scope)


# --------------------------------------------------------------------------- #
# config parsing
# --------------------------------------------------------------------------- #


def test_auth_key_shapes(config) -> None:
    string_form = ServerSpec.from_json("a", {"url": "https://a.test", "auth": "oauth"})
    dict_form = ServerSpec.from_json("b", {"url": "https://b.test", "auth": {"type": "oauth"}})
    absent = ServerSpec.from_json("c", {"url": "https://c.test"})
    other = ServerSpec.from_json("d", {"url": "https://d.test", "auth": "basic"})

    assert string_form.oauth and dict_form.oauth
    assert not absent.oauth and not other.oauth
    assert not string_form.oauth_scope  # no scopes asked for


def test_scopes_reach_the_spec_from_either_dict_spelling(config) -> None:
    """Atlassian-shaped servers only mint a *refresh* token when the
    authorization request names scopes, so the config's spelling (string or
    list) must both land as the space-joined string the OAuth metadata wants."""
    as_string = ServerSpec.from_json(
        "a", {"url": "https://a.test", "auth": {"type": "oauth", "scopes": "read offline_access"}}
    )
    as_list = ServerSpec.from_json(
        "b",
        {"url": "https://b.test", "auth": {"type": "oauth", "scopes": ["read", "offline_access"]}},
    )

    assert as_string.oauth and as_string.oauth_scope == "read offline_access"
    assert as_list.oauth and as_list.oauth_scope == "read offline_access"


def test_client_metadata_carries_the_requested_scope() -> None:
    from lumi.tools.web.providers.mcp_oauth import _client_metadata

    scoped = _client_metadata("Lumi (a)", ["http://localhost:1/callback"], "read write")
    bare = _client_metadata("Lumi (b)", ["http://localhost:2/callback"])

    assert scoped.scope == "read write"
    assert bare.scope is None  # absent beats empty: no scope param on the wire


def test_with_scope_makes_the_configured_request_win() -> None:
    """The SDK's scope-selection strategy overwrites client_metadata.scope
    with whatever the server advertises (or nothing) — so the configured
    scopes are re-applied to the built URL, and an unscoped config leaves the
    URL untouched."""
    from lumi.tools.web.providers.mcp_oauth import _with_scope

    url = "http://as.test/authorize?state=s1&scope=server-chosen%20scope&resource=r"
    assert _with_scope(url, "") == url  # nothing configured: server's choice stands

    rewritten = _with_scope(url, "read offline_access")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(rewritten).query)
    assert query["scope"] == ["read offline_access"]
    assert query["state"] == ["s1"] and query["resource"] == ["r"]  # nothing else lost

    # The SDK dropped the parameter entirely (server advertised no scopes):
    # the configured one is added back.
    plain = "http://as.test/authorize?state=s2&code_challenge=c"
    assert (
        _with_scope(plain, "read")
        == "http://as.test/authorize?state=s2&code_challenge=c&scope=read"
    )


# --------------------------------------------------------------------------- #
# FileTokenStorage
# --------------------------------------------------------------------------- #


async def test_tokens_round_trip_and_isolation(tmp_path) -> None:
    from mcp.shared.auth import OAuthToken

    first = FileTokenStorage(tmp_path, "linear")
    second = FileTokenStorage(tmp_path, "notion")
    tokens = OAuthToken(access_token="at", refresh_token="rt", expires_in=3600, scope="read write")

    await first.set_tokens(tokens)
    client = {"client_id": "cid", "redirect_uris": ["http://localhost:1/callback"]}
    await first.set_client_info(
        type("Info", (), {"model_dump": staticmethod(lambda mode="": client)})()
    )

    got = await first.get_tokens()
    assert got is not None and got.access_token == "at" and got.refresh_token == "rt"
    assert (await second.get_tokens()) is None  # per-server files

    data = json.loads((tokens_dir(tmp_path) / "linear.json").read_text())
    assert data["tokens"]["access_token"] == "at"
    assert data["client_info"] == client
    # Writes go through a temp file renamed into place — nothing half-written
    # must survive next to the real one.
    assert list(tokens_dir(tmp_path).glob("*.tmp")) == []


async def test_a_token_file_is_only_readable_by_the_owner(tmp_path) -> None:
    storage = FileTokenStorage(tmp_path, "linear")

    from mcp.shared.auth import OAuthToken

    await storage.set_tokens(OAuthToken(access_token="at"))
    mode = stat.S_IMODE(storage.path.stat().st_mode)
    assert mode == 0o600
    dir_mode = stat.S_IMODE(storage.path.parent.stat().st_mode)
    assert dir_mode & 0o077 == 0


async def test_a_corrupt_token_file_reads_as_nothing(tmp_path) -> None:
    storage = FileTokenStorage(tmp_path, "linear")
    storage.path.parent.mkdir(parents=True)
    storage.path.write_text("{not json", encoding="utf-8")

    assert (await storage.get_tokens()) is None
    assert (await storage.get_client_info()) is None

    storage.forget()
    assert not storage.path.exists()


# --------------------------------------------------------------------------- #
# the provider the model's sessions use
# --------------------------------------------------------------------------- #


async def test_session_auth_never_opens_a_browser(tmp_path) -> None:
    """A session whose login has expired must fail with the remedy, not try
    to take over a browser the model has no business driving."""
    provider = oauth_provider_for(remote_spec("https://linear.test"), tmp_path)

    assert isinstance(provider, OAuthClientProvider)
    metadata = provider.context.client_metadata
    assert metadata.grant_types == ["authorization_code", "refresh_token"]
    with pytest.raises(MCPOAuthLoginRequired, match="/mcp auth fake"):
        await provider.context.redirect_handler("https://auth.test/authorize")
    with pytest.raises(MCPOAuthLoginRequired, match="/mcp auth fake"):
        await provider.context.callback_handler()


# --------------------------------------------------------------------------- #
# the loopback callback server
# --------------------------------------------------------------------------- #


def code_of(result: object) -> str:
    """The code from a callback result, whichever shape the SDK wants."""
    if isinstance(result, tuple):
        return result[0]
    return result.code  # type: ignore[attr-defined]


async def test_the_callback_server_catches_the_browser_redirect() -> None:
    with CallbackServer() as callback:
        code, state = "ac-123", "st-xyz"

        def browser() -> None:
            query = urllib.parse.urlencode({"code": code, "state": state})
            urllib.request.urlopen(f"http://localhost:{callback.port}/callback?{query}")

        await asyncio.get_running_loop().run_in_executor(None, browser)

        result = await callback.wait()
        assert code_of(result) == code
        assert callback.port > 0


async def test_the_callback_server_reports_a_declined_login() -> None:
    with CallbackServer() as callback:

        def browser() -> None:
            urllib.request.urlopen(
                f"http://localhost:{callback.port}/callback?error=access_denied"
                "&error_description=The+user+denied+the+request"
            )

        await asyncio.get_running_loop().run_in_executor(None, browser)

        # The description is the server's own words about *why* — the only
        # part of a real provider's refusal that is worth iterating on.
        with pytest.raises(RuntimeError, match="access_denied") as caught:
            await callback.wait()
        assert "The user denied the request" in str(caught.value)


# --------------------------------------------------------------------------- #
# authorize_server — end to end against a fake OAuth server
# --------------------------------------------------------------------------- #


class FakeOAuthServer:
    """A minimal authorization + resource server speaking MCP's OAuth dialect.

    Real HTTP on the loopback, driven by the *installed* SDK: 401 challenge
    with protected-resource metadata, authorization-server metadata, dynamic
    registration, the authorization redirect, and the code exchange. The
    fake "browser" is the test itself — it reads the redirect URI the SDK
    registered and mints a code straight into the loopback callback.
    """

    def __init__(self) -> None:
        self.mcp_auth_header: str | None = None  # last Authorization seen on /mcp
        self.registered_redirect: str | None = None  # what the client registered
        self.registration_count = 0

    def start(self) -> FakeOAuthServer:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                url = urllib.parse.urlparse(self.path)
                port = self.server.server_address[1]  # type: ignore[attr-defined]
                base = f"http://127.0.0.1:{port}"
                if url.path == "/mcp":
                    if self.headers.get("Authorization") == "Bearer test-access-token":
                        server.mcp_auth_header = self.headers.get("Authorization")
                        self._json({"status": "authorized"})
                    else:
                        self.send_response(401)
                        self.send_header(
                            "WWW-Authenticate",
                            f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"',
                        )
                        self.end_headers()
                elif url.path in (
                    "/.well-known/oauth-protected-resource",
                    "/.well-known/oauth-protected-resource/mcp",
                ):
                    self._json(
                        {
                            "resource": f"{base}/mcp",
                            "authorization_servers": [base],
                        }
                    )
                elif url.path == "/.well-known/oauth-authorization-server":
                    self._json(
                        {
                            "issuer": base,
                            "authorization_endpoint": f"{base}/authorize",
                            "token_endpoint": f"{base}/token",
                            "registration_endpoint": f"{base}/register",
                            "response_types_supported": ["code"],
                            "grant_types_supported": ["authorization_code", "refresh_token"],
                            "code_challenge_methods_supported": ["S256"],
                            "token_endpoint_auth_methods_supported": ["none"],
                        }
                    )
                else:
                    self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length or 0)
                # Registration arrives as JSON; the token exchange as a
                # form body — parse each the way its client sent it.
                if "json" in (self.headers.get("Content-Type") or ""):
                    body = json.loads(raw or b"{}")
                else:
                    body = {
                        key: values[0]
                        for key, values in urllib.parse.parse_qs(raw.decode()).items()
                    }
                if self.path == "/register":
                    server.registration_count += 1
                    server.registered_redirect = (body.get("redirect_uris") or [""])[0]
                    self._json({**body, "client_id": "fake-client-1", "client_id_issued_at": 1})
                elif self.path == "/token":
                    grant = body.get("grant_type")
                    if (
                        grant == "authorization_code"
                        and body.get("code") == "ac-1"
                        and body.get("code_verifier")
                    ):
                        self._json(
                            {
                                "access_token": "test-access-token",
                                "token_type": "Bearer",
                                "expires_in": 3600,
                                "refresh_token": "test-refresh-token",
                                "scope": "issues:write",
                            }
                        )
                    else:
                        self._json({"error": "invalid_grant"}, status=400)
                else:
                    self.send_error(404)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        import threading

        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"


async def test_a_full_login_through_the_installed_sdk(tmp_path) -> None:
    fake = FakeOAuthServer().start()
    try:
        links: list[str] = []

        async def on_redirect(url: str) -> None:
            links.append(url)
            # The fake browser: mint a code and land it on the loopback
            # callback the same way the owner's browser would. The iss is
            # RFC 9207's authorization-response issuer — 2.x validates it.
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            state = query["state"][0]
            origin = (
                urllib.parse.urlparse(url)
                ._replace(path="", params="", query="", fragment="")
                .geturl()
            )
            redirect = fake.registered_redirect
            assert redirect, "the SDK must register before redirecting"
            landing = (
                f"{redirect}?code=ac-1&state={urllib.parse.quote(state)}"
                f"&iss={urllib.parse.quote(origin)}"
            )
            await asyncio.to_thread(urllib.request.urlopen, landing)

        spec = remote_spec(fake.url)
        summary = await authorize_server(spec, tmp_path, on_redirect=on_redirect)

        # The link was shown to the owner exactly once.
        assert len(links) == 1
        assert query_state(links[0])  # a state parameter was minted
        assert summary["registered"] is True
        assert summary["scopes"] == "issues:write"
        assert summary["refreshable"] is True

        # The tokens are on disk, and the retried request really carried one.
        stored = await FileTokenStorage(tmp_path, "fake").get_tokens()
        assert stored is not None and stored.refresh_token == "test-refresh-token"
        assert fake.mcp_auth_header == "Bearer test-access-token"
        assert fake.registration_count == 1
    finally:
        fake.stop()


async def test_the_authorization_request_carries_the_configured_scopes(tmp_path) -> None:
    """A server like Atlassian only issues a refresh token when the
    authorization URL names scopes — so the config's scopes must appear as
    the ``scope`` parameter the provider's browser link is built with."""
    fake = FakeOAuthServer().start()
    try:
        seen: list[str] = []

        async def on_redirect(url: str) -> None:
            seen.append(url)
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            origin = (
                urllib.parse.urlparse(url)
                ._replace(path="", params="", query="", fragment="")
                .geturl()
            )
            landing = (
                f"{fake.registered_redirect}?code=ac-1&state={query['state'][0]}"
                f"&iss={urllib.parse.quote(origin)}"
            )
            await asyncio.to_thread(urllib.request.urlopen, landing)

        spec = remote_spec(fake.url, scope="issues:write offline_access")
        summary = await authorize_server(spec, tmp_path, on_redirect=on_redirect)

        assert summary["registered"] is True
        scopes = urllib.parse.parse_qs(urllib.parse.urlparse(seen[0]).query).get("scope")
        assert scopes == ["issues:write offline_access"]
    finally:
        fake.stop()


async def test_a_second_login_reuses_the_registration(tmp_path) -> None:
    fake = FakeOAuthServer().start()
    try:

        async def on_redirect(url: str) -> None:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            origin = (
                urllib.parse.urlparse(url)
                ._replace(path="", params="", query="", fragment="")
                .geturl()
            )
            landing = (
                f"{fake.registered_redirect}?code=ac-1&state={query['state'][0]}"
                f"&iss={urllib.parse.quote(origin)}"
            )
            await asyncio.to_thread(urllib.request.urlopen, landing)

        spec = remote_spec(fake.url)
        first = await authorize_server(spec, tmp_path, on_redirect=on_redirect)
        assert first["registered"] is True

        # Keep the stored registration but drop the tokens — exactly the
        # state an expired access *and* refresh token leaves behind.
        storage = FileTokenStorage(tmp_path, "fake")
        data = json.loads(storage.path.read_text())
        storage.path.write_text(json.dumps({"client_info": data["client_info"]}))

        second = await authorize_server(spec, tmp_path, on_redirect=on_redirect)
        assert second["registered"] is False
        assert fake.registration_count == 1  # not re-registered
    finally:
        fake.stop()


def query_state(url: str) -> bool:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return bool(query.get("state")) and bool(query.get("code_challenge"))
