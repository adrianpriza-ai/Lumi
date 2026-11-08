"""The image_gen tool: endpoint dispatch, project save, artifact delivery."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from lumi.config import ImageGenConfig, load_config
from lumi.llm.base import LLMError
from lumi.llm.image_api import ImageGenClient
from lumi.tools import build_registry
from lumi.tools.base import ToolContext, ToolError
from lumi.tools.imagegen import ImageGenTool, _stem

#: A valid 1x1 transparent PNG.
PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)


class StubClient:
    """Stands in for the HTTP client; records calls, replays results."""

    def __init__(self, *results: tuple[str, bytes] | LLMError) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    async def generate(self, prompt: str, size: str = "") -> tuple[str, bytes]:
        self.calls.append((prompt, size))
        result = self.results.pop(0)
        if isinstance(result, LLMError):
            raise result
        return result


def make_tool(tmp_path: Path, toml: str = "") -> ImageGenTool:
    (tmp_path / "PERSONALITY.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        "[bot]\nrequire_owner = false\n[llm]\nmodel = 'test-model'\n" + toml,
        encoding="utf-8",
    )
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return ImageGenTool(load_config(tmp_path))


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_unavailable_without_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ok, reason = make_tool(tmp_path).available()
    assert not ok
    assert "OPENAI_API_KEY" in reason


def test_available_with_key(key: None, tmp_path: Path) -> None:
    assert make_tool(tmp_path).available() == (True, "")


def test_unavailable_without_httpx(
    key: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The client imports httpx per request; say so instead of failing on every prompt."""
    import builtins

    real_import = builtins.__import__

    def no_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)

    ok, reason = make_tool(tmp_path).available()

    assert not ok
    assert "httpx" in reason


def test_custom_endpoint_without_key_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A self-hosted endpoint may take no auth; keyless is a real configuration."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tool = make_tool(
        tmp_path,
        toml='[tools.image_gen]\nprovider = "custom"\nbase_url = "http://localhost:8787/v1"\n',
    )
    assert tool.available() == (True, "")


async def test_a_keyless_custom_endpoint_sends_no_auth_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import httpx

    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"data": [{"b64_json": PNG_1PX}]}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["headers"] = kwargs.get("headers")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json):
            captured["url"] = url
            return FakeResponse()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    tool = make_tool(
        tmp_path,
        toml='[tools.image_gen]\nprovider = "custom"\nbase_url = "http://localhost:8787/v1"\n',
    )
    tool._api = ImageGenClient(tool._settings)

    result = await tool.invoke({"prompt": "a red cube"}, ToolContext())

    assert result.ok
    assert "Authorization" not in (captured["headers"] or {})
    assert captured["url"] == "http://localhost:8787/v1/images/generations"


async def test_a_hosted_endpoint_without_key_is_named_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tool = make_tool(tmp_path)
    tool._api = ImageGenClient(tool._settings)

    with pytest.raises(ToolError, match="OPENAI_API_KEY is not set"):
        await tool.invoke({"prompt": "x"}, ToolContext())


async def test_generate_saves_png_and_registers_artifact(key: None, tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    tool._api = StubClient(("image/png", base64.b64decode(PNG_1PX)))

    result = await tool.invoke({"prompt": "a red cube on white"}, ToolContext())

    assert result.ok
    image = tmp_path / result.data["path"]
    assert image.read_bytes() == base64.b64decode(PNG_1PX)
    assert image.suffix == ".png"
    assert image.parent == (tmp_path / "workspace" / "images").resolve()
    (artifact,) = result.artifacts
    assert artifact.path == result.data["path"]
    assert artifact.absolute.is_file()


async def test_size_reaches_the_client(key: None, tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    stub = StubClient(("image/png", base64.b64decode(PNG_1PX)))
    tool._api = stub

    await tool.invoke({"prompt": "wide", "size": "1792x1024"}, ToolContext())

    assert stub.calls == [("wide", "1792x1024")]


async def test_http_error_becomes_tool_error(key: None, tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    tool._api = StubClient(LLMError("HTTP 400: unknown size"))

    with pytest.raises(ToolError, match="unknown size"):
        await tool.invoke({"prompt": "x"}, ToolContext())


async def test_unknown_mime_is_refused(key: None, tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    tool._api = StubClient(("image/x-weird", b"bits"))

    with pytest.raises(ToolError, match="image/x-weird"):
        await tool.invoke({"prompt": "x"}, ToolContext())


async def test_empty_prompt_fails(key: None, tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    tool._api = StubClient()

    result = await tool.invoke({"prompt": "  "}, ToolContext())

    assert not result.ok


def test_custom_provider_needs_base_url() -> None:
    assert "base_url" in ImageGenConfig(provider="custom", base_url="").endpoint_problem()


def test_custom_provider_pointed_at_openai_default_is_caught() -> None:
    assert "base_url" in ImageGenConfig(provider="custom").endpoint_problem()


def test_unknown_provider_is_rejected() -> None:
    assert "unknown tools.image_gen.provider" in ImageGenConfig(provider="dalle").endpoint_problem()


def test_openai_provider_keeps_default_base_url() -> None:
    assert ImageGenConfig().endpoint_problem() == ""


def test_stem_is_safe_and_bounded() -> None:
    stem = _stem("A very long prompt: with punctuation!! and Ünïcödé " * 5)
    assert len(stem) <= 80
    assert "/" not in stem and "!" not in stem
    assert _stem("???").endswith("image")  # timestamp-prefixed fallback


def test_registry_builds_image_gen(key: None, tmp_path: Path) -> None:
    config = load_config(tmp_path)
    registry = build_registry(config, memory=None)  # type: ignore[arg-type]
    assert "image_gen" in registry


def test_registry_respects_enabled_list(key: None, tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        "[bot]\nrequire_owner = false\n[llm]\nmodel = 'test-model'\n"
        '[tools]\nenabled = ["image_gen"]\n',
        encoding="utf-8",
    )
    registry = build_registry(load_config(tmp_path), memory=None)  # type: ignore[arg-type]
    assert registry.names() == ["image_gen"]
