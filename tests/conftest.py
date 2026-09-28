"""Shared fixtures.

Every test runs against a throwaway project directory so nothing touches the real
MEMORY.md, workspace, or history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumi.config import load_config  # noqa: E402


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal Lumi project: config, personality, memory, writable workspace."""
    (tmp_path / "PERSONALITY.md").write_text("You are Lumi. Terse.\n", encoding="utf-8")
    (tmp_path / "MEMORY.md").write_text(
        "# Memory\n\n## Facts\n\n"
        "<!-- lumi:managed:start -->\n"
        "<!-- managed -->\n"
        "<!-- lumi:managed:end -->\n\n"
        "## Context\n\nnotes\n",
        encoding="utf-8",
    )
    (tmp_path / "config.toml").write_text(
        "[bot]\nrequire_owner = false\n"
        "[llm]\nmodel = 'test-model'\n"
        "[tools.shell]\ncwd = 'workspace'\ntimeout_seconds = 5\n",
        encoding="utf-8",
    )
    (tmp_path / "workspace").mkdir()
    (tmp_path / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=test-token\nTELEGRAM_OWNER_ID=42\nOPENAI_API_KEY=test-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LUMI_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "42")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for leaked in ("TAVILY_API_KEY", "FIRECRAWL_API_KEY", "LUMI_MCP_URL"):
        monkeypatch.delenv(leaked, raising=False)
    return tmp_path


@pytest.fixture
def config(project: Path):
    return load_config(project)


class FakeLLM:
    """Scripted LLM client.

    Feed it a list of :class:`ScriptedReply` and it returns them in order,
    recording every messages array it was handed so tests can assert on the
    conversation the agent actually built.
    """

    def __init__(self, replies: list) -> None:
        from lumi.llm.base import LLMReply

        self.replies = [
            r if isinstance(r, LLMReply) else LLMReply(**r) for r in replies
        ]
        self.calls: list[list[dict]] = []
        self.index = 0

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(m) for m in messages])
        if self.index >= len(self.replies):
            raise AssertionError(f"LLM called {self.index + 1} times, only {len(self.replies)} scripted")
        reply = self.replies[self.index]
        self.index += 1
        return reply

    def describe(self) -> str:
        return "fake"


def make_reply(
    text: str = "",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    *,
    reasoning: str = "",
    reasoning_tokens: int = 0,
):
    """Build an LLMReply with tool calls given as (id, name, arguments) tuples."""
    from lumi.llm.base import LLMReply, ToolCall

    calls = [
        ToolCall(id=cid, name=name, arguments=args) for cid, name, args in (tool_calls or [])
    ]
    return LLMReply(
        text=text,
        reasoning=reasoning,
        tool_calls=calls,
        finish_reason=finish_reason if not calls else "tool_calls",
        usage={"reasoning": reasoning_tokens} if reasoning_tokens else {},
    )
