"""Shared fixtures.

Every test runs against a throwaway project directory so nothing touches the real
MEMORY.md, workspace, or history.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumi.config import load_config  # noqa: E402
from lumi.llm.base import DELTA_TEXT, DELTA_THINKING  # noqa: E402
from lumi.tools.safety import SESSION_START  # noqa: E402


def backdate(path: Path, seconds: float = 60.0) -> Path:
    """Age *path* so it looks older than this process.

    The files tool asks the owner before it writes, appends to or edits a file
    that appeared or changed during the session. Tests whose subject is
    something else put their fixture files back in the past instead of paying
    that gate.
    """
    stamp = SESSION_START - seconds
    os.utime(path, (stamp, stamp))
    return path


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
    # A throwaway HOME: the real one holds ~/.agents/mcp.json and the
    # user-level skill directory (~/.agents/skills), and neither may leak into
    # a test — a stray user skill would show up in the system prompt and change
    # what the context tests are measuring. Tests that exercise those files
    # re-point HOME at their own directory.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The user config home (~/.config/lumi) derives from XDG_CONFIG_HOME when
    # set; pin it under the throwaway HOME so the real user layer — and an
    # exported XDG_CONFIG_HOME — can never leak into a test.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.delenv("LUMI_USER_CONFIG", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "42")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # The real ~/.agents/mcp.json must never leak into tests; the MCP tests
    # that exercise the user-level file re-enable it against a throwaway HOME.
    monkeypatch.setenv("LUMI__TOOLS__WEB__PROVIDERS__MCP__USER_CONFIG_FILE", "")
    for leaked in (
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
        "EXA_API_KEY",
        "KEENABLE_API_KEY",
        "JINA_API_KEY",
        "LUMI_MCP_URL",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_VISION_MODEL",
    ):
        monkeypatch.delenv(leaked, raising=False)
    return tmp_path


@pytest.fixture
def config(project: Path):
    return load_config(project)


def make_store(config):
    """The per-chat memory store, wired the way the app wires it.

    Scoped files are created on demand under ``data/memory/<scope>/MEMORY.md``;
    the legacy root MEMORY.md seeds the default scope (the owner's) on first
    use, exactly as a deployed bot migrating would.
    """
    return config.build_memory_store()


class FakeLLM:
    """Scripted LLM client.

    Feed it a list of :class:`ScriptedReply` and it returns them in order,
    recording every messages array it was handed so tests can assert on the
    conversation the agent actually built.

    With ``stream=True`` it plays a streaming client too: the reasoning trace
    and then the answer (in two pieces) go to ``on_delta`` before the reply is
    returned — the shape the TUI's ``delta``/``thinking_delta`` events are
    tested against, with no network and no SDK.
    """

    def __init__(self, replies: list, *, stream: bool = False) -> None:
        from lumi.llm.base import LLMReply

        self.replies = [r if isinstance(r, LLMReply) else LLMReply(**r) for r in replies]
        self.calls: list[list[dict]] = []
        self.index = 0
        self.stream = stream

    async def complete(self, messages, *, tools=None, on_delta=None):
        self.calls.append([dict(m) for m in messages])
        if self.index >= len(self.replies):
            raise AssertionError(
                f"LLM called {self.index + 1} times, only {len(self.replies)} scripted"
            )
        reply = self.replies[self.index]
        self.index += 1
        if self.stream and on_delta is not None:
            if reply.reasoning:
                await on_delta(DELTA_THINKING, reply.reasoning)
            if reply.text:
                middle = len(reply.text) // 2
                for piece in (reply.text[:middle], reply.text[middle:]):
                    if piece:
                        await on_delta(DELTA_TEXT, piece)
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

    calls = [ToolCall(id=cid, name=name, arguments=args) for cid, name, args in (tool_calls or [])]
    return LLMReply(
        text=text,
        reasoning=reasoning,
        tool_calls=calls,
        finish_reason=finish_reason if not calls else "tool_calls",
        usage={"reasoning": reasoning_tokens} if reasoning_tokens else {},
    )


def mcp_error(cls: type, error) -> BaseException:
    """An instance of the MCP error class *cls* wrapping *error*.

    1.x takes the ErrorData directly — ``McpError(error)``; 2.x takes
    ``(code, message)`` and offers ``from_error_data`` to build from one,
    so the direct shape raises ``TypeError`` there. Using the classmethod
    when it exists keeps tests' construction on both major versions.
    """
    from_data = getattr(cls, "from_error_data", None)
    if from_data is not None:
        return from_data(error)
    return cls(error)


#: Why the Code Mode sandbox tests skip rather than fail when the host cannot
#: run them. One definition, so every file that needs the sandbox says the
#: same thing — a test that *failed* here used to read as a broken sandbox on
#: CI, when the runner simply had no bubblewrap and no Python under /usr.
SANDBOX_SKIP_REASON = (
    "Code Mode integration tests require Linux, bubblewrap (bwrap), and a Python runtime under /usr"
)


def sandbox_unavailable() -> bool:
    """True when :func:`lumi.tools.mcp.sandbox._bubblewrap_command` would refuse.

    Mirrors the check in ``test_mcp_sandbox.py``: the sandbox is available
    exactly when it can build a bubblewrap command for a scratch directory.
    """
    from pathlib import Path

    from lumi.tools.mcp import sandbox

    return sandbox._bubblewrap_command(Path("/tmp")) is None


def skip_without_sandbox():
    """``pytest.mark.skipif`` for a test (or a module, via ``pytestmark``) that
    needs the real Code Mode sandbox."""
    return pytest.mark.skipif(sandbox_unavailable(), reason=SANDBOX_SKIP_REASON)


# --------------------------------------------------------------------------- #
# CI enforcement: the sandbox tests must RUN where the runner can run them
# --------------------------------------------------------------------------- #

#: Set by the CI workflow on a runner that has bubblewrap and a /usr Python.
#: There, a sandbox skip is a failure — the test did not happen.
SANDBOX_REQUIRED_ENV = "LUMI_CI_SANDBOX_REQUIRED"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """With ``LUMI_CI_SANDBOX_REQUIRED=1``, any sandbox skip fails the run.

    The point of that runner is to exercise the Code Mode sandbox. If bubblewrap
    went missing, or the interpreter moved out of /usr, every sandbox test would
    skip "for a good reason" and the build would go green while testing nothing.
    This hook makes that impossible: the run ends non-zero, and the message says
    which prerequisite is unmet. The terminal reporter's collected outcomes are
    the source — they carry each skip's reason.
    """
    if (os.environ.get(SANDBOX_REQUIRED_ENV) or "").strip().lower() not in ("1", "true", "yes"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    sandbox_skips = [
        rep
        for rep in getattr(reporter, "stats", {}).get("skipped", [])
        if SANDBOX_SKIP_REASON in str(getattr(rep, "longrepr", ""))
    ]
    if sandbox_skips:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter.write_sep("=", f"FAILED: {len(sandbox_skips)} sandbox test(s) SKIPPED", red=True)
        reporter.write_line(
            f"This runner must run the Code Mode sandbox ({SANDBOX_REQUIRED_ENV}=1), "
            f"but {len(sandbox_skips)} test(s) skipped: {SANDBOX_SKIP_REASON}.\n"
            "The runner is not set up — install bubblewrap and run pytest on a "
            "Python that lives under /usr.",
            red=True,
        )
