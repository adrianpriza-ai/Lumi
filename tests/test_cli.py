"""The command line: argument handling and exit codes.

Exit codes matter here — a shell command's failure has to survive all the way to
``$?`` so the CLI composes in a script.
"""

from __future__ import annotations

import pytest

from lumi.__main__ import _tui_binary, build_parser, cmd_tui_serve, cmd_ui, main
from lumi.paths import UI_BINARY_NAME

# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_every_command_is_registered() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command_name"]
    assert actions
    assert set(actions[0].choices) == {
        "run",
        "chat",
        "ask",
        "tui",
        "tui-serve",
        "shell",
        "search",
        "memory",
        "doctor",
        "config",
    }


def test_no_command_opens_the_ui() -> None:
    """One command, like `vibe`: bare `lumi` is the Rust UI, not a usage error."""
    args = build_parser().parse_args([])
    assert args.command_name is None
    assert args.func is cmd_ui


def test_tui_opens_the_same_ui() -> None:
    assert build_parser().parse_args(["tui"]).func is cmd_ui


def test_tui_serve_is_the_internal_transport() -> None:
    assert build_parser().parse_args(["tui-serve"]).func is cmd_tui_serve


def test_ui_binary_honours_the_override(monkeypatch) -> None:
    monkeypatch.setenv("LUMI_TUI", "/opt/lumi/lumi-tui")
    assert _tui_binary() == "/opt/lumi/lumi-tui"


def test_a_bundled_ui_wins_over_the_checkout(monkeypatch, tmp_path) -> None:
    """`pip install .` puts the binary in `lumi/_bin`, like Vibe's `vibe-rs`."""
    import lumi.paths as paths

    binary = tmp_path / "_bin" / UI_BINARY_NAME
    binary.parent.mkdir()
    binary.write_text("")
    monkeypatch.setattr(paths, "PACKAGE_DIR", tmp_path)
    monkeypatch.delenv("LUMI_TUI", raising=False)
    assert _tui_binary() == str(binary)


def test_shell_keeps_the_command_verbatim() -> None:
    """Flags inside the command must survive: argparse must not eat `ls -la`."""
    args = build_parser().parse_args(["shell", "ls -la --color"])
    # REMAINDER always yields a list; the command is rejoined with spaces.
    assert args.command == ["ls -la --color"]


def test_no_confirm_works_before_the_command() -> None:
    args = build_parser().parse_args(["shell", "--no-confirm", "rm -rf x"])
    assert args.no_confirm is True
    assert args.command == ["rm -rf x"]


def test_no_confirm_works_after_the_command(config) -> None:
    """REMAINDER would otherwise swallow a trailing flag."""
    assert main(["shell", "mkdir -p trailing", "--no-confirm"]) == 0
    assert (config.shell_cwd / "trailing").is_dir()


# --------------------------------------------------------------------------- #
# exit codes
# --------------------------------------------------------------------------- #


def test_shell_passes_through_the_exit_code(config, capsys) -> None:
    assert main(["shell", "exit 7"]) == 7
    assert "exit: 7" in capsys.readouterr().out


def test_shell_returns_zero_on_success(config, capsys) -> None:
    assert main(["shell", "echo ok"]) == 0


def test_shell_refuses_a_dangerous_command(config, capsys) -> None:
    """A hard deny is not an approval prompt: it just fails, loudly."""
    assert main(["shell", "rm -rf /"]) == 1
    assert "Refused" in capsys.readouterr().out


def test_shell_refuses_a_confirm_tier_command_without_the_flag(config, capsys) -> None:
    assert main(["shell", "rm -rf workspace/x"]) == 3


def test_shell_runs_a_confirm_tier_command_with_the_flag(config, capsys) -> None:
    assert main(["shell", "--no-confirm", "mkdir -p from-cli"]) == 0
    assert (config.shell_cwd / "from-cli").is_dir()


def test_shell_without_a_command_is_usage(config, capsys) -> None:
    assert main(["shell"]) == 2


def test_config_key_prints_one_value(config, capsys) -> None:
    assert main(["config", "--key", "llm.model"]) == 0
    assert capsys.readouterr().out.strip() == "test-model"


def test_config_key_reports_the_effective_key_strategy(config, capsys, monkeypatch) -> None:
    """The raw field is empty when the environment supplied the value."""
    config.llm.key_strategy = ""
    monkeypatch.setenv("LUMI__LLM__KEY_STRATEGY", "round_robin")
    assert main(["config", "--key", "llm.key_strategy"]) == 0
    assert capsys.readouterr().out.strip() == "round_robin"


def test_config_key_rejects_nonsense(config, capsys) -> None:
    assert main(["config", "--key", "llm.nonexistent"]) == 2


def test_config_prints_json(config, capsys) -> None:
    import json

    assert main(["config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["llm"]["model"] == "test-model"
    assert payload["root"] == str(config.root)


def test_memory_show_add_and_forget(config, capsys) -> None:
    assert main(["memory"]) == 0
    assert "## Context" in capsys.readouterr().out

    assert main(["memory", "--add", "cli added fact"]) == 0
    assert "remembered" in capsys.readouterr().out

    assert main(["memory", "--forget", "1"]) == 0
    assert "cli added fact" in capsys.readouterr().out


def test_doctor_reports_problems_with_a_nonzero_code(config, monkeypatch, capsys) -> None:
    # The fixture's .env supplies the key, so remove the file too.
    (config.root / ".env").unlink()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["doctor"]) == 1
    assert "blocking problem" in capsys.readouterr().out


def test_doctor_is_happy_with_a_complete_setup(config, capsys) -> None:
    assert main(["doctor"]) == 0
    assert "no blocking problems" in capsys.readouterr().out


def test_doctor_does_not_promise_a_missing_ocr_key_fallback(config, monkeypatch, capsys) -> None:
    monkeypatch.setenv("LUMI__TOOLS__OCR__BASE_URL", "https://api.mistral.ai/v1")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert main(["doctor"]) == 0
    assert "unavailable (no automatic fallback)" in capsys.readouterr().out


def test_doctor_warns_about_a_missing_model_instead_of_failing(config, monkeypatch, capsys) -> None:
    """A custom endpoint with no model just stays out of the tool list."""
    monkeypatch.setenv("LUMI__TOOLS__OCR__BASE_URL", "https://ocr.example/v1")
    monkeypatch.setenv("LUMI__TOOLS__IMAGE_GEN__BASE_URL", "https://img.example/v1")
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[tools.ocr] model is unset" in out and "ocr tool is unavailable" in out
    assert "[tools.image_gen] model is unset" in out


def test_doctor_reports_disabled_ocr(config, monkeypatch, capsys) -> None:
    from lumi.doctor import WARN, Check

    monkeypatch.setenv("LUMI__TOOLS__OCR__ENABLED", "false")
    assert main(["doctor"]) == 0
    assert Check("ocr", WARN, "disabled in config").render() in capsys.readouterr().out


def test_doctor_reports_the_exa_key(config, capsys) -> None:
    """Exa sits in the default provider_order, so the doctor must name its key."""
    assert main(["doctor"]) == 0
    assert "EXA_API_KEY" in capsys.readouterr().out


def test_doctor_reports_a_bundled_ui(config, monkeypatch, tmp_path, capsys) -> None:
    """The wheel's `lumi/_bin/lumi-tui` is the UI an installed `lumi` runs."""
    import lumi.paths as paths

    binary = tmp_path / "_bin" / UI_BINARY_NAME
    binary.parent.mkdir()
    binary.write_text("")
    monkeypatch.setattr(paths, "PACKAGE_DIR", tmp_path)
    monkeypatch.delenv("LUMI_TUI", raising=False)
    assert main(["doctor"]) == 0
    assert f"rust ui — bundled: {binary}" in capsys.readouterr().out


def test_doctor_reports_a_missing_ui(config, monkeypatch, capsys) -> None:
    import lumi.doctor as doctor

    monkeypatch.setattr(doctor, "ui_binary", lambda: None)
    assert main(["doctor"]) == 0
    assert "rust ui — not built" in capsys.readouterr().out


def test_doctor_names_the_running_version(config, capsys) -> None:
    """Which build is running is the first question about a deployment."""
    from lumi import __version__

    assert main(["doctor"]) == 0
    assert __version__ in capsys.readouterr().out


def test_version_flag_prints_the_version(capsys) -> None:
    """``lumi --version`` — the answer has to be on stdout and nothing else."""
    from lumi import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "lumi" in out


def test_ask_reports_a_missing_model(config, monkeypatch, capsys) -> None:
    (config.root / ".env").unlink()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["ask", "hello"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_chat_prints_the_answer_after_an_approval(config, capsys, monkeypatch) -> None:
    """Approving a pending tool call must resume the turn and print its reply."""
    import asyncio

    from conftest import FakeLLM, make_reply

    from lumi import __main__ as cli
    from lumi.agent import Agent
    from lumi.memory import History
    from lumi.personality import Personality
    from lumi.tools import build_registry

    memory = config.build_memory_store()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(
            [
                make_reply(tool_calls=[("c1", "run_shell", {"command": "rm x.txt"})]),
                make_reply("done"),
            ]
        ),
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)
    scripted = iter(["tidy up", "y", "/quit"])

    async def fake_prompt(message: str) -> str:
        return next(scripted)

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0
    assert "done" in capsys.readouterr().out


def _chat_agent(config, replies):
    """An agent wired with scripted replies, ready for ``_chat`` to drive."""
    from conftest import FakeLLM

    from lumi.agent import Agent
    from lumi.memory import History
    from lumi.personality import Personality
    from lumi.tools import build_registry

    memory = config.build_memory_store()
    return Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(replies),
    )


def test_chat_answers_a_whole_approval_card_with_one_yes(config, capsys, monkeypatch) -> None:
    """Two waiting commands are one question, settled in one model turn."""
    import asyncio

    from conftest import make_reply

    from lumi import __main__ as cli

    first = config.shell_cwd / "one.txt"
    second = config.shell_cwd / "two.txt"
    first.write_text("x", encoding="utf-8")
    second.write_text("y", encoding="utf-8")
    agent = _chat_agent(
        config,
        [
            make_reply(
                tool_calls=[
                    ("c1", "run_shell", {"command": f"rm {first}"}),
                    ("c2", "run_shell", {"command": f"rm {second}"}),
                ]
            ),
            make_reply("both gone"),
        ],
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)
    prompts: list[str] = []
    scripted = iter(["tidy up", "y", "/quit"])

    async def fake_prompt(message: str) -> str:
        prompts.append(message)
        return next(scripted)

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0

    assert "both gone" in capsys.readouterr().out
    assert not first.exists() and not second.exists()
    assert prompts.count("  run all? [y/N] ") == 1
    assert "  run it? [y/N] " not in prompts, "one card asks one question"


def test_chat_asks_again_when_the_resumed_turn_needs_approval(config, monkeypatch) -> None:
    """An approval raised by the answer must be asked about, not left queued."""
    import asyncio

    from conftest import make_reply

    from lumi import __main__ as cli

    first = config.shell_cwd / "one.txt"
    second = config.shell_cwd / "two.txt"
    first.write_text("x", encoding="utf-8")
    second.write_text("y", encoding="utf-8")
    agent = _chat_agent(
        config,
        [
            make_reply(tool_calls=[("c1", "run_shell", {"command": f"rm {first}"})]),
            make_reply(tool_calls=[("c2", "run_shell", {"command": f"rm {second}"})]),
            make_reply("both gone"),
        ],
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)
    prompts: list[str] = []
    scripted = iter(["tidy up", "y", "y", "/quit"])

    async def fake_prompt(message: str) -> str:
        prompts.append(message)
        return next(scripted)

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0

    assert prompts.count("  run it? [y/N] ") == 2, "the second approval was offered"
    assert not first.exists() and not second.exists()
    assert agent.conversation("cli").pending == []


def test_the_version_matches_the_project_metadata() -> None:
    """``lumi --version`` reads the package; ``pip`` ships the pyproject one.

    They are bumped by hand, and once they drifted: the CLI and ``/doctor``
    reported 0.4.1 while the project metadata said 0.4.2.
    """
    import tomllib
    from pathlib import Path

    from lumi import __version__

    root = Path(__file__).resolve().parent.parent
    metadata = tomllib.loads(root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == metadata["project"]["version"]


def test_chat_exits_cleanly_when_the_approval_prompt_hits_eof(config, capsys, monkeypatch) -> None:
    """Ctrl-D at a safety prompt means exit, and used to be a traceback."""
    import asyncio

    from conftest import make_reply

    from lumi import __main__ as cli

    doomed = config.shell_cwd / "one.txt"
    doomed.write_text("x", encoding="utf-8")
    agent = _chat_agent(
        config,
        [make_reply(tool_calls=[("c1", "run_shell", {"command": f"rm {doomed}"})])],
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)
    scripted = iter(["tidy up"])

    async def fake_prompt(message: str) -> str:
        try:
            return next(scripted)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0

    assert "bye" in capsys.readouterr().out
    assert doomed.exists(), "nothing ran on the way out"


def test_chat_memory_shares_the_window_with_the_prompt(config, capsys, monkeypatch) -> None:
    """``/memory`` in the REPL must apply the same cap the system prompt uses.

    Called with no limit it printed the whole file while the bot's ``/memory``
    and the prompt both applied ``Agent.memory_limit()``.
    """
    import asyncio

    from conftest import FakeLLM

    from lumi import __main__ as cli
    from lumi.agent import Agent
    from lumi.memory import MANAGED_END, MANAGED_START, History
    from lumi.personality import Personality
    from lumi.tools import build_registry

    # A small window, so an ordinary memory file overflows its share.
    config.llm.context_window = 4_000
    config.llm.context_headroom = 1_000
    facts = "\n".join(f"- [2026-01-01] fact {i:03d} " + "x" * 60 for i in range(60))
    config.memory_file.write_text(
        f"# Memory\n\n## Facts\n\n{MANAGED_START}\n{facts}\n{MANAGED_END}\n",
        encoding="utf-8",
    )

    memory = config.build_memory_store()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([]),
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)
    scripted = iter(["/memory", "/quit"])

    async def fake_prompt(message: str) -> str:
        return next(scripted)

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0
    out = capsys.readouterr().out
    assert "elided" in out  # bounded by the window share, not printed whole
    assert len(out) < len(memory.get(config.default_memory_scope).text)


def test_ask_without_a_question_is_rejected_by_argparse(config) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["ask"])
    assert exit_info.value.code == 2


def test_search_reports_when_no_provider_is_configured(config, monkeypatch, capsys) -> None:
    """With every provider disabled, ``lumi search`` reports unavailability.

    Tavily supports a keyless tier, so just unsetting the keys is no longer
    enough to make the provider unavailable — the test clears the provider
    order through the env override the CLI actually reads.
    """
    monkeypatch.setenv("LUMI__TOOLS__WEB__PROVIDER_ORDER", "")
    assert main(["search", "anything"]) == 2
    assert "unavailable" in capsys.readouterr().err


def test_search_once_closes_web_transports(config) -> None:
    import asyncio

    from lumi import __main__ as cli
    from lumi.tools.base import ToolResult

    class FakeWebTool:
        def __init__(self) -> None:
            self.closed = False

        async def invoke(self, arguments, ctx):
            return ToolResult(text="ok")

        async def aclose(self) -> None:
            self.closed = True

    tool = FakeWebTool()
    result = asyncio.run(cli._search_once(tool, {"action": "search"}, None))

    assert result.ok
    assert tool.closed


# --------------------------------------------------------------------------- #
# chat reasoning & live steps
# --------------------------------------------------------------------------- #


def test_hooks_for_respects_show_reasoning(config) -> None:
    from lumi import __main__ as cli

    config.llm.show_reasoning = True
    hooks = cli._hooks_for(config)
    assert hooks.on_step is cli._note_step
    assert hooks.on_message is cli._note_message

    config.llm.show_reasoning = False
    hooks = cli._hooks_for(config)
    assert hooks.on_step is None
    assert hooks.on_message is cli._note_message


def test_chat_silences_live_steps_when_show_reasoning_is_false(config, capsys, monkeypatch) -> None:
    """Turning show_reasoning off must suppress both live steps and post-turn trace."""
    import asyncio

    from conftest import FakeLLM, make_reply

    from lumi import __main__ as cli
    from lumi.agent import Agent
    from lumi.memory import History
    from lumi.personality import Personality
    from lumi.tools import build_registry

    config.llm.show_reasoning = False

    memory = config.build_memory_store()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(
            [
                make_reply(
                    tool_calls=[("c1", "run_shell", {"command": "echo hello"})],
                    reasoning="thinking about echo",
                ),
                make_reply("done", reasoning="finished"),
            ]
        ),
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)
    scripted = iter(["echo hello", "y", "/quit"])

    async def fake_prompt(message: str) -> str:
        return next(scripted)

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0

    err = capsys.readouterr().err
    assert "[thinking]" not in err
    assert "[run_shell]" not in err
    assert "[thought for" not in err


def test_chat_prints_live_steps_when_show_reasoning_is_true(config, capsys, monkeypatch) -> None:
    """When show_reasoning is on, live step lines and thought summary go to stderr."""
    import asyncio

    from conftest import FakeLLM, make_reply

    from lumi import __main__ as cli
    from lumi.agent import Agent
    from lumi.memory import History
    from lumi.personality import Personality
    from lumi.tools import build_registry

    config.llm.show_reasoning = True

    memory = config.build_memory_store()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(
            [
                make_reply(
                    tool_calls=[("c1", "run_shell", {"command": "echo hello"})],
                    reasoning="thinking about echo",
                ),
                make_reply("done", reasoning="finished"),
            ]
        ),
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)
    scripted = iter(["echo hello", "y", "/quit"])

    async def fake_prompt(message: str) -> str:
        return next(scripted)

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0

    err = capsys.readouterr().err
    assert "[thinking]" in err
    assert "[run_shell]" in err
    assert "[thought for" in err


def test_chat_reasoning_command_in_repl(config, capsys, monkeypatch) -> None:
    """/reasoning reports status and on|off flips config.llm.show_reasoning live."""
    import asyncio

    from conftest import FakeLLM

    from lumi import __main__ as cli
    from lumi.agent import Agent
    from lumi.memory import History
    from lumi.personality import Personality
    from lumi.tools import build_registry

    memory = config.build_memory_store()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([]),
    )
    monkeypatch.setattr(cli, "_build_agent", lambda cfg: agent)

    scripted = iter(
        [
            "/reasoning",
            "/reasoning off",
            "/reasoning on",
            "/reasoning invalid",
            "/quit",
        ]
    )

    async def fake_prompt(message: str) -> str:
        return next(scripted)

    monkeypatch.setattr(cli, "_prompt", fake_prompt)
    assert asyncio.run(cli._chat(config)) == 0

    out = capsys.readouterr().out
    assert "reasoning:" in out
    assert "live steps: on" in out
    assert "live steps off" in out
    assert "live steps on" in out
    assert "usage: /reasoning or /reasoning on|off" in out
