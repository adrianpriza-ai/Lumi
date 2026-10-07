"""The command line: argument handling and exit codes.

Exit codes matter here — a shell command's failure has to survive all the way to
``$?`` so the CLI composes in a script.
"""

from __future__ import annotations

import pytest

from lumi.__main__ import build_parser, main

# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_every_command_is_registered() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command_name"]
    assert actions
    assert set(actions[0].choices) == {
        "run", "chat", "ask", "shell", "search", "memory", "doctor", "config"
    }


def test_no_command_is_an_error() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


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


def test_doctor_reports_the_exa_key(config, capsys) -> None:
    """Exa sits in the default provider_order, so the doctor must name its key."""
    assert main(["doctor"]) == 0
    assert "EXA_API_KEY" in capsys.readouterr().out


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
    from lumi.memory import History, MemoryFile
    from lumi.personality import Personality
    from lumi.tools import build_registry

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
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


def test_chat_memory_shares_the_window_with_the_prompt(config, capsys, monkeypatch) -> None:
    """``/memory`` in the REPL must apply the same cap the system prompt uses.

    Called with no limit it printed the whole file while the bot's ``/memory``
    and the prompt both applied ``Agent.memory_limit()``.
    """
    import asyncio

    from conftest import FakeLLM

    from lumi import __main__ as cli
    from lumi.agent import Agent
    from lumi.memory import MANAGED_END, MANAGED_START, History, MemoryFile
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

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
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
    assert len(out) < len(memory.text)


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
