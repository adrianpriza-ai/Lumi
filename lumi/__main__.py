"""Command line entry point: ``lumi <command>`` or ``python -m lumi <command>``.

Commands
--------
``run``      start the Telegram bot (long polling)
``chat``     talk to the agent in this terminal — same agent, no Telegram needed
``ask``      one-shot question, prints the answer, exits
``shell``    run one command through the safety engine
``search``   web search from the terminal
``memory``   show or edit MEMORY.md
``doctor``   diagnose the environment
``config``   print the resolved configuration

``chat`` is the reason this project is debuggable: the agent loop, tools, memory
and providers are all reachable without a bot token.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from .config import Config, ConfigError, load_config, validate
from .util.log import get_logger, setup_logging
from .util.text import format_error

log = get_logger("lumi.cli")

BANNER = "lumi — /help for commands, /quit to exit, /reset to clear the conversation"


def _config_and_log(args: argparse.Namespace) -> Config:
    config = load_config()
    setup_logging(
        level="DEBUG" if getattr(args, "verbose", False) else config.logging.level,
        log_file=config.log_file,
    )
    return config


def _report_problems(config: Config, *, fatal: bool) -> bool:
    problems = validate(config)
    if not problems:
        return True
    print("configuration problems:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    if fatal:
        print("\nfix these in .env or config.toml, then try again.", file=sys.stderr)
    return False


def _build_agent(config: Config) -> Any:
    from .agent import Agent
    from .memory import History, MemoryFile
    from .personality import Personality

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    personality = Personality.load(config.personality_file)
    history = History(config.history_dir)
    return Agent(config, personality, memory, history)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    from .bot import run

    config = _config_and_log(args)
    if not _report_problems(config, fatal=True):
        return 2
    run(config)
    return 0


async def _ask_once(config: Config, question: str, chat_id: str = "cli") -> int:
    agent = _build_agent(config)
    result = await agent.handle(chat_id, question, source="cli")
    if result.pending:
        print("this needs approval and cannot run unattended:")
        for action in result.pending:
            print(f"  {action.tool}: {action.reason}")
            print(f"  $ {action.preview}")
        return 3
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
    print(result.text or "(no answer)")
    return 0 if result.text else 1


def cmd_ask(args: argparse.Namespace) -> int:
    config = _config_and_log(args)
    if not _report_problems(config, fatal=True):
        return 2
    question = " ".join(args.question).strip()
    if not question:
        print("usage: lumi ask <question>", file=sys.stderr)
        return 2
    return asyncio.run(_ask_once(config, question))


async def _prompt(message: str) -> str:
    """Read a line without blocking the event loop.

    The REPL is the only thing running, but the agent may still have background
    work in flight; running input() in a thread keeps the loop free to cancel it
    on ctrl-c.
    """
    return await asyncio.to_thread(input, message)


async def _chat(config: Config) -> int:
    agent = _build_agent(config)
    print(BANNER)
    print(f"model: {config.llm.model_of()} via {config.llm.base_url_of()}")
    print(f"tools: {', '.join(agent.registry.names()) or 'none'}\n")

    while True:
        try:
            line = (await _prompt("you > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        lowered = line.lower()
        if lowered in {"/quit", "/exit", ":q"}:
            print("bye")
            return 0
        if lowered == "/reset":
            agent.reset("cli")
            print("conversation cleared\n")
            continue
        if lowered in {"/help", "/?"}:
            print(BANNER)
            print("/quit  /reset\n")
            continue
        if lowered == "/tools":
            print(agent.registry.describe(), "\n")
            continue
        if lowered.startswith("/memory"):
            print(agent.memory.for_prompt(), "\n")
            continue
        if lowered == "/personality":
            print(agent.personality.text, "\n")
            continue

        print("lumi > ", end="", flush=True)
        try:
            result = await agent.handle("cli", line, source="cli")
        except Exception as exc:  # noqa: BLE001
            print(f"\ni broke: {format_error(exc)}\n")
            continue

        if result.pending:
            print("\n".join(f"  needs approval — {a.reason}: {a.preview}" for a in result.pending))
            answer = (await _prompt("  run it? [y/N] ")).strip().lower()
            for action in list(result.pending):
                approved = answer in {"y", "yes"}
                print("  working…", flush=True)
                result = await agent.resolve("cli", action.id, approved, source="cli")
        elif result.text:
            print(result.text)
        if result.error:
            print(f"[{result.error}]", file=sys.stderr)
        print()


def cmd_chat(args: argparse.Namespace) -> int:
    config = _config_and_log(args)
    if not _report_problems(config, fatal=True):
        return 2
    return asyncio.run(_chat(config))


def cmd_shell(args: argparse.Namespace) -> int:
    config = _config_and_log(args)
    from .tools.base import NeedsApproval, ToolContext
    from .tools.shell import ShellTool

    command = args.command if isinstance(args.command, str) else " ".join(args.command)
    if not command.strip():
        print("usage: lumi shell <command>", file=sys.stderr)
        return 2

    tool = ShellTool(config)
    # Default is confirmed=False, so the safety engine is respected. --no-confirm
    # is the explicit opt-out for when you are running it yourself anyway.
    ctx = ToolContext(cwd=config.shell_cwd, source="cli", confirmed=bool(args.no_confirm))

    try:
        result = asyncio.run(tool.invoke({"command": command}, ctx))
    except NeedsApproval as exc:
        print(f"refused without approval: {exc.reason}\n\n  $ {command}\n", file=sys.stderr)
        print("re-run with --no-confirm if you are sure.", file=sys.stderr)
        return 3
    print(result.text)
    if result.data.get("timed_out"):
        return 124
    code = result.data.get("exit_code")
    if code is not None:
        return int(code)
    # A refusal or a crash has no exit code of its own; still report failure so
    # this composes in a script.
    return 0 if result.ok else 1


def cmd_search(args: argparse.Namespace) -> int:
    config = _config_and_log(args)
    from .tools.base import ToolContext
    from .tools.web import WebTool

    tool = WebTool(config)
    ok, reason = tool.available()
    if not ok:
        print(f"web search unavailable: {reason}", file=sys.stderr)
        return 2
    ctx = ToolContext(cwd=config.shell_cwd, source="cli")
    arguments: dict[str, Any] = {"action": "search", "query": " ".join(args.query)}
    if args.limit:
        arguments["max_results"] = args.limit
    result = asyncio.run(tool.invoke(arguments, ctx))
    print(result.text)
    return 0 if result.ok else 1


def cmd_memory(args: argparse.Namespace) -> int:
    config = _config_and_log(args)
    from .memory import MemoryFile

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()

    if args.add:
        print(memory.remember(" ".join(args.add), source="cli"))
        return 0
    if args.forget:
        removed = memory.forget(args.forget)
        print("\n".join(removed) if removed else "nothing to forget")
        return 0
    print(memory.text)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config_and_log(args)
    from .doctor import platform_line, run_checks

    print(f"lumi doctor — {platform_line()}\n")
    for check in run_checks(config):
        print(f"  {check.render()}")
    problems = validate(config)
    print()
    if problems:
        print(f"{len(problems)} blocking problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("no blocking problems. run `lumi chat` to try it, or `lumi run` for Telegram.")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict, is_dataclass

    config = _config_and_log(args)

    def encode(value: Any) -> Any:
        if hasattr(value, "__fspath__"):
            return str(value)
        if is_dataclass(value):
            return {k: encode(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value

    if args.key:
        node: Any = config
        for part in args.key.split("."):
            node = getattr(node, part, None)
            if node is None:
                print(f"no such config key: {args.key}", file=sys.stderr)
                return 2
        # Some keys are derived from the environment; print what is actually in
        # effect rather than the raw field, which is often empty.
        resolved = {
            "llm.base_url": config.llm.base_url_of,
            "llm.model": config.llm.model_of,
            "llm.key_strategy": config.llm.strategy_of,
        }.get(args.key)
        print(resolved() if resolved else node)
        return 0

    print(json.dumps(encode(config), indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumi",
        description="A portable, modular Telegram AI agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "config lives next to this package: config.toml for behaviour, .env for secrets.\n"
            "nothing is read from or written to your home directory unless you ask for it."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command_name", required=True)

    run_parser = sub.add_parser("run", help="start the Telegram bot")
    run_parser.set_defaults(func=cmd_run)

    chat_parser = sub.add_parser("chat", help="talk to the agent in this terminal")
    chat_parser.set_defaults(func=cmd_chat)

    ask_parser = sub.add_parser("ask", help="ask one question and exit")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.set_defaults(func=cmd_ask)

    shell_parser = sub.add_parser("shell", help="run one command through the safety engine")
    shell_parser.add_argument("command", nargs=argparse.REMAINDER)
    shell_parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="treat the command as pre-approved (skips the confirm tier)",
    )
    shell_parser.set_defaults(func=cmd_shell)

    search_parser = sub.add_parser("search", help="search the web")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("--limit", type=int, default=0)
    search_parser.set_defaults(func=cmd_search)

    memory_parser = sub.add_parser("memory", help="show or edit MEMORY.md")
    memory_parser.add_argument("--add", nargs="+", help="save a fact")
    memory_parser.add_argument("--forget", type=int, metavar="N", help="drop the newest N facts")
    memory_parser.set_defaults(func=cmd_memory)

    doctor_parser = sub.add_parser("doctor", help="diagnose the environment")
    doctor_parser.set_defaults(func=cmd_doctor)

    config_parser = sub.add_parser("config", help="print the resolved configuration")
    config_parser.add_argument("--key", help="print one value, e.g. llm.model")
    config_parser.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `lumi shell <command>` uses argparse.REMAINDER so the command reaches the
    # shell untouched — but REMAINDER also swallows anything after the command,
    # including our own flags. Pull them out first so the order never matters.
    no_confirm = "--no-confirm" in argv
    argv = [arg for arg in argv if arg != "--no-confirm"]

    parser = build_parser()
    args = parser.parse_args(argv)
    if no_confirm:
        args.no_confirm = True
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:  # noqa: BLE001
        log.debug("unhandled CLI error", exc_info=True)
        print(f"error: {format_error(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
