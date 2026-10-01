"""The ``run_shell`` tool: execute a command, with the guardrails in place.

Three layers, in order:

1. :mod:`lumi.tools.safety` decides allow / confirm / deny.
2. This module runs the command with a scrubbed environment, a hard timeout, a
   killed process group, and a capped output buffer.
3. The caller (:class:`lumi.agent.Agent`) turns a ``confirm`` into a Telegram
   prompt and re-invokes with ``ctx.confirmed = True``.

Environment scrubbing is the part people forget: the process would otherwise
inherit ``OPENAI_API_KEY`` and ``TELEGRAM_BOT_TOKEN``, and any command the model
runs could post them to the internet. Everything that looks like a secret is
stripped before ``execve``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from pathlib import Path
from typing import Any

from ..config import Config
from ..paths import is_within
from ..util.log import get_logger
from ..util.text import truncate
from . import safety
from .base import NeedsApproval, Tool, ToolContext, ToolError, ToolResult

log = get_logger(__name__)

#: Environment variables that pass through to the child process.
ENV_PASSTHROUGH: frozenset[str] = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TERM",
        "SHELL",
        "USER",
        "LOGNAME",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "HOME",
        "TMPDIR",
        # Build tooling that legitimately needs to know where it is.
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "NODE_PATH",
        "PYTHONPATH",
        "GOPATH",
        "GOROOT",
    }
)

#: Anything matching this is stripped, whatever its name.
SECRET_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION|COOKIE)", re.IGNORECASE
)

#: Hard ceiling regardless of config, so a bad config cannot wedge the bot.
MAX_TIMEOUT_SECONDS = 600


class ShellTool(Tool):
    name = "run_shell"
    description = """
Run a shell command in the project directory and return its output.

Use this for real work: inspecting the repository, running tests, checking
whether a file exists, building something. The working directory is the
project's workspace folder.

Rules:
- `cwd` is optional and must stay inside the project directory.
- Commands that destroy data, change permissions, or write outside the project
  are refused outright, and commands that merely change state need the owner's
  approval first. If you get "needs approval", say what the command does and
  why, and let the owner decide — do not try to rephrase it into something
  sneakier.
- The working directory is judged after any `cd`, so `cd .. && rm -rf x` is
  checked against the directory it would actually delete from.
- Output is sent whole. If a command produces a lot of it, the old results are
  what gets dropped from the context later, not this one — so redirect to a
  file and read that back when you only need part of it.
- The environment is scrubbed: API keys are not visible to the command.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command line to run, e.g. `git status --short`.",
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Optional working directory relative to the project root, "
                    "e.g. `.` or `backend`. Must stay inside the project."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional override of the configured timeout (max 600).",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_SECONDS,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.shell_config = config.tools.shell
        self.root = config.root
        self.cwd = config.shell_cwd
        # The write boundary is configurable and defaults to the whole project,
        # which is the behaviour people expect from a coding bot that has to be
        # able to edit the code it is working on.
        self.workspace = (
            self.shell_config.writable_roots()[0] if self.shell_config.writable else config.root
        )
        # The working directory is part of the layout, not something the user has
        # to create by hand. A fresh clone should be runnable immediately.
        try:
            self.cwd.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("could not create the shell working directory %s: %s", self.cwd, exc)

    # -- plumbing ---------------------------------------------------------- #

    def available(self) -> tuple[bool, str]:
        if not self.shell_config.enabled:
            return False, "disabled in config (tools.shell.enabled = false)"
        return True, ""

    def _resolve_cwd(self, requested: str | None) -> Path:
        if not requested or requested in {".", "./"}:
            return self.cwd
        candidate = (self.root / requested).resolve()
        if not is_within(candidate, self.root):
            raise ToolError(f"cwd {requested!r} is outside the project directory")
        if not candidate.is_dir():
            raise ToolError(f"cwd {requested!r} does not exist")
        return candidate

    def _timeout(self, requested: Any) -> int:
        configured = self.shell_config.timeout_seconds
        if requested is None:
            return min(configured, MAX_TIMEOUT_SECONDS)
        try:
            return max(1, min(int(requested), MAX_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            return min(configured, MAX_TIMEOUT_SECONDS)

    def _build_env(self) -> dict[str, str]:
        """A minimal, secret-free environment for the child process."""
        env = {key: value for key, value in os.environ.items() if key in ENV_PASSTHROUGH}
        stripped = [k for k in os.environ if k not in ENV_PASSTHROUGH and SECRET_PATTERN.search(k)]
        if stripped:
            log.debug("scrubbed %d secret-looking variables from the child env", len(stripped))

        if self.shell_config.home == "inherit":
            env.setdefault("HOME", str(Path.home()))
        else:
            # Pinning HOME to the project keeps every stray dotfile the command
            # writes inside the repo, which is the whole portability story.
            env["HOME"] = str(self.root)
        env.setdefault("TERM", "dumb")
        env["LUMI_PROJECT_ROOT"] = str(self.root)
        env["LUMI_SHELL_CWD"] = str(self.cwd)
        return env

    def classify(self, command: str, cwd: Path) -> safety.Verdict:
        return safety.classify(
            command,
            cwd=cwd,
            project_root=self.root,
            home=Path.home(),
            # Writes are confined to the workspace, not merely to the project:
            # the rest of the repo is Lumi's own source and the owner's config.
            write_root=self.workspace,
            ask_before_risky=self.shell_config.ask_before_risky,
            extra_deny=self.shell_config.extra_deny,
            extra_confirm=self.shell_config.extra_confirm,
        )

    # -- execution --------------------------------------------------------- #

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(arguments.get("command", "")).strip()
        if not command:
            raise ToolError("command is empty")

        cwd = self._resolve_cwd(arguments.get("cwd"))
        timeout = self._timeout(arguments.get("timeout_seconds"))
        verdict = self.classify(command, cwd)

        log.info(
            "shell verdict=%s cwd=%s cmd=%s", verdict.tier.value, cwd, truncate(command, 200, "…")
        )

        if verdict.blocked:
            # Deliberately does not echo the rule name: no need to teach a model
            # how to phrase a command so it slips past the filter.
            return ToolResult.failure(
                f"Refused: {verdict.reason}. "
                "Pick a different approach, or ask the owner to run it themselves."
            )

        if verdict.needs_approval and not ctx.confirmed:
            raise NeedsApproval(
                tool=self.name,
                arguments=dict(arguments),
                reason=verdict.reason,
                preview=command,
            )

        started = asyncio.get_running_loop().time()
        code, stdout, stderr, timed_out = await self._run(command, cwd, timeout)
        elapsed = asyncio.get_running_loop().time() - started

        return self._render(command, cwd, code, stdout, stderr, timed_out, elapsed, ctx)

    async def _run(
        self,
        command: str,
        cwd: Path,
        timeout: int,  # noqa: ASYNC109 - a configured budget, not a per-call API knob
    ) -> tuple[int, str, str, bool]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=self._build_env(),
                # Own process group, so a timeout can kill the whole tree rather
                # than leaving orphaned children behind.
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolError(f"could not start the command: {exc}") from exc

        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode or 0, _decode(raw_out), _decode(raw_err), False
        except TimeoutError:
            log.warning(
                "command timed out after %ss, killing the process group: %s", timeout, command
            )
            _kill_group(proc)
            try:
                raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=5)
            except TimeoutError:  # pragma: no cover - zombie child
                raw_out, raw_err = b"", b""
            return -1, _decode(raw_out), _decode(raw_err), True

    def _render(
        self,
        command: str,
        cwd: Path,
        code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
        elapsed: float,
        ctx: ToolContext,
    ) -> ToolResult:
        cap = self.shell_config.max_output_chars
        lines = [
            f"$ {command}",
            f"cwd: {cwd.relative_to(self.root) if is_within(cwd, self.root) else cwd}",
        ]

        if timed_out:
            lines.append(f"TIMED OUT after {self._timeout(None)}s — the command was killed.")
            code = -1
        else:
            lines.append(f"exit: {code}  ({elapsed:.2f}s)")

        if stdout.strip():
            lines += ["--- stdout ---", truncate(stdout.strip(), cap)]
        if stderr.strip():
            lines += ["--- stderr ---", truncate(stderr.strip(), cap // 2 if cap > 0 else 0)]
        if not stdout.strip() and not stderr.strip() and not timed_out:
            lines.append("(no output)")

        text = "\n".join(lines)
        status = "ok" if code == 0 else "error"
        return ToolResult(
            text=text,
            ok=code == 0,
            data={
                "command": command,
                "exit_code": code,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
                "cwd": str(cwd),
            },
            summary=f"{status} exit={code} {truncate(command, 60, '…')}",
        )

    def summary_line(self) -> str:
        return f"`{self.name}` — run a shell command in `{self.cwd.relative_to(self.root)}`"


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the whole process group; fall back to the process itself."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


__all__ = ["ShellTool", "MAX_TIMEOUT_SECONDS", "ENV_PASSTHROUGH", "SECRET_PATTERN"]
