"""The agent loop: prompt in, tools called, answer out.

One user message becomes a bounded conversation with the model:

1. build the system prompt from ``PERSONALITY.md`` + ``MEMORY.md`` + live facts,
2. replay recent history so the bot remembers what you said ten messages ago,
3. call the model, execute whatever tools it asks for, feed the results back,
4. repeat until it produces prose or hits ``llm.max_tool_iterations``.

Approval is part of the loop, not a wrapper around it. A tool that wants
confirmation raises :class:`~lumi.tools.base.NeedsApproval`; the loop records the
call as pending and stops. The chat renders a Confirm/Cancel pair, and
:meth:`Agent.resolve` runs the call for real (or tells the model it was refused)
and picks the conversation back up.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import Config
from .llm import LLMClient, ToolCall, build_llm
from .llm.base import LLMError, LLMReply
from .memory import History, MemoryFile
from .paths import relative_to_root
from .personality import Personality
from .tools import NeedsApproval, ToolContext, ToolRegistry, build_registry
from .util.log import get_logger

log = get_logger(__name__)

#: Cap on messages held per conversation, before trimming on a user boundary.
MAX_MESSAGES = 60

RUNTIME_PREAMBLE = """\
## Runtime

These are facts about right now, not instructions. Use them, do not restate them.

- Current time: {now} ({timezone})
- Project root: {root}
- Shell working directory: {cwd}
- Today is {day}, {date}.
"""

TOOL_PREAMBLE = """\
## Tools

{tools}

When you use a tool:

- A call can come back as "needs approval". That means the owner has to tap
  Confirm. Say plainly what the command does and why you want it, then stop.
  Never rephrase the same dangerous command to get around the check.
- Report what actually happened. If a command failed, show the error rather
  than describing what you think it probably did.
- Prefer reading a file over guessing its contents, and prefer the `files` tool
  over shell redirection when writing.
- Cite web results by their bracketed number, e.g. "as of today [2]".
- Save durable facts about the owner with the `memory` tool, sparingly.
"""


@dataclass(slots=True)
class PendingAction:
    """A tool call waiting on the owner's yes or no."""

    id: str
    tool: str
    arguments: dict[str, Any]
    reason: str
    preview: str

    def to_tool_call(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.tool, arguments=self.arguments)

    def one_line(self) -> str:
        if self.tool == "run_shell":
            return f"`{self.preview}`"
        target = self.arguments.get("path") or self.arguments.get("query") or ""
        return f"`{self.tool}` on {target}" if target else f"`{self.tool}`"


@dataclass(slots=True)
class TurnResult:
    """What one turn produced."""

    text: str = ""
    pending: list[PendingAction] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    error: str = ""
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def needs_approval(self) -> bool:
        return bool(self.pending)


@dataclass(slots=True)
class Conversation:
    """Per-chat state: the live message list plus the pending queue."""

    chat_id: int | str
    messages: list[dict[str, Any]] = field(default_factory=list)
    session: str = ""
    pending: list[PendingAction] = field(default_factory=list)
    seeded: bool = False


class Agent:
    def __init__(
        self,
        config: Config,
        personality: Personality,
        memory: MemoryFile,
        history: History,
        registry: ToolRegistry | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.personality = personality
        self.memory = memory
        self.history = history
        self.registry = registry if registry is not None else build_registry(config, memory)
        self.llm = llm if llm is not None else build_llm(config)
        self._conversations: dict[int | str, Conversation] = {}

    # -- prompt ------------------------------------------------------------ #

    def system_prompt(self) -> str:
        now = datetime.now(UTC)
        parts = [
            self.personality.text,
            RUNTIME_PREAMBLE.format(
                now=now.strftime("%Y-%m-%d %H:%M:%S"),
                timezone=now.tzname() or "UTC",
                root=self.config.root,
                cwd=relative_to_root(self.config.shell_cwd),
                day=now.strftime("%A"),
                date=now.strftime("%d %B %Y"),
            ),
            TOOL_PREAMBLE.format(tools=self.registry.describe() or "_none_"),
            "## Long-term memory\n\nThis is your memory file. It persists across conversations.\n\n"
            + self.memory.for_prompt(),
        ]
        return "\n\n".join(part.strip() for part in parts if part.strip())

    # -- conversation plumbing --------------------------------------------- #

    def conversation(self, chat_id: int | str) -> Conversation:
        conv = self._conversations.get(chat_id)
        if conv is None:
            conv = Conversation(chat_id=chat_id, session=self.history.new_session(chat_id))
            self._conversations[chat_id] = conv
        if not conv.seeded:
            conv.seeded = True
            conv.messages.append({"role": "system", "content": self.system_prompt()})
            for entry in self.history.recent_dialogue(chat_id, self.config.llm.history_turns):
                conv.messages.append(entry)
        return conv

    def reset(self, chat_id: int | str) -> None:
        self._conversations.pop(chat_id, None)
        log.info("conversation reset for chat %s", chat_id)

    def _context(self, source: str, chat_id: int | str) -> ToolContext:
        return ToolContext(cwd=self.config.shell_cwd, source=source, chat_id=chat_id)

    def _trim(self, conv: Conversation) -> None:
        """Drop old messages, always cutting on a user boundary.

        Cutting anywhere else would leave a ``tool`` message whose preceding
        ``assistant`` tool_calls entry is gone, which the API rejects.
        """
        if len(conv.messages) <= MAX_MESSAGES:
            return
        cut = next(
            (i for i, m in enumerate(conv.messages) if i >= 1 and m.get("role") == "user"),
            None,
        )
        if cut is None or cut == 0:
            return
        conv.messages = conv.messages[cut:]
        log.debug("trimmed conversation for %s to %d messages", conv.chat_id, len(conv.messages))

    def _refresh_system_prompt(self, conv: Conversation) -> None:
        """Keep the system prompt current without losing the message history."""
        if conv.messages and conv.messages[0].get("role") == "system":
            conv.messages[0]["content"] = self.system_prompt()

    # -- the loop ---------------------------------------------------------- #

    async def handle(self, chat_id: int | str, text: str, *, source: str = "chat") -> TurnResult:
        """Run one full turn for *text*."""
        conv = self.conversation(chat_id)
        self._refresh_system_prompt(conv)
        conv.messages.append({"role": "user", "content": text})
        self.history.append(chat_id, "user", text, session=conv.session)
        return await self._loop(conv, self._context(source, chat_id))

    async def _loop(
        self,
        conv: Conversation,
        ctx: ToolContext,
        tools_used: list[str] | None = None,
    ) -> TurnResult:
        started = time.monotonic()
        tools = list(tools_used or [])
        result = TurnResult()

        specs = self.registry.specs()
        if not specs:
            log.warning("no tools available; the model can only answer from the prompt")

        for iteration in range(1, self.config.llm.max_tool_iterations + 1):
            result.iterations = iteration
            self._trim(conv)

            try:
                reply = await self.llm.complete(conv.messages, tools=specs or None)
            except LLMError as exc:
                result.error = str(exc)
                result.elapsed = time.monotonic() - started
                self.history.append(
                    conv.chat_id, "assistant", f"[error] {exc}", session=conv.session
                )
                return result

            conv.messages.append(_assistant_message(reply))

            if not reply.wants_tools:
                result.text = reply.text
                result.tools_used = tools
                result.elapsed = time.monotonic() - started
                if reply.text:
                    self.history.append(conv.chat_id, "assistant", reply.text, session=conv.session)
                    log.info(
                        "turn done: %d iteration(s), %d tool call(s), %.1fs",
                        iteration, len(tools), result.elapsed,
                    )
                return result

            for call in reply.tool_calls:
                if await self._handle_tool_call(conv, call, ctx, tools, result):
                    # Something needs the owner; stop rather than continue with a
                    # half-finished set of side effects.
                    result.tools_used = tools
                    result.elapsed = time.monotonic() - started
                    return result

        # Iteration cap: something is looping. Report it instead of hanging.
        result.text = result.text or (
            "I hit my tool-call limit for this message and stopped. "
            "Try asking for one thing at a time."
        )
        result.error = f"tool iteration limit ({self.config.llm.max_tool_iterations}) reached"
        result.tools_used = tools
        result.elapsed = time.monotonic() - started
        log.warning("tool iteration cap hit for chat %s", conv.chat_id)
        return result

    async def _handle_tool_call(
        self,
        conv: Conversation,
        call: ToolCall,
        ctx: ToolContext,
        tools_used: list[str],
        result: TurnResult,
    ) -> bool:
        """Run one tool call. Returns True if it needs owner approval."""
        try:
            tool_result = await self.registry.invoke(call.name, call.arguments, ctx)
        except NeedsApproval as exc:
            pending = PendingAction(
                id=call.id or uuid.uuid4().hex,
                tool=exc.tool,
                arguments=exc.arguments,
                reason=exc.reason,
                preview=exc.preview,
            )
            conv.pending.append(pending)
            result.pending = list(conv.pending)
            # Exactly one tool message per tool_call, or the next request 400s.
            conv.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": (
                        f"Not executed. It {exc.reason}, so it needs the owner's approval. "
                        "Waiting for their decision."
                    ),
                }
            )
            log.info("tool %s needs approval: %s", call.name, exc.reason)
            return True

        conv.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": tool_result.for_model(),
            }
        )
        tools_used.append(call.name)
        self.history.append(
            conv.chat_id, "tool", f"{call.name}: {tool_result.summary}", session=conv.session, tool=call.name
        )
        return False

    # -- approval resolution ----------------------------------------------- #

    async def resolve(
        self, chat_id: int | str, action_id: str, approved: bool, *, source: str = "chat"
    ) -> TurnResult:
        """Run a pending action for real, or record the refusal, then continue."""
        conv = self.conversation(chat_id)
        ctx = self._context(source, chat_id)

        action = next((p for p in conv.pending if p.id == action_id), None)
        if action is None:
            log.info("approval %s is no longer pending for chat %s", action_id, chat_id)
            return TurnResult(error="That request has already been handled or cancelled.")

        if approved:
            log.info("owner approved %s: %s", action.tool, action.preview[:120])
            tool_result = await self.registry.invoke_approved(action, ctx)
            outcome = f"[Approved by the owner. {action.tool} ran and returned:]\n{tool_result.text}"
            used = [action.tool]
        else:
            log.info("owner declined %s: %s", action.tool, action.preview[:120])
            outcome = "[The owner declined this one. Do not run it, and do not retry it in another form.]"
            used = []

        conv.pending = [p for p in conv.pending if p.id != action_id]
        conv.messages.append({"role": "user", "content": outcome})
        self.history.append(conv.chat_id, "user", outcome, session=conv.session, tool=action.tool)
        return await self._loop(conv, ctx, tools_used=used)

    # -- direct invocation (used by /run, /search, /fetch) ----------------- #

    async def run_tool(
        self, chat_id: int | str, tool: str, arguments: dict[str, Any], *, source: str = "chat"
    ) -> TurnResult:
        """Call one tool outside the model loop and continue the conversation.

        This is what ``/run``, ``/search`` and ``/fetch`` use: the owner asked for
        a specific action, so it runs directly, and the model is then asked to
        narrate the result so the chat still reads like a conversation.
        """
        conv = self.conversation(chat_id)
        self._refresh_system_prompt(conv)
        ctx = self._context(source, chat_id)

        try:
            tool_result = await self.registry.invoke(tool, arguments, ctx)
        except NeedsApproval as exc:
            pending = PendingAction(
                id=uuid.uuid4().hex,
                tool=exc.tool,
                arguments=exc.arguments,
                reason=exc.reason,
                preview=exc.preview,
            )
            conv.pending.append(pending)
            self.history.append(conv.chat_id, "tool", f"{tool}: needs approval", session=conv.session, tool=tool)
            return TurnResult(pending=[pending], text=f"{exc.reason.capitalize()} — confirm below.")

        self.history.append(
            chat_id, "tool", f"{tool}: {tool_result.summary}", session=conv.session, tool=tool
        )
        conv.messages.append(
            {
                "role": "user",
                "content": (
                    f"[Owner asked directly: run `{tool}` with {arguments}. Result:]\n{tool_result.text}"
                ),
            }
        )
        # Seed the counter so the caller can see the direct call happened.
        return await self._loop(conv, ctx, tools_used=[tool])

    # -- maintenance ------------------------------------------------------- #

    def reload(self) -> list[str]:
        """Re-read config-driven prompt sources. Returns human-readable notes."""
        notes = [self.personality.reload()]
        self.memory.load()
        notes.append(f"memory reloaded ({self.memory.stats()['bullet_count']} bullets)")
        for chat_id, conv in self._conversations.items():
            self._refresh_system_prompt(conv)
            self.history.append(chat_id, "system", "reloaded personality and memory")
        return notes

    def close(self) -> None:
        self._conversations.clear()


def _assistant_message(reply: LLMReply) -> dict[str, Any]:
    """Rebuild the assistant turn in the exact shape the API expects."""
    if not reply.wants_tools:
        return {"role": "assistant", "content": reply.text}
    return {
        "role": "assistant",
        "content": reply.text or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in reply.tool_calls
        ],
    }


__all__ = ["Agent", "TurnResult", "PendingAction", "Conversation", "MAX_MESSAGES"]
