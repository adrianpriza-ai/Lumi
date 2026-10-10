"""The agent loop: prompt in, tools called, answer out.

One user message becomes a bounded conversation with the model:

1. build the system prompt from a hardcoded conduct block (how the bot works,
   what it refuses) + ``PERSONALITY.md`` (the voice) + ``MEMORY.md`` + live
   facts,
2. replay recent history so the bot remembers what you said ten messages ago,
3. call the model, execute whatever tools it asks for, feed the results back,
4. repeat until it produces prose or hits ``llm.max_tool_iterations``.

What "bounded" means is :mod:`lumi.context`'s business, not this loop's: the
conversation is measured against the model's context window, and when it no
longer fits the old part is condensed into a record rather than deleted. The
loop itself only has to ask for the fit before every call.

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

from .artifacts import Artifact
from .config import Config
from .context import ContextReport, ContextWindow
from .llm import LLMClient, ToolCall, build_llm
from .llm.base import USAGE_REASONING, LLMError, LLMReply
from .memory import History, MemoryFile
from .paths import relative_to_root
from .personality import Personality
from .tools import NeedsApproval, ToolContext, ToolRegistry, build_registry
from .util.log import get_logger

log = get_logger(__name__)

#: Conversations held in memory at once. Also a hard cap, so a bad
#: ``llm.context_window`` cannot turn into unbounded memory; the live
#: conversation is never evicted, and an evicted one replays from its transcript.
MAX_CONVERSATIONS = 32

RUNTIME_PREAMBLE = """\
## Runtime

These are facts about right now, not instructions. Use them, do not restate them.

- Current time: {now} ({timezone})
- Project root: {root}
- Shell working directory: {cwd}
- Today is {day}, {date}.

This date is authoritative. Your training data ends earlier than today, so
anything that could have changed since then — versions, releases, prices,
news, who holds an office — must come from a fresh web search, not from memory.
When you answer such a question, say "as of {date}" rather than an unqualified
claim.
"""

#: Behaviour and boundaries, as opposed to the voice in PERSONALITY.md. These
#: live in the code on purpose: the owner edits the voice file freely, but a
#: rule that keeps the bot from doing something irreversible should not be one
#: edit away from disappearing — and it should hold even if PERSONALITY.md is
#: missing and the fallback voice is in use.
CONDUCT_PREAMBLE = """\
## How you work

- Say what you ran and show the real output, including errors. Never claim a
  command worked without having seen it work.
- Ask once, plainly, when a tool needs approval — then wait for the owner's
  answer. Never rephrase a refused action to sneak it past the check.
- Don't guess at file contents or command output: read them. When you are
  unsure whether something is true, say so and go check.
- Use the web tools whenever a question depends on anything current or
  specific, rather than answering from memory.
- Keep MEMORY.md tidy: short declarative facts, one per line, no transcripts.

## Boundaries

- The shell and file tools only ever act on the owner's requests. Refuse
  anything that looks like system destruction, privilege escalation, or
  writes outside this project directory.
- Don't run destructive commands on a hunch. The confirmation prompt exists
  for exactly that — ask first.
- Don't send anything anywhere, post anything, or commit anything unless the
  owner asks for it in that conversation.
- Never read or exfiltrate secrets. API keys are scrubbed from the shell
  environment on purpose; do not try to work around it.
- When you have to decline, say so plainly and say why. No lectures.
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
- Cite web results by their bracketed number, e.g. "as of today [2]". Check the
  publish date on each hit: a result is not current just because it ranks
  first, and a page older than the question deserves a second search.
- To give the owner a downloadable file, write it with the `files` tool using
  `upload: true` (or call the `upload` action on an existing file). The file
  arrives in the chat as a document automatically — never paste file contents
  into the message as a substitute.
- Save durable facts about the owner with the `memory` tool, sparingly.
- Only the tools listed above are available right now. If a capability is
  missing here it is because an API key is unset or the tool was disabled in
  config — say so plainly rather than trying to call it.
"""

FORMAT_PREAMBLE = """\
## Formatting (important)

Your messages go to a Telegram chat that parses HTML, not Markdown. Write
Telegraph-style HTML tags — they are the only thing that renders:

- Bold: <b>like this</b>. Italic: <i>like this</i>. Underline: <u>underlined</u>.
- Inline code: <code>like this</code>. Code blocks: <pre>like this</pre> (multi-line
  is fine). Never use backticks, ``` fences, asterisks or underscores as
  formatting — in this chat they show up literally, and stray `_` or `*`
  characters look like noise. Write file paths, shell commands, model names,
  versions and URLs inside <code></code>.
- Escape <, > and & outside tags as &lt; &gt; &amp; when you mean them as text,
  e.g. "a &lt; b" or "M&amp;Ms". A bare < followed by a letter starts a tag and
  can swallow the rest of your message.
- Use formatting sparingly, the way you would in a good chat message: bold for
  a short headline when a message is long, code style for anything you could
  type into a terminal. No headings, no bullet lists unless the answer is
  genuinely a list, and at most one emoji per message.

If a tool result contains Markdown (a web page, a README), convert it: do not
paste **double asterisks**, # headings or [links](https://example.com) raw —
say the link's text and put the URL in <code></code>, or just describe it.
"""


#: How long an unanswered approval stays answerable. An inline button nobody
#: presses must not pin its conversation in memory forever — after this the
#: action is cancelled, the model is told, and the chat becomes evictable.
APPROVAL_TTL_SECONDS = 12 * 3600


@dataclass(slots=True)
class PendingAction:
    """A tool call waiting on the owner's yes or no."""

    id: str
    tool: str
    arguments: dict[str, Any]
    reason: str
    preview: str
    #: Monotonic clock of when the question was asked; see APPROVAL_TTL_SECONDS.
    created: float = field(default_factory=time.monotonic)

    def expired(self, now: float | None = None) -> bool:
        """True once the owner has left this question unanswered for too long."""
        return (time.monotonic() if now is None else now) - self.created > APPROVAL_TTL_SECONDS


@dataclass(slots=True)
class TurnResult:
    """What one turn produced."""

    text: str = ""
    pending: list[PendingAction] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    error: str = ""
    elapsed: float = 0.0
    #: The model's thinking, joined across every iteration of the turn. Empty
    #: unless a reasoning model was in use. Deliberately *not* written back into
    #: the conversation: the trace is not part of the message history, and
    #: replaying it would be rejected by most providers.
    reasoning: str = ""
    #: Reasoning tokens the provider reported, when it reports the split.
    reasoning_tokens: int = 0
    #: Seconds spent inside the model, summed over iterations. Separate from
    #: :attr:`elapsed` because that also counts tool execution, and "thought for
    #: 20s" would be a lie on a turn that spent 18 of them waiting on a search.
    thinking_seconds: float = 0.0
    #: Files the tools produced for the chat. Collected from every ToolResult
    #: during the turn; the presentation layer delivers them (Telegram as
    #: documents). The CLI prints their paths instead.
    artifacts: list[Artifact] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def needs_approval(self) -> bool:
        return bool(self.pending)

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning.strip())


@dataclass(slots=True)
class Conversation:
    """Per-chat state: the live message list, the pending queue, and the counters
    that say what the context window has had to do to it."""

    chat_id: int | str
    messages: list[dict[str, Any]] = field(default_factory=list)
    session: str = ""
    pending: list[PendingAction] = field(default_factory=list)
    seeded: bool = False
    #: The condensed record standing in for turns that no longer fit. Persisted
    #: to the transcript, so a restart resumes the thread rather than forgetting
    #: it a second time.
    summary: str = ""
    #: Counters for ``/context``: how much has been summarised, elided, dropped.
    summarised: int = 0
    elided: int = 0
    dropped: int = 0
    #: Set when the system prompt and the newest turn alone exceed the window —
    #: a misconfiguration, reported in ``/context`` rather than papered over.
    truncated: bool = False
    #: Monotonic clock of last use, for evicting the quietest chat.
    touched: float = 0.0


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
        self.context = ContextWindow(config, llm=self.llm, history=history)
        self._conversations: dict[int | str, Conversation] = {}
        self._max_conversations = max(1, int(config.llm.max_conversations or MAX_CONVERSATIONS))

    # -- prompt ------------------------------------------------------------ #

    def memory_limit(self) -> int:
        """Characters of MEMORY.md to put in the system prompt.

        Delegated to the context window, which is where the rule lives: the
        memory file is the one part of the prompt nothing can condense away, so
        it is the one part that has to be bounded up front.
        """
        return self.context.memory_limit()

    def system_prompt(self) -> str:
        now = datetime.now(UTC)
        limit = self.memory_limit()
        memory_block = self.memory.for_prompt(limit)
        parts = [
            self.personality.text,
            CONDUCT_PREAMBLE,
            RUNTIME_PREAMBLE.format(
                now=now.strftime("%Y-%m-%d %H:%M:%S"),
                timezone=now.tzname() or "UTC",
                root=self.config.root,
                cwd=relative_to_root(self.config.shell_cwd),
                day=now.strftime("%A"),
                date=now.strftime("%d %B %Y"),
            ),
            TOOL_PREAMBLE.format(tools=self.registry.describe(available_only=True) or "_none_"),
            FORMAT_PREAMBLE,
            "## Long-term memory\n\n"
            + (
                "This is your memory file. It persists across conversations. Anything "
                "marked as elided below is still on disk, just too old to send — say so "
                "rather than pretending you were never told.\n\n"
                if "[older remembered fact" in memory_block or "[older memory elided]" in memory_block
                else "This is your memory file. It persists across conversations.\n\n"
            )
            + memory_block,
        ]
        return "\n\n".join(part.strip() for part in parts if part.strip())

    # -- conversation plumbing --------------------------------------------- #

    def conversation(self, chat_id: int | str) -> Conversation:
        conv = self._conversations.get(chat_id)
        if conv is None:
            conv = Conversation(chat_id=chat_id, session=self.history.new_session(chat_id))
            self._conversations[chat_id] = conv
            self._evict_quietest(keep=chat_id)
        conv.touched = time.monotonic()
        if not conv.seeded:
            conv.seeded = True
            conv.messages.append({"role": "system", "content": self.system_prompt()})
            self.context.replay(conv)
        return conv

    def _evict_quietest(self, *, keep: int | str) -> None:
        """Forget the quietest conversation once there are too many in memory.

        A conversation cannot grow past the window it was assembled for, so the
        count of chats is the only thing that decides how much this process
        holds. Evicting costs a re-read: the transcript is on disk, and
        :meth:`context.ContextWindow.replay` rebuilds the chat from its tail —
        which is the same code path a restart takes.

        A chat with a pending approval is never evicted, because the owner has
        to be able to answer it and the state lives only in memory. "Never"
        becomes "not while it is still answerable": before picking victims the
        stale approvals are cancelled (:meth:`_expire_approvals`), so one
        dangling button cannot hold the cache open without limit.
        """
        self._expire_approvals()
        while len(self._conversations) > self._max_conversations:
            candidates = [
                conv for chat_id, conv in self._conversations.items()
                if chat_id != keep and not conv.pending
            ]
            if not candidates:
                return
            quietest = min(candidates, key=lambda c: c.touched)
            del self._conversations[quietest.chat_id]
            log.info(
                "evicted the conversation for chat %s from memory (%d held); "
                "it replays from its transcript on the next message",
                quietest.chat_id, len(self._conversations),
            )

    def _expire_approvals(self) -> None:
        """Cancel pending actions the owner has left unanswered for too long.

        Runs before eviction chooses its victims: a pending action is the one
        thing that keeps a conversation out of the candidates, so without a
        deadline a single dangling approval would let ``max_conversations`` be
        exceeded without limit. Each expiry is written into the conversation,
        so the model hears that the wait ended instead of assuming the owner is
        still deciding.
        """
        now = time.monotonic()
        for conv in self._conversations.values():
            stale = [p for p in conv.pending if p.expired(now)]
            if not stale:
                continue
            conv.pending = [p for p in conv.pending if not p.expired(now)]
            for action in stale:
                self._note_approval_expired(conv, action)
            log.info(
                "cancelled %d unanswered approval(s) for chat %s",
                len(stale), conv.chat_id,
            )

    def _note_approval_expired(self, conv: Conversation, action: PendingAction) -> None:
        """Tell the model an unanswered approval was cancelled.

        The conversation already holds the tool message saying the call was
        waiting for the owner; without this note the model would never learn
        that the wait ended.
        """
        note = (
            f"[The owner never answered the request to {action.tool}; it expired "
            "and was cancelled without running. Do not run it or retry it.]"
        )
        conv.messages.append({"role": "user", "content": note})
        self.history.append(conv.chat_id, "user", note, session=conv.session, tool=action.tool)

    def reset(self, chat_id: int | str) -> None:
        self._conversations.pop(chat_id, None)
        log.info("conversation reset for chat %s", chat_id)

    def context_report(self, chat_id: int | str) -> ContextReport:
        """What the context for *chat_id* currently holds. For ``/context``."""
        return self.context.report(self.conversation(chat_id))

    def conversation_state(self, chat_id: int | str) -> Conversation | None:
        """The live conversation for *chat_id*, or ``None`` if it is not in memory.

        Does not create one: this is for looking at a chat, not starting it.
        """
        return self._conversations.get(chat_id)

    def conversations_in_memory(self) -> int:
        return len(self._conversations)

    def _context(self, source: str, chat_id: int | str) -> ToolContext:
        return ToolContext(cwd=self.config.shell_cwd, source=source, chat_id=chat_id)

    def _refresh_system_prompt(self, conv: Conversation) -> None:
        """Keep the system prompt current without losing the message history."""
        if conv.messages and conv.messages[0].get("role") == "system":
            conv.messages[0]["content"] = self.system_prompt()

    # -- the loop ---------------------------------------------------------- #

    async def handle(
        self,
        chat_id: int | str,
        text: str,
        *,
        source: str = "chat",
        image_b64: str | None = None,
        image_mime: str = "image/jpeg",
    ) -> TurnResult:
        """Run one full turn for *text*, optionally with an image.

        When *image_b64* is provided the user message is sent as a multimodal
        content array (text + image_url) so vision-capable models can see it.
        """
        conv = self.conversation(chat_id)
        self._refresh_system_prompt(conv)
        content: str | list[dict[str, Any]]
        if image_b64:
            content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            ]
        else:
            content = text
        conv.messages.append({"role": "user", "content": content})
        self.history.append(chat_id, "user", text, session=conv.session)
        return await self._loop(conv, self._context(source, chat_id))

    async def _loop(
        self,
        conv: Conversation,
        ctx: ToolContext,
        tools_used: list[str] | None = None,
    ) -> TurnResult:
        """Run a turn, and leave the conversation inside its window afterwards.

        The fit happens before every call *and* once the turn is over: a reply
        is itself a message, and a conversation left over budget is a
        conversation whose next turn starts by throwing something away. Doing
        it here rather than at the start of the next turn means the record of
        what was condensed is written while the turns it covers are still in
        front of us.
        """
        await self.context.prepare(conv)
        result = await self._turn(conv, ctx, tools_used)
        await self.context.prepare(conv)
        return result

    async def _turn(
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
            await self.context.prepare(conv)

            try:
                asked = time.monotonic()
                reply = await self.llm.complete(conv.messages, tools=specs or None)
            except LLMError as exc:
                result.error = str(exc)
                result.elapsed = time.monotonic() - started
                self.history.append(
                    conv.chat_id, "assistant", f"[error] {exc}", session=conv.session
                )
                return result
            result.thinking_seconds += time.monotonic() - asked
            # The provider's own prompt count is the only exact figure available,
            # so it is fed straight back into the estimator for the next call.
            self.context.observe(reply.usage, conv.messages)

            conv.messages.append(_assistant_message(reply))

            # Accumulate onto the result rather than a local, so every exit from
            # this loop reports the same thinking: a turn that stops to ask for
            # approval has already thought through the first attempt, and
            # throwing that away would hide the most interesting part.
            if reply.reasoning:
                result.reasoning = f"{result.reasoning}\n\n{reply.reasoning}".strip()
            result.reasoning_tokens += reply.usage.get(USAGE_REASONING, 0)

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

            # Every call in this reply is dispatched before the turn stops: an
            # assistant message carrying N tool_calls must be followed by N
            # ``role: "tool"`` messages, or the provider 400s the next request
            # and the malformed pair stays in the conversation until /reset.
            # Approval pauses the *loop*, not the set — and each call is judged
            # on its own, so a sibling that needs nothing still runs.
            needs_owner = False
            for call in reply.tool_calls:
                if await self._handle_tool_call(conv, call, ctx, tools, result):
                    needs_owner = True
            if needs_owner:
                # Stop rather than start another round of side effects while the
                # owner is still deciding.
                result.tools_used = tools
                result.elapsed = time.monotonic() - started
                return result

        # Iteration cap: something is looping. Report it instead of hanging —
        # and say it *in* the conversation, not only to the caller. The last
        # thing the model saw was a tool result, so without this assistant turn
        # the next turn would open on a half-finished exchange it was never
        # told had ended.
        cut_off = result.text or (
            "I hit my tool-call limit for this message and stopped. "
            "Try asking for one thing at a time."
        )
        conv.messages.append({"role": "assistant", "content": cut_off})
        result.text = cut_off
        result.error = f"tool iteration limit ({self.config.llm.max_tool_iterations}) reached"
        result.tools_used = tools
        result.elapsed = time.monotonic() - started
        self.history.append(conv.chat_id, "assistant", cut_off, session=conv.session)
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
        # Stage any files this call produced for the chat. Collected on the
        # result, delivered by the presentation layer — the loop stays
        # transport-agnostic.
        if tool_result.artifacts:
            result.artifacts.extend(tool_result.artifacts)
            log.info(
                "turn collected %d artifact(s) from %s: %s",
                len(tool_result.artifacts), call.name,
                ", ".join(a.path for a in tool_result.artifacts),
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

        if action.expired():
            # A tap on a day-old button must not run the day-old command.
            conv.pending = [p for p in conv.pending if p.id != action_id]
            self._note_approval_expired(conv, action)
            log.info("approval %s expired before it was answered (chat %s)", action_id, chat_id)
            return TurnResult(error="That request expired before it was answered, so it was cancelled.")

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
        resumed = await self._loop(conv, ctx, tools_used=used)
        if approved and tool_result.artifacts:
            # An approved write can carry artifacts too (files written with
            # upload: true that needed an overwrite confirmation). _loop only
            # collects artifacts from model-driven tool calls, so merge these
            # in before handing the turn back to the presentation layer.
            resumed.artifacts.extend(tool_result.artifacts)
        return resumed

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


def _assistant_message(reply: LLMReply) -> dict[str, Any]:
    """Rebuild the assistant turn in the exact shape the API expects.

    ``reply.reasoning`` is left out on purpose. Replaying a thinking trace is
    not something the chat-completions API asks for, and several providers
    reject the whole request if an assistant turn carries a field they do not
    recognise.
    """
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


__all__ = [
    "Agent",
    "TurnResult",
    "PendingAction",
    "Conversation",
    "ContextReport",
    "MAX_CONVERSATIONS",
]
