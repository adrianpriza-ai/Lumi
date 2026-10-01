"""The context window: what goes to the model, and what gets dropped when it stops fitting.

A 200k-token model does not need 60 messages. It needs *the last 200k tokens*,
and it needs them to still be there on the next turn, and the turn after that.
Counting messages is what makes a bot forget: 60 messages is a busy afternoon
of chat, and the moment the count is exceeded the oldest prefix is deleted
outright, so the model is later asked about something it has never seen and
answers from imagination. That is the hallucination.

So history is budgeted in tokens, and when the budget runs out the old part is
*condensed* before it is dropped:

1. **Elide tool output.** A ``cat`` of a log file is the bulkiest thing in any
   context and the least conversational. Old tool results shrink to one line
   saying what ran and that its output is gone.
2. **Summarise.** The oldest remaining turns go to the model once, with
   instructions to write down what matters — decisions, facts about the owner,
   open threads, what was already tried and rejected. The result stays in the
   prompt in place of the turns it replaces, and is written to the transcript
   so a restart inherits it instead of starting blind.
3. **Drop verbatim.** Only if summarising was not enough (compaction off, or
   the model declined) does the old part actually go away.
4. **Shrink.** Last resort, and only for a message that is on its own bigger
   than the window.

The newest ``llm.context_keep_recent`` messages are never touched, and an
assistant's tool calls always move with their tool results, because a
transcript holding half of either is a 400 rather than a shorter conversation.

Memory is bounded by construction: a conversation cannot exceed the window it
was assembled for, the transcript is read tail-first rather than whole, and
only ``llm.max_conversations`` chats are held at once — an evicted chat
replays itself from disk, so eviction costs a re-read and nothing else.

Token counts are estimated, not exact: no tokenizer is worth a dependency here,
and every provider reports the real prompt size afterwards, so
:meth:`ContextWindow.observe` recalibrates the estimate from the number the
provider actually counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .util.log import get_logger
from .util.text import truncate

log = get_logger(__name__)

#: What the API adds per message on top of the text: role, separators, the
#: handful of tokens every envelope costs.
MESSAGE_OVERHEAD_TOKENS = 4

#: A tool result under this is not worth eliding — the marker would cost more
#: than the output it replaces.
MIN_ELIDE_TOKENS = 120

#: Marker on the summary note, so it can be told apart from a turn the model
#: actually wrote and re-rendered in place instead of stacking up.
SUMMARY_FLAG = "lumi_summary"

#: Marker on a tool result whose body has been dropped.
ELIDED_FLAG = "lumi_elided"

#: What a dropped tool result is replaced with. The tool name is kept, because
#: "it ran, and here is nothing" is what stops the model running it again.
ELIDED_NOTE = "[{name} ran earlier; its output was dropped to fit the context window]"

#: How the condensed past is introduced. A ``user`` message rather than a
#: second ``system`` one: several providers only accept the system role in
#: first position, and this way the note reads as something that happened
#: rather than as an instruction to follow.
SUMMARY_NOTE = (
    "[condensed record of earlier conversation — the turns below were summarised to "
    "make room. Treat it as what actually happened; do not apologise for it or ask "
    "whether it is real.]\n\n{body}"
)

#: Instructions for the one call that does the condensing. Asked for prose, not
#: a list, and told explicitly what to leave out: a summary that lists the tool
#: calls is as useless as no summary, because keeping the conclusions and
#: dropping the steps is the whole point.
SUMMARY_INSTRUCTIONS = """\
You are condensing the earlier part of a conversation so it can be replaced by a
short record. The full transcript is being dropped from the model's context, so
anything you leave out is gone for good.

Write the record in the third person, in plain prose, under 400 words. Keep:
- what was asked, and what was concluded or decided
- facts about the person you are talking to (preferences, environment, constraints)
- files, commands, versions and URLs that were actually used
- anything still unfinished, promised, or being argued about
- corrections the owner made, since those are the ones that get forgotten

Leave out: the wording of the turns, the sequence of individual tool calls and
their raw output, and anything you would only repeat because it happened.

Output the record and nothing else."""

#: Ceiling on one condensed record, in characters. Four hundred words is the
#: instruction above; this is what stops a chatty model from writing a record
#: that eats the window it was written to protect.
SUMMARY_MAX_CHARS = 8_000

#: What share of the context window the long-term memory file may take when
#: ``llm.max_memory_chars`` does not say otherwise. An eighth leaves the rest
#: for the conversation, which is the part that is actually the point.
MEMORY_WINDOW_SHARE = 8


@dataclass(slots=True)
class ContextReport:
    """What the context currently holds. For ``/context`` and the log."""

    tokens: int = 0
    budget: int = 0
    window: int = 0
    messages: int = 0
    chars: int = 0
    summarised: int = 0
    elided: int = 0
    dropped: int = 0
    calibrated: bool = False
    #: The window is smaller than the floor this conversation needs.
    over_window: bool = False

    @property
    def fill(self) -> float:
        return (self.tokens / self.budget) if self.budget else 0.0

    def lines(self) -> list[str]:
        """The report as chat lines, with a percentage that means something."""
        out = [
            f"window {self.window:,} tokens · budget {self.budget:,} after headroom",
            f"in use {self.tokens:,} tokens ({self.fill:.0%}) · {self.messages} messages"
            f" · {self.chars:,} chars",
        ]
        if self.summarised or self.elided or self.dropped:
            out.append(
                f"condensed: {self.summarised} message(s) summarised · "
                f"{self.elided} tool result(s) elided · {self.dropped} dropped"
            )
        else:
            out.append("condensed: nothing yet — every turn is still verbatim")
        if self.over_window:
            out.append(
                "⚠ the system prompt and the newest turn alone are larger than the "
                "window. Raise llm.context_window, or lower llm.context_keep_recent."
            )
        out.append(
            "token count: calibrated against the provider's own"
            if self.calibrated
            else "token count: estimated from characters (no provider count yet)"
        )
        return out


def content_chars(message: dict[str, Any]) -> int:
    """Characters in a message's content, whichever shape the content has."""
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        # Multimodal turn: text parts plus base64 image parts. Images are not
        # counted in characters at all — their cost is in vision tokens, which
        # the provider reports and observe() absorbs.
        return sum(
            len(str(part.get("text", "")))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return 0


def blocks(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group *messages* into the units that have to move together.

    An assistant turn that asked for tools and the tool results answering them
    are one unit: the API rejects a transcript holding either half. A plain
    turn is a unit of one. Every operation here works on units, so none of them
    can produce a request the provider refuses.
    """
    out: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            group = [message]
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
            out.append(group)
            continue
        out.append([message])
        index += 1
    return out


def flatten(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [message for group in groups for message in group]


class ContextWindow:
    """Keeps one conversation inside the model's context window.

    Built by :class:`lumi.agent.Agent` out of the pieces it has: the config for
    the budget, the client for the single summarising call, and the transcript
    store for replay and for persisting what was condensed.
    """

    def __init__(self, config: Any, *, llm: Any = None, history: Any = None) -> None:
        self.config = config
        self.llm = llm
        self.history = history
        #: Chars per token, corrected by :meth:`observe` once the provider has
        #: reported a real prompt size.
        self._chars_per_token = max(1.0, float(config.llm.chars_per_token))
        self._calibrated = False

    # -- budget ------------------------------------------------------------ #

    @property
    def window(self) -> int:
        return max(1, int(self.config.llm.context_window))

    @property
    def headroom(self) -> int:
        return max(0, int(self.config.llm.context_headroom))

    @property
    def budget(self) -> int:
        """Tokens available for what the conversation already holds."""
        return max(256, self.window - self.headroom)

    def tokens_for(self, text: str) -> int:
        return int(len(text) / self._chars_per_token)

    @property
    def chars_per_token(self) -> float:
        """The divisor in force, for tests and for ``/context``."""
        return self._chars_per_token

    def chars_for(self, tokens: int) -> int:
        return int(max(0, tokens) * self._chars_per_token)

    def memory_limit(self) -> int:
        """Characters ``MEMORY.md`` may occupy in the system prompt.

        The one cap that is not optional. Everything else in the prompt can be
        condensed away when the window fills, and the system prompt itself is
        never cut — so an oversized memory file would grow into the window with
        nothing able to reclaim it, and the conversation would quietly get
        smaller instead. An explicit ``llm.max_memory_chars`` wins; otherwise the
        file gets an eighth of the window, which is room for hundreds of facts
        and never a threat to the conversation.
        """
        cap = int(self.config.llm.max_memory_chars or 0)
        if cap > 0:
            return cap
        return self.chars_for(self.budget // MEMORY_WINDOW_SHARE)

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        """Tokens for *messages*, to the nearest token.

        Cheap and slightly pessimistic. Every provider reports the true number
        in ``usage.prompt_tokens``, and :meth:`observe` corrects the divisor
        from it, so this only has to be the right order of magnitude.
        """
        chars = 0
        for message in messages:
            chars += content_chars(message)
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message["tool_calls"]:
                    function = call.get("function") or {}
                    chars += len(str(function.get("name", "")))
                    chars += len(str(function.get("arguments", "")))
        return int(chars / self._chars_per_token) + MESSAGE_OVERHEAD_TOKENS * len(messages)

    def observe(self, usage: dict[str, int], messages: list[dict[str, Any]]) -> None:
        """Recalibrate against the prompt size the provider actually counted.

        Nudged rather than replaced, because one call's count includes provider
        framing the estimate has no model of, and a single odd reading must not
        be able to make the window look twice as large as it is.
        """
        reported = int(usage.get("prompt") or 0)
        if reported <= 0:
            return
        estimated = max(1, self.estimate(messages))
        if reported < estimated * 0.25 or reported > estimated * 4:
            # Too far out for a nudge to mean anything: the two numbers are not
            # describing the same thing, so keep the configured divisor.
            return
        chars = sum(content_chars(m) for m in messages) or 1
        measured = chars / reported
        self._chars_per_token = max(1.0, min(12.0, self._chars_per_token * 0.7 + measured * 0.3))
        self._calibrated = True
        log.debug(
            "context estimate recalibrated: %.2f chars/token (provider counted %d tokens)",
            self._chars_per_token, reported,
        )

    # -- transcript replay -------------------------------------------------- #

    def replay(self, conv: Any) -> None:
        """Fill a fresh conversation with as much of the transcript as fits.

        Read tail-first and packed backwards until the budget is reached, so a
        chat that has been running for a week comes back with the last several
        days of it rather than whatever twenty turns happened to be on the end
        of the file.
        """
        if self.history is None:
            return
        budget = self.budget - self.estimate(conv.messages)
        if budget <= 0:
            return

        # history_turns is a floor, not a ceiling: 0 (the default) means "as
        # much as fits", and a nonzero value asks for at least that much.
        wanted = int(self.config.llm.history_turns or 0) * 2
        replayed = self.history.recent_dialogue(
            conv.chat_id,
            max_messages=wanted or None,
            budget_tokens=budget,
            # The same divisor the estimator uses, so a conversation the model
            # reads densely is not replayed into more tokens than it can hold.
            chars_per_token=self._chars_per_token,
        )
        if not replayed.messages and not replayed.summary:
            return

        if replayed.summary:
            self._install_summary(conv, replayed.summary, remember=False)
        first = len(conv.messages)
        conv.messages.extend(replayed.messages)
        # The record was not in the budget it was packed against, so hand back
        # the oldest replayed turns if it pushed the whole over.
        self._give_back(conv, first)
        log.info(
            "replayed %d message(s) for chat %s (~%d of %d tokens)%s",
            len(replayed.messages), conv.chat_id, self.estimate(conv.messages), self.budget,
            " after a condensed record" if replayed.summary else "",
        )

    def _give_back(self, conv: Any, first: int) -> None:
        """Drop replayed turns from *first* onwards until the window holds.

        Only ever touches replayed history, and never leaves an answer without
        the question in front of it, so a seeded conversation is exactly as
        well-formed as a live one.
        """
        while self.estimate(conv.messages) > self.budget and first < len(conv.messages):
            del conv.messages[first]
            if (
                first < len(conv.messages)
                and conv.messages[first].get("role") == "assistant"
                and (first == 0 or conv.messages[first - 1].get("role") == "user")
            ):
                del conv.messages[first]
            log.debug("dropped a replayed turn for chat %s to fit the window", conv.chat_id)

    def _install_summary(self, conv: Any, summary: str, *, remember: bool) -> None:
        """Put the condensed record into the prompt, and optionally on disk."""
        conv.summary = summary.strip()
        note = {"role": "user", "content": SUMMARY_NOTE.format(body=conv.summary), SUMMARY_FLAG: True}
        # Directly after the system prompt, and only ever one of them: a second
        # note would mean the first outlived its own summary.
        messages = conv.messages
        insert_at = 1 if messages and messages[0].get("role") == "system" else 0
        for index in range(insert_at, len(messages)):
            if messages[index].get(SUMMARY_FLAG):
                messages[index] = note
                break
        else:
            messages.insert(insert_at, note)
        if remember:
            self._write_summary(conv, conv.summary)

    def _write_summary(self, conv: Any, summary: str) -> None:
        """Persist a condensed record so a restart inherits it."""
        if self.history is None:
            return
        try:
            self.history.append(conv.chat_id, "summary", summary, session=conv.session)
        except OSError as exc:  # pragma: no cover - a lost note must not break a turn
            log.warning("could not persist the condensed record: %s", exc)

    # -- fitting ----------------------------------------------------------- #

    async def prepare(self, conv: Any) -> None:
        """Bring *conv* inside the window. Called before every model call.

        Cheap when there is nothing to do, which is nearly always: the estimate
        is one pass over the messages, and only the model itself fills a
        context up.
        """
        over = self.estimate(conv.messages) - self.budget
        if over <= 0:
            return

        before = len(conv.messages)
        log.info(
            "context full for chat %s: ~%d tokens over a %d budget; condensing",
            conv.chat_id, over + self.budget, self.budget,
        )
        self._elide_tool_output(conv, over)
        if self.estimate(conv.messages) > self.budget:
            await self._condense(conv)
        if self.estimate(conv.messages) > self.budget:
            self._drop_oldest(conv)
        if self.estimate(conv.messages) > self.budget:
            self._shrink_largest(conv)

        self._refresh_note(conv)
        log.info(
            "context for chat %s: %d -> %d messages, ~%d of %d tokens, record is %d chars",
            conv.chat_id, before, len(conv.messages), self.estimate(conv.messages),
            self.budget, len(conv.summary),
        )

    def _tail_start(self, groups: list[list[dict[str, Any]]]) -> int:
        """Index of the first block in the tail that must never be touched."""
        keep = max(0, int(self.config.llm.context_keep_recent))
        if keep <= 0:
            return len(groups)
        count = 0
        for index in range(len(groups) - 1, -1, -1):
            count += len(groups[index])
            if count >= keep:
                return index
        return 0

    def _split(self, conv: Any) -> tuple[list[list[dict[str, Any]]], ...]:
        """The conversation as ``(head, middle, tail)`` groups.

        *head* is the system prompt and the record note, *tail* the protected
        recent turns, and *middle* everything that may be condensed or dropped.
        The note rides with the head rather than sitting in the middle:
        condensing it would be summarising a summary, and dropping it would
        throw away the only version of the past that exists.
        """
        groups = blocks(conv.messages)
        cut = self._tail_start(groups)
        headed = bool(groups) and groups[0][0].get("role") == "system"
        start = 1 if headed else 0
        region = groups[start:cut]
        head = [group for group in groups[:start] if group]
        head += [group for group in region if any(m.get(SUMMARY_FLAG) for m in group)]
        middle = [group for group in region if not any(m.get(SUMMARY_FLAG) for m in group)]
        return head, middle, groups[cut:]

    def _rejoin(self, conv: Any, parts: tuple[list[list[dict[str, Any]]], ...]) -> None:
        head, middle, tail = parts
        conv.messages = flatten([*head, *middle, *tail])

    def _elide_tool_output(self, conv: Any, over: int) -> None:
        """Shrink old tool results to a one-line marker, oldest first.

        Cheaper than summarising, and lossy in exactly the place where the loss
        costs least: the model needed that output to do one thing, long since
        done, and the fact that the command ran is what it still needs in order
        not to run it again.

        Everything except the final message is fair game, which includes results
        from the current turn: a tool result the model has not answered yet is
        the one thing here that must survive, and a request that has already got
        its answer behind it does not need the body again.
        """
        saved_total = 0
        for message in conv.messages[:-1]:
            if message.get("role") != "tool" or message.get(ELIDED_FLAG):
                continue
            content = message.get("content")
            if not isinstance(content, str) or self.tokens_for(content) < MIN_ELIDE_TOKENS:
                continue
            saved_total += self.tokens_for(content)
            message["content"] = ELIDED_NOTE.format(name=message.get("name") or "a tool")
            message[ELIDED_FLAG] = True
            conv.elided += 1
            if saved_total >= over:
                log.info(
                    "elided %d tool result(s) for chat %s to fit the window",
                    conv.elided, conv.chat_id,
                )
                return

    async def _condense(self, conv: Any) -> None:
        """Summarise the oldest turns in one extra call, and keep the result.

        The record replaces the turns it covers, so the past survives as prose
        rather than vanishing. If the call fails nothing is dropped here: the
        caller decides what to give up next, which is the safer order.
        """
        if not self.config.llm.compaction or self.llm is None:
            return
        parts = self._split(conv)
        head, middle, _ = parts
        if not middle:
            return

        # The oldest turns, up to a third of the budget: enough to be worth a
        # call, small enough that the record is cheap to carry afterwards.
        allowance = self.budget // 3
        taken = 0
        used = 0
        while taken < len(middle):
            weight = self.estimate(middle[taken])
            if taken and used + weight > allowance:
                break
            used += weight
            taken += 1
        # Not worth a call — and not worth losing turns for a record barely
        # smaller than what it replaces. The drop below handles that case.
        if not taken or used < self.budget // 8:
            return

        prior = (conv.summary or "").strip()
        transcript = _as_transcript(flatten(middle[:taken]))
        summary = await self._summarise(transcript, conv, prior=prior)
        if not summary:
            return

        removed = sum(len(group) for group in middle[:taken])
        self._rejoin(conv, (head, middle[taken:], parts[2]))
        conv.summarised += removed
        # The record is the one thing that may never be allowed to grow without
        # bound, so it is capped against the window rather than trusted.
        self._install_summary(
            conv,
            truncate(summary, self.chars_for(self.budget // 4), marker="\n… [record cut]"),
            remember=True,
        )
        log.info(
            "condensed %d message(s) for chat %s into a %d-char record",
            removed, conv.chat_id, len(conv.summary),
        )

    async def _summarise(self, transcript: str, conv: Any, *, prior: str = "") -> str:
        """The one extra call. ``""`` when the model would not play along.

        *prior* is the record written last time, if there was one. Handing it
        back with the new turns lets the model fold the two together, which is
        what keeps the record one document instead of a pile: a record that
        merely accumulates every old one is a record that grows until it is the
        thing that has to be dropped.
        """
        from .llm.base import LLMError

        instructions = SUMMARY_INSTRUCTIONS
        if prior:
            instructions += (
                "\n\nA record of earlier conversation is included below, before the new "
                "turns. Fold them into it: return the updated record as one document, "
                "not the old one with the new one appended."
            )
            body = f"Record so far:\n\n{prior}\n\n---\n\nTurns to fold in:\n\n{transcript}"
        else:
            body = f"Condense this conversation:\n\n{transcript}"

        request = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": body},
        ]
        try:
            reply = await self.llm.complete(request, tools=None)
        except LLMError as exc:
            log.warning("could not condense the conversation for chat %s: %s", conv.chat_id, exc)
            return ""
        except Exception as exc:  # noqa: BLE001 - housekeeping must not break a turn
            log.warning("condensing the conversation for chat %s failed: %s", conv.chat_id, exc)
            return ""
        summary = (getattr(reply, "text", "") or "").strip()
        if not summary:
            log.info("the model returned no record for chat %s; keeping the turns verbatim", conv.chat_id)
            return ""
        return truncate(summary, SUMMARY_MAX_CHARS, marker="\n… [record cut]")

    def _drop_oldest(self, conv: Any) -> None:
        """Give up the oldest turns, condensed record notwithstanding.

        Normally a handful of messages rather than the conversation: a record is
        prose, so it is smaller than the turns it stands in for.
        """
        head, middle, tail = self._split(conv)
        while middle and self.estimate(flatten([*head, *middle, *tail])) > self.budget:
            conv.dropped += len(middle[0])
            middle = middle[1:]
        self._rejoin(conv, (head, middle, tail))

    def _shrink_largest(self, conv: Any) -> None:
        """Last resort: cut the biggest message until the window holds.

        Reached when even the protected tail is too large — one enormous file
        read, one pasted log. Two things are never cut: the system prompt,
        which is the most valuable and most stable text in the window, and the
        newest turn, which is the question being answered.

        If neither leaves enough room, the window is simply smaller than the
        floor this conversation needs. That is a configuration problem, so it
        is named rather than papered over: silently mangling the personality
        file to make a number fit would be a worse bug than an honest warning.
        """
        skipped: set[int] = set()
        for _ in range(24):
            if self.estimate(conv.messages) <= self.budget:
                return
            # Everything except the system prompt, the message being answered,
            # and the record (which is capped against the window already, and
            # would only be restored by the note refresh at the end of prepare).
            body = [
                m
                for m in conv.messages[1:-1]
                if content_chars(m) > MIN_ELIDE_TOKENS * 3
                and not m.get(SUMMARY_FLAG)
                and id(m) not in skipped
            ]
            if not body:
                break
            biggest = max(body, key=content_chars)
            if not isinstance(biggest.get("content"), str):
                # A multimodal turn: its image costs vision tokens that a
                # character count cannot see, so leave it be.
                skipped.add(id(biggest))
                continue
            overage = self.estimate(conv.messages) - self.budget
            keep = max(400, len(biggest["content"]) - self.chars_for(overage) - 200)
            if keep >= len(biggest["content"]):
                break
            biggest["content"] = truncate(
                biggest["content"], keep, marker="\n… [cut to fit the context window]"
            )
            biggest[ELIDED_FLAG] = True
            conv.elided += 1

        if self.estimate(conv.messages) > self.budget:
            log.warning(
                "chat %s needs ~%d tokens of context and the window allows %d: the "
                "system prompt and the newest turn alone exceed it. Raise "
                "llm.context_window (currently %d) or lower llm.context_keep_recent "
                "(currently %d).",
                conv.chat_id, self.estimate(conv.messages), self.budget, self.window,
                int(self.config.llm.context_keep_recent),
            )
            conv.truncated = True

    def _refresh_note(self, conv: Any) -> None:
        """Keep the summary note in step with ``conv.summary``."""
        note = next((m for m in conv.messages if m.get(SUMMARY_FLAG)), None)
        if note is not None:
            note["content"] = SUMMARY_NOTE.format(body=(conv.summary or "").strip())
        elif conv.summary:
            self._install_summary(conv, conv.summary, remember=False)

    # -- reporting --------------------------------------------------------- #

    def report(self, conv: Any) -> ContextReport:
        return ContextReport(
            tokens=self.estimate(conv.messages),
            budget=self.budget,
            window=self.window,
            messages=len(conv.messages),
            chars=sum(content_chars(m) for m in conv.messages),
            summarised=conv.summarised,
            elided=conv.elided,
            dropped=conv.dropped,
            calibrated=self._calibrated,
            over_window=conv.truncated,
        )


def _as_transcript(messages: list[dict[str, Any]]) -> str:
    """Render messages as a plain transcript for the condensing call."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role == "system" or message.get(SUMMARY_FLAG):
            continue
        if role == "tool":
            name = message.get("name") or "tool"
            body = message.get("content")
            lines.append(f"tool ({name}): {truncate(body if isinstance(body, str) else '', 1_200, marker='…')}")
            continue
        if role == "assistant" and message.get("tool_calls"):
            names = ", ".join(
                str((call.get("function") or {}).get("name", "?")) for call in message["tool_calls"]
            )
            lines.append(f"assistant: (called {names})")
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        lines.append(f"{role or 'unknown'}: {content or ''}")
    return "\n\n".join(lines)


__all__ = [
    "ContextWindow",
    "ContextReport",
    "ELIDED_NOTE",
    "ELIDED_FLAG",
    "MESSAGE_OVERHEAD_TOKENS",
    "SUMMARY_FLAG",
    "SUMMARY_INSTRUCTIONS",
    "SUMMARY_NOTE",
    "blocks",
    "content_chars",
    "flatten",
]
