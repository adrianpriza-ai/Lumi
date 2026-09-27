"""Command safety classification.

This is a *guardrail*, not a sandbox. Shell is a Turing-complete language and no
amount of pattern matching makes ``sh -c`` safe to hand to a language model.
What this module does buy you:

* a hard ``DENY`` list for the commands that destroy machines, so a prompt
  injection or a hallucinated command cannot reboot, reformat, or escalate;
* a ``CONFIRM`` tier for anything that mutates state, so a mistake becomes a
  two-tap undo in the chat instead of a filesystem incident;
* path zoning, so writes stay inside the project directory;
* timeouts and output caps, applied by the shell tool on top of this.

If you need a hard boundary, run the bot inside a container or under a
dedicated unprivileged user. See the README section "Threat model".

Every rule is a ``(regex, tier, reason)`` triple in one of the tables below, so
the policy is auditable on one screen and testable case by case.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from ..util.log import get_logger

log = get_logger(__name__)


class Tier(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(slots=True)
class Verdict:
    tier: Tier
    reason: str = ""
    rule: str = ""

    @property
    def blocked(self) -> bool:
        return self.tier is Tier.DENY

    @property
    def needs_approval(self) -> bool:
        return self.tier is Tier.CONFIRM

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.tier.value.upper()}: {self.reason}" if self.reason else self.tier.value


# --------------------------------------------------------------------------- #
# Rule tables
# --------------------------------------------------------------------------- #

DenyRule: TypeAlias = tuple[re.Pattern[str], str]
ConfirmRule: TypeAlias = tuple[re.Pattern[str], str]

#: Unrecoverable or privilege-escalating. Never runs, not even with approval.
DENY_RULES: list[DenyRule] = [
    (re.compile(r":\s*\(\s*\)\s*\{.*\};\s*:"), "fork bomb"),
    (
        re.compile(
            r"\b(curl|wget|fetch|iwr|invoke-webrequest)\b[^|;]{0,300}\|\s*(sudo\s+)?(ba|z|k|da|fi|a)?sh\b",
            re.IGNORECASE,
        ),
        "piping a download straight into a shell",
    ),
    (
        re.compile(
            r"\b(curl|wget)\b[^|;]{0,200}\|\s*(sudo\s+)?(python[23]?|perl|ruby|node)\b",
            re.IGNORECASE,
        ),
        "piping a download straight into an interpreter",
    ),
    (re.compile(r"\b(mkfs(\.\w+)?|fdisk|sfdisk|parted|wipefs|shred|blkdiscard)\b"), "filesystem or disk destruction"),
    (re.compile(r"\bdd\b[^;]*\bof\s*=\s*/dev/", re.IGNORECASE), "writing directly to a device with dd"),
    (re.compile(r"(>>?|>)\s*/dev/(sd|hd|vd|xvd|nvme|disk|mmcblk|loop)"), "writing to a raw block device"),
    (
        re.compile(r"\b(shutdown|reboot|poweroff|halt|systemctl\s+(poweroff|reboot|halt))\b", re.IGNORECASE),
        "power state change",
    ),
    (re.compile(r"\binit\s+[06]\b"), "power state change via init"),
    (re.compile(r"(^|[;&|]\s*)(sudo|doas|pkexec)\b", re.IGNORECASE), "privilege escalation"),
    (re.compile(r"(^|[;&|]\s*)su\b(?!\w)", re.IGNORECASE), "privilege escalation via su"),
    (re.compile(r"\bchmod\b[^;]*[ugoa]*\+s\b"), "setting the setuid/setgid bit"),
    (re.compile(r"\b(rm|unlink)\b[^;]*\s(/|/\*|~|\$HOME|\$PWD)\s*($|[;&|])"), "recursive delete of /, ~ or $PWD"),
    (re.compile(r"\brm\b[^;]*(-{1,2}[\w-]*[rf][\w-]*)[^;]*/\s*($|[;&|])"), "recursive force-delete of /"),
    (re.compile(r"\bhistory\s+-c\b|\brm\b[^;]*\.(bash|zsh)_history\b"), "clearing shell history"),
    (re.compile(r"\bcrontab\s+-r\b|\brm\b[^;]*/etc/cron"), "removing scheduled jobs"),
    (re.compile(r"\bsetenforce\s+0\b"), "disabling SELinux"),
    (re.compile(r"\biptables\b[^;]*\s-F\b|\bufw\b\s+(disable|reload)\b"), "flushing firewall rules"),
    (re.compile(r"\b(git|svn|hg)\b[^;]*\b(push|commit|apply|rebase)\b[^;]*--force\b"), "force-pushing over a shared ref"),
]

#: Mutates state. Allowed, but the owner taps Confirm first.
CONFIRM_RULES: list[ConfirmRule] = [
    (re.compile(r"\brm\b"), "deleting files"),
    (re.compile(r"\b(mv|cp|rsync|install)\b"), "moving or overwriting files"),
    (re.compile(r"\btruncate\b"), "truncating a file"),
    (re.compile(r"\b(sed|perl|ruby)\b[^;]*(\s-i\b|--in-place)"), "in-place edit"),
    (re.compile(r"\b(chmod|chown|chgrp|chattr)\b"), "changing permissions or ownership"),
    (re.compile(r"\b(kill|pkill|killall)\b"), "signalling processes"),
    (re.compile(r"\b(systemctl|service)\b[^;]*\b(stop|restart|reload|disable|enable)\b"), "service control"),
    (
        re.compile(
            r"\bgit\b[^;]*\b(clean\s+-[a-z]*[fdxd]|reset\s+--hard|push\b.*--force(?!-with-lease)"
            r"|checkout\s+--\s+\.|restore\s+\.|branch\s+-D|reflog\s+expire)\b"
        ),
        "discarding git work",
    ),
    (re.compile(r"\b(npm|pnpm|yarn|bun)\b[^;]*\b(uninstall|publish|add)\b"), "changing installed packages"),
    (
        re.compile(r"\b(pip[0-9]?|uv|poetry|pipenv)\b[^;]*\b(uninstall|install|sync|add)\b"),
        "changing the Python environment",
    ),
    (
        re.compile(r"\b(apt|apt-get|dnf|yum|pacman|apk)\b[^;]*\b(install|remove|purge|upgrade|update)\b"),
        "system package management",
    ),
    (re.compile(r"\b(tar|unzip|7z)\b[^;]*\s-x[a-zA-Z]*\b|\bunzip\b|\bgunzip\b"), "extracting an archive can overwrite files"),
    (re.compile(r"\bdd\b"), "raw device copy"),
    (re.compile(r"\btee\b"), "writing through tee"),
    (re.compile(r"\b(curl|wget)\b[^;]*\s(-O|--output|--output-document|-o\s)\b"), "downloading to a file"),
    (re.compile(r"\b(gh|glab|hub)\b[^;]*\b(pr|issue|merge|release|repo)\b"), "touching remote repositories"),
    (
        re.compile(r"\bgit\b[^;]*\b(commit|push|merge|rebase|cherry-pick|tag|gc|repaint)\b"),
        "changing git history or the working tree",
    ),
    (re.compile(r"\b(ln)\b"), "creating links"),
    (re.compile(r"\b(touch|mkdir|rmdir)\b"), "creating or removing entries"),
]

#: Interpreters that must never appear on the right of a pipe.
PIPE_TARGETS: frozenset[str] = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "python", "python2", "python3", "perl", "ruby", "node", "deno", "bun"}
)

#: Paths the operating system owns. Mutating anything here is a hard deny.
SYSTEM_PREFIXES: tuple[str, ...] = (
    "/etc", "/bin", "/sbin", "/usr", "/boot", "/lib", "/lib64", "/dev", "/proc", "/sys",
    "/var/lib", "/var/spool", "/var/log", "/root", "/opt", "/srv",
    "/System", "/Library", "/Applications",
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
)

#: For these, only the *last* path operand is a write target; the earlier ones
#: are sources being read. Getting this wrong would make ``cp /etc/x .`` look
#: like a write to /etc, so it is worth spelling out.
DESTINATION_LAST_COMMANDS: frozenset[str] = frozenset(
    {"cp", "mv", "ln", "install", "rsync", "gcp", "ditto", "tar"}
)

#: Files that grant code execution or hold credentials. Mutating one of these is
#: a hard deny even inside the home directory: these are the files an attacker
#: (or a hallucinating model) reaches for, and Lumi's whole point is that its
#: state lives under the project directory instead.
PROTECTED_BASENAMES: frozenset[str] = frozenset(
    {
        ".bashrc", ".bash_profile", ".bash_logout", ".bash_aliases",
        ".profile", ".zshrc", ".zshenv", ".zprofile", ".zlogin", ".zlogout",
        ".kshrc", ".cshrc", ".login", ".netrc", ".npmrc", ".pypirc", ".gitconfig",
        "authorized_keys", "known_hosts", "id_rsa", "id_ed25519",
        ".ssh", ".gnupg", ".aws", ".kube", ".docker",
    }
)

#: Commands that only read. These may touch SYSTEM_PREFIXES without a prompt.
#: Anything not listed here is treated as mutating: guessing wrong in the safe
#: direction costs a confirmation tap, guessing wrong the other way costs data.
READ_ONLY_COMMANDS: frozenset[str] = frozenset(
    {
        "cat", "less", "more", "head", "tail", "grep", "egrep", "fgrep", "rg", "ag",
        "ls", "dir", "vdir", "stat", "file", "wc", "find", "fd", "tree", "du", "df",
        "readlink", "echo", "printf", "date", "whoami", "id", "groups", "uname",
        "hostname", "which", "type", "command", "env", "printenv", "ps", "top", "htop",
        "uptime", "free", "lscpu", "lsmem", "lsblk", "lsof", "pwd", "realpath",
        "basename", "dirname", "md5sum", "sha256sum", "xxd", "od", "jq", "sort", "uniq",
        "cut", "tr", "nl", "seq", "yes", "sleep", "test", "column", "diff",
        "awk", "sed", "man", "help", "history", "true", "false", "ping", "dig", "host",
        "getent", "locale", "tty", "who", "w",
    }
)

_OPERATORS = {";", "&&", "||", "|", "&", "(", ")", "{", "}", ">", ">>", "<<<", "<"}
_REDIRECTS = {">", ">>", "<<<"}
_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "{", "}", "\n"}


# --------------------------------------------------------------------------- #
# Tokenising
# --------------------------------------------------------------------------- #


def tokenize(command: str) -> list[str]:
    """Split a command line into words and shell operators, as one flat list.

    ``shlex`` with ``punctuation_chars`` handles quotes the way a shell does and
    returns ``|``, ``;``, ``>`` as separate tokens, which is what the structural
    checks need. Unparseable input (unbalanced quotes) falls back to whitespace
    splitting, and the caller still gets the regex verdicts.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return command.split()


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        c, r = candidate.resolve(), root.resolve()
    except OSError:  # pragma: no cover - broken symlink loop
        return False
    return c == r or r in c.parents


def _program_name(tokens: list[str]) -> str:
    """Name of the command being run, skipping any leading VAR=value assignments."""
    for token in tokens:
        if token in _SEPARATORS:
            return ""  # a leading separator: not something we can reason about
        if "=" in token and not token.startswith(("-", "/", "./", "../")):
            continue  # environment assignment
        return Path(token).name
    return ""


def _path_like(word: str) -> str | None:
    """Return the path portion of *word*, or None if it does not look like one."""
    if "=" in word:
        _, _, tail = word.partition("=")
        if tail.startswith(("/", "~", "./", "../")):
            return tail
        return None
    if word.startswith(("/", "~", "./", "../")) or word in {".", ".."}:
        return word
    if "/" in word and not word.startswith("-") and not word.endswith(":"):
        return word
    return None


def referenced_paths(tokens: list[str], cwd: Path, home: Path | None) -> list[Path]:
    """Best-effort extraction of path-looking arguments.

    Handles bare paths, ``--flag=/path`` and bare ``a/b`` forms. Not exhaustive
    by design; it feeds the write-zone check, which is one of several layers.
    """
    found: list[Path] = []
    for word in tokens:
        if word in _SEPARATORS:
            continue
        # _path_like pulls the value out of --flag=/path, so ask it first and let
        # it reject plain flags rather than skipping every dash-word here.
        raw = _path_like(word)
        if raw is None:
            continue
        if raw.startswith("~"):
            base = home if home is not None else Path.home()
            found.append(Path(os.path.normpath(base / raw.lstrip("~/")))
                         if raw != "~" else base)
            continue
        expanded = Path(raw).expanduser()
        if not expanded.is_absolute():
            expanded = cwd / expanded
        found.append(Path(os.path.normpath(expanded)))
    return found


def _zone(path: Path, project_root: Path, home: Path | None) -> str:
    """Classify a path as ``system`` / ``home`` / ``outside`` / ``project``."""
    text = os.path.normpath(str(path))
    for prefix in SYSTEM_PREFIXES:
        normalised = os.path.normpath(prefix)
        if text == normalised or text.startswith(normalised + os.sep):
            return "system"
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if _is_within(resolved, project_root):
        return "project"
    if home is not None and _is_within(resolved, home):
        return "home"
    return "outside" if resolved.is_absolute() else "project"


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #


def explain(
    command: str,
    *,
    cwd: Path,
    project_root: Path,
    home: Path | None = None,
    extra_deny: tuple[str, ...] | list[str] = (),
    extra_confirm: tuple[str, ...] | list[str] = (),
) -> list[Verdict]:
    """Every rule that fires for *command*, in table order.

    :func:`classify` folds this into a single verdict. Exposed so the policy can
    be inspected by ``/doctor`` and asserted on in tests.
    """
    verdicts: list[Verdict] = []
    command = command.strip()
    if not command:
        return [Verdict(Tier.DENY, "empty command", "empty")]

    # 0. Your own rules take precedence over the built-ins.
    for pattern in extra_deny:
        try:
            if re.search(pattern, command):
                verdicts.append(Verdict(Tier.DENY, f"matches extra_deny {pattern!r}", "extra_deny"))
        except re.error:
            log.warning("extra_deny is not a valid regex, ignoring: %r", pattern)
    for pattern in extra_confirm:
        try:
            if re.search(pattern, command):
                verdicts.append(Verdict(Tier.CONFIRM, f"matches extra_confirm {pattern!r}", "extra_confirm"))
        except re.error:
            log.warning("extra_confirm is not a valid regex, ignoring: %r", pattern)

    # 1. Hard denies.
    for pattern, reason in DENY_RULES:
        if pattern.search(command):
            verdicts.append(Verdict(Tier.DENY, reason, pattern.pattern[:48]))

    # 2. Confirm-worthy commands.
    for pattern, reason in CONFIRM_RULES:
        if pattern.search(command):
            verdicts.append(Verdict(Tier.CONFIRM, reason, pattern.pattern[:48]))

    tokens = tokenize(command)
    program = _program_name(tokens)

    # 3. Structural checks. These beat the regex tables because they see the
    #    actual token structure, so `ls -> f` and `cmd 2>&1` are not mistaken
    #    for redirections the way a `>` pattern would.
    for index, token in enumerate(tokens):
        if token != "|" or index + 1 >= len(tokens):
            continue
        target = Path(tokens[index + 1]).name
        if target in PIPE_TARGETS:
            verdicts.append(Verdict(Tier.DENY, f"piping into {target}", "pipe-to-interpreter"))

    for index, token in enumerate(tokens):
        if token not in _REDIRECTS:
            continue
        # shlex splits "->" into "-" and ">", so an arrow in a human-written
        # command would otherwise look like a redirection. `2>file` is a real
        # redirect and is preceded by the fd number, not by a bare dash.
        if index > 0 and tokens[index - 1] == "-":
            continue
        destination = tokens[index + 1] if index + 1 < len(tokens) else "(stdout)"
        verdicts.append(
            Verdict(Tier.CONFIRM, f"redirecting output into {destination}", "redirect-write")
        )

    # 4. Path zoning.
    has_redirect = any(token in _REDIRECTS for token in tokens)
    mutating = (program not in READ_ONLY_COMMANDS) or has_redirect or not program
    recursive_delete = program == "rm" and any(
        token.startswith("-") and "r" in token for token in tokens
    )

    if mutating:
        targets = referenced_paths(tokens, cwd, home)
        # For cp/mv/ln the earlier operands are sources, so only the last path
        # is actually written to.
        if program in DESTINATION_LAST_COMMANDS and targets:
            targets = [targets[-1]]

        for path in targets:
            zone = _zone(path, project_root, home)
            if zone == "system":
                verdicts.append(
                    Verdict(Tier.DENY, f"would modify the system path {path}", "zone:system-write")
                )
            elif path.name in PROTECTED_BASENAMES or (
                zone != "project" and any(part in PROTECTED_BASENAMES for part in path.parts)
            ):
                verdicts.append(
                    Verdict(
                        Tier.DENY,
                        f"{path.name} holds credentials or runs code on login; leaving it alone",
                        "zone:protected-file",
                    )
                )
            elif zone == "home":
                if recursive_delete:
                    verdicts.append(
                        Verdict(
                            Tier.DENY,
                            f"recursive delete inside your home directory ({path})",
                            "zone:home-recursive-delete",
                        )
                    )
                else:
                    verdicts.append(
                        Verdict(
                            Tier.CONFIRM,
                            f"would write inside your home directory ({path})",
                            "zone:home-write",
                        )
                    )
            elif zone == "outside":
                verdicts.append(
                    Verdict(Tier.CONFIRM, f"would write outside the project ({path})", "zone:outside-write")
                )

    return verdicts


def classify(
    command: str,
    *,
    cwd: Path,
    project_root: Path,
    home: Path | None = None,
    ask_before_risky: bool = True,
    extra_deny: tuple[str, ...] | list[str] = (),
    extra_confirm: tuple[str, ...] | list[str] = (),
) -> Verdict:
    """Fold every matching rule into one verdict; the strictest tier wins."""
    verdicts = explain(
        command,
        cwd=cwd,
        project_root=project_root,
        home=home,
        extra_deny=extra_deny,
        extra_confirm=extra_confirm,
    )
    denies = [v for v in verdicts if v.tier is Tier.DENY]
    if denies:
        return Verdict(Tier.DENY, denies[0].reason, denies[0].rule)

    confirms = [v for v in verdicts if v.tier is Tier.CONFIRM]
    if confirms:
        if not ask_before_risky:
            return Verdict(
                Tier.DENY,
                f"{confirms[0].reason} — blocked because tools.shell.ask_before_risky is off",
                confirms[0].rule,
            )
        return Verdict(Tier.CONFIRM, confirms[0].reason, confirms[0].rule)

    return Verdict(Tier.ALLOW)


def describe_policy() -> str:
    """One-line summary of the active policy, for /doctor and the README."""
    return (
        f"{len(DENY_RULES)} hard-deny rules, {len(CONFIRM_RULES)} confirm rules, "
        f"{len(SYSTEM_PREFIXES)} protected system prefixes, "
        f"{len(PIPE_TARGETS)} pipe targets blocked"
    )


__all__ = [
    "Tier",
    "Verdict",
    "classify",
    "explain",
    "tokenize",
    "describe_policy",
    "referenced_paths",
    "DENY_RULES",
    "CONFIRM_RULES",
    "SYSTEM_PREFIXES",
    "READ_ONLY_COMMANDS",
    "PIPE_TARGETS",
]
