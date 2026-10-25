"""Command safety classification.

This is a *guardrail*, not a sandbox. Shell is a Turing-complete language and no
amount of pattern matching makes ``sh -c`` safe to hand to a language model.
What this module does buy you:

* a hard ``DENY`` list for the commands that destroy machines, so a prompt
  injection or a hallucinated command cannot reboot, reformat, or escalate;
* a ``CONFIRM`` tier for anything that destroys or overwrites state — deleting,
  clobbering a file that already exists, writing outside the workspace — so a
  mistake becomes a two-tap undo in the chat instead of a filesystem incident.
  *Creating* something new inside the project just runs; it costs the owner
  nothing to redo and nothing to keep;
* path zoning, so writes stay inside the configured workspace;
* timeouts and output caps, applied by the shell tool on top of this.

If you need a hard boundary, run the bot inside a container or under a
dedicated unprivileged user. See TOOLS.md, section "Threat model".

Most of the policy is a ``(regex, tier, reason)`` triple in one of the tables
below, so it stays auditable on one screen and testable case by case. The
exceptions are the *structural* checks in :func:`explain` — ``cd`` tracking,
quoted-string extraction, redirect targets — which read the token stream rather
than the raw text because those are the places a pattern alone gets fooled.
"""

from __future__ import annotations

import os
import re
import shlex
import time
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
    # The fork bomb under any trigger character and any quoting: `:(){ :|:& };:`
    # and `.(){ .|.&\};.` are the same program. Empty parens plus a pipe plus a
    # background job is the shape; the trailing trigger need not be a colon.
    (re.compile(r"\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}"), "fork bomb"),
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
    (
        re.compile(r"\b(mkfs(\.\w+)?|fdisk|sfdisk|parted|wipefs|shred|blkdiscard)\b"),
        "filesystem or disk destruction",
    ),
    (
        re.compile(r"\bdd\b[^;]*\bof\s*=\s*/dev/", re.IGNORECASE),
        "writing directly to a device with dd",
    ),
    (
        re.compile(r"(>>?|>)\s*/dev/(sd|hd|vd|xvd|nvme|disk|mmcblk|loop)"),
        "writing to a raw block device",
    ),
    (
        re.compile(
            r"\b(shutdown|reboot|poweroff|halt|systemctl\s+(poweroff|reboot|halt))\b", re.IGNORECASE
        ),
        "power state change",
    ),
    (re.compile(r"\binit\s+[06]\b"), "power state change via init"),
    (re.compile(r"(^|[;&|]\s*)(sudo|doas|pkexec)\b", re.IGNORECASE), "privilege escalation"),
    (re.compile(r"(^|[;&|]\s*)su\b(?!\w)", re.IGNORECASE), "privilege escalation via su"),
    (re.compile(r"\bchmod\b[^;]*[ugoa]*\+s\b"), "setting the setuid/setgid bit"),
    (
        re.compile(r"\b(rm|unlink)\b[^;]*\s(/|/\*|~|\$HOME|\$PWD)\s*($|[;&|])"),
        "recursive delete of /, ~ or $PWD",
    ),
    (
        re.compile(r"\brm\b[^;]*(-{1,2}[\w-]*[rf][\w-]*)[^;]*/\s*($|[;&|])"),
        "recursive force-delete of /",
    ),
    (re.compile(r"\brm\b[^;]*--no-preserve-root\b"), "removing the root safeguard from rm"),
    (re.compile(r"\b(rm|shred|truncate)\b[^;]*/\s*$"), "recursive delete of the filesystem root"),
    # `find / -delete` walks the whole tree and unlinks as it goes, which is the
    # same outcome as `rm -rf /` with a plausible-looking command line. Order
    # varies in the wild, so match both arrangements. Note there is no word
    # boundary after the root path: `/` is not a word character, so `\b` there
    # can never match and the rule would silently never fire.
    (
        re.compile(
            r"(\bfind\b[^;]*-(delete|exec|execdir)\b[^;]*\s/(/|\*|\s*($|[;&|]))"
            r"|\bfind\b\s+(/\*|/|~|\$HOME)(\s|$)[^;]*-(delete|exec|execdir)\b)"
        ),
        "find deleting or executing across a root directory",
    ),
    (
        re.compile(
            r"\b(chown|chmod)\b[^;]*\s-{1,2}[a-zA-Z-]*[Rr][a-zA-Z-]*\b[^;]*\s/(/|\*|\s*($|[;&|]))"
        ),
        "recursive ownership or permission change on /",
    ),
    (re.compile(r"\b(history\s+-c\b|\brm\b[^;]*\.(bash|zsh)_history\b)"), "clearing shell history"),
    (re.compile(r"\bcrontab\s+-r\b|\brm\b[^;]*/etc/cron"), "removing scheduled jobs"),
    (re.compile(r"\bsetenforce\s+0\b"), "disabling SELinux"),
    (
        re.compile(r"\biptables\b[^;]*\s-F\b|\bufw\b\s+(disable|reload)\b"),
        "flushing firewall rules",
    ),
    (
        re.compile(r"\b(insmod|rmmod|modprobe|sysctl\s+-w)\b"),
        "loading kernel modules or writing kernel parameters",
    ),
    (re.compile(r"\b(umount)\b|\bmount\b[^;]*(\s-o\s|\s--bind\b)"), "changing the mount table"),
    (re.compile(r"\bkill\b\s+(-\S+\s+)*1\s*($|[;&|])"), "killing pid 1"),
    (
        re.compile(r"\b(docker|podman)\b[^;]*\s-v\s+/:\S"),
        "bind-mounting the host root into a container",
    ),
    # A reverse shell is what an injected command actually wants: this bot runs
    # with the owner's network access and no secrets in the env, which is enough.
    (re.compile(r"/dev/(tcp|udp)/[\d.]+/\d+"), "opening a reverse-shell socket"),
    (
        re.compile(r"\b(nc|ncat|netcat)\b[^;]*\s-[a-z]*e\b", re.IGNORECASE),
        "netcat with an executable payload",
    ),
    (re.compile(r"\bsocat\b[^;]*\b(exec|system):", re.IGNORECASE), "socat spawning a shell"),
    # `core.pager`, `core.sshCommand` and friends run on the *next* git command,
    # so this is code execution that outlives the invocation that set it.
    (
        re.compile(r"\bgit\b[^;]*\bconfig\b[^;]*(--global|--system|--file\b|-f\s)"),
        "writing persistent git configuration",
    ),
    (
        re.compile(r"\b(git|svn|hg)\b[^;]*\b(push|commit|apply|rebase)\b[^;]*--force\b"),
        "force-pushing over a shared ref",
    ),
]

#: Mutates state. Allowed, but the owner taps Confirm first.
CONFIRM_RULES: list[ConfirmRule] = [
    (re.compile(r"\brm\b"), "deleting files"),
    (re.compile(r"\btruncate\b"), "truncating a file"),
    (re.compile(r"\b(sed|perl|ruby)\b[^;]*(\s-i\b|--in-place)"), "in-place edit"),
    (re.compile(r"\b(chmod|chown|chgrp|chattr)\b"), "changing permissions or ownership"),
    (re.compile(r"\b(kill|pkill|killall)\b"), "signalling processes"),
    (
        re.compile(r"\b(systemctl|service)\b[^;]*\b(stop|restart|reload|disable|enable)\b"),
        "service control",
    ),
    (
        re.compile(
            r"\bgit\b[^;]*\b(clean\s+-[a-z]*[fdxd]|reset\s+--hard|push\b.*--force(?!-with-lease)"
            r"|checkout\s+--\s+\.|restore\s+\.|branch\s+-D|reflog\s+expire)\b"
        ),
        "discarding git work",
    ),
    (
        re.compile(r"\b(npm|pnpm|yarn|bun)\b[^;]*\b(uninstall|publish|add)\b"),
        "changing installed packages",
    ),
    (
        re.compile(r"\b(pip[0-9]?|uv|poetry|pipenv)\b[^;]*\b(uninstall|install|sync|add)\b"),
        "changing the Python environment",
    ),
    (
        re.compile(
            r"\b(apt|apt-get|dnf|yum|pacman|apk)\b[^;]*\b(install|remove|purge|upgrade|update)\b"
        ),
        "system package management",
    ),
    (
        re.compile(r"\b(tar|unzip|7z)\b[^;]*\s-x[a-zA-Z]*\b|\bunzip\b|\bgunzip\b"),
        "extracting an archive can overwrite files",
    ),
    (re.compile(r"\bdd\b"), "raw device copy"),
    (
        re.compile(r"\b(curl|wget)\b[^;]*\s(-O|--output|--output-document|-o\s)\b"),
        "downloading to a file",
    ),
    (
        re.compile(r"\b(gh|glab|hub)\b[^;]*\b(pr|issue|merge|release|repo)\b"),
        "touching remote repositories",
    ),
    (
        re.compile(r"\bgit\b[^;]*\b(commit|push|merge|rebase|cherry-pick|tag|gc|repaint)\b"),
        "changing git history or the working tree",
    ),
    (re.compile(r"\brmdir\b"), "removing a directory"),
    (
        re.compile(r"\bfind\b[^;]*\s-(delete|exec|execdir|ok|okdir)\b"),
        "find is about to unlink or rewrite what it finds",
    ),
    (
        re.compile(r"\|\s*xargs\b[^|;]*\b(rm|mv|cp|shred|dd|truncate|chmod|chown)\b"),
        "xargs is about to run a destructive command",
    ),
    # A file being executed later is still a write. These are the ones that
    # quietly become persistence: a profile line, a git hook, a unit file.
    (re.compile(r"\b(crontab|at|systemd-run|launchctl\s+load)\b"), "scheduling a job to run later"),
    (
        re.compile(r"\b(docker|podman)\b[^;]*\b(volume|network|system)\b[^;]*\b(rm|prune)\b"),
        "removing container volumes or networks",
    ),
    (
        re.compile(
            r"\b(terraform|pulumi|ansible-playbook|helm|kubectl|aws|gcloud|az)\b[^;]*"
            r"\b(apply|destroy|delete|upgrade|rollout|terminate|delete-db|revoke)\b"
        ),
        "changing deployed or cloud infrastructure",
    ),
    (
        re.compile(
            r"\b(redis-cli|mongosh|mongo|psql|mysql|sqlite3)\b[^;]*\b(DROP|TRUNCATE|FLUSHALL|FLUSHDB)\b",
            re.IGNORECASE,
        ),
        "destructive database statement",
    ),
    (re.compile(r"\bopenssl\b[^;]*\b(genrsa|genpkey)\b"), "generating a private key"),
    (re.compile(r"\bssh-keygen\b[^;]*(\s-f\s|\s-R\s)"), "overwriting or removing an ssh key"),
    (
        re.compile(
            r"\b(systemctl|service|initctl|rc-service)\b[^;]*\b(set-default|isolate|mask)\b"
        ),
        "changing service defaults",
    ),
    (
        re.compile(r"\b(export|set)\b[^;]*\b(LD_PRELOAD|LD_LIBRARY_PATH|BASH_ENV|ENV=)\b"),
        "changing how programs are loaded",
    ),
    (re.compile(r"\bchattr\b[^;]*\s\+[ai]"), "setting an immutable or append-only attribute"),
]

#: Interpreters that must never appear on the right of a pipe.
PIPE_TARGETS: frozenset[str] = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "dash",
        "ksh",
        "csh",
        "tcsh",
        "fish",
        "python",
        "python2",
        "python3",
        "perl",
        "ruby",
        "node",
        "deno",
        "bun",
    }
)

#: Paths the operating system owns. Mutating anything here is a hard deny.
SYSTEM_PREFIXES: tuple[str, ...] = (
    "/etc",
    "/bin",
    "/sbin",
    "/usr",
    "/boot",
    "/lib",
    "/lib64",
    "/dev",
    "/proc",
    "/sys",
    "/var/lib",
    "/var/spool",
    "/var/log",
    "/root",
    "/opt",
    "/srv",
    "/System",
    "/Library",
    "/Applications",
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
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
        ".bashrc",
        ".bash_profile",
        ".bash_logout",
        ".bash_aliases",
        ".profile",
        ".zshrc",
        ".zshenv",
        ".zprofile",
        ".zlogin",
        ".zlogout",
        ".kshrc",
        ".cshrc",
        ".login",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".gitconfig",
        "authorized_keys",
        "known_hosts",
        "id_rsa",
        "id_ed25519",
        ".ssh",
        ".gnupg",
        ".aws",
        ".kube",
        ".docker",
    }
)

#: Directories *inside* the workspace that still hand code execution to
#: something else. They are checked even though the project zone would wave them
#: through, because a hook file is a program that git runs on the owner's
#: machine the next time they type ``git commit``.
PROTECTED_SUBPATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        (".git", "hooks"),
        (".git", "config"),
        (".config", "gh"),
        (".vscode", "tasks.json"),
        # The user-level MCP table names the command the web provider spawns.
        # It is outside the project, so the write zone already asks; naming it
        # here makes the answer a deny rather than a tap.
        (".agents", "mcp.json"),
    }
)

#: Project-relative files that grant access or change what the bot will run.
#: They sit inside the shell's default write root (the whole project), so the
#: zone check waves them through, and unlike ``.env`` they are not credentials:
#: writing one is privilege escalation. ``data/whitelist.json`` awards the
#: bot's tools to whoever it lists, ``.mcp.json`` names a local command the web
#: provider will spawn, and ``config.toml`` decides the write roots and deny
#: rules themselves.
PROTECTED_PROJECT_PATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("data", "whitelist.json"),
        (".mcp.json",),
        ("config.toml",),
        ("config.local.toml",),
    }
)

#: Files whose *contents* are credentials. The write guards above protect
#: files because changing them changes what the machine does; this list
#: protects files because *reading* one hands the model exactly what the
#: child-env scrub in ``run_shell`` exists to hide — ``cat .env`` needs no
#: write at all to defeat that control. Deliberately a different set from
#: :data:`PROTECTED_BASENAMES`: a login file like ``.bashrc`` is dangerous to
#: write and boring to read, while ``.env`` is the other way round.
SECRET_FILE_BASENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".mcp.json",
        "config.local.toml",
        "whitelist.json",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
    }
)

#: Key material, named by extension because the issuer decided the basename.
SECRET_FILE_SUFFIXES: tuple[str, ...] = (".pem", ".key", ".p12", ".pfx")

#: System files that hold credentials under names that look ordinary. A
#: read-only command may read a system path — ``cat /etc/passwd`` is fine —
#: but these three are the password hashes and the sudo policy, and reading
#: them hands the model something it should never see.
SYSTEM_SECRET_PATHS: frozenset[str] = frozenset(
    {
        "/etc/shadow",
        "/etc/gshadow",
        "/etc/sudoers",
    }
)

#: Directories whose *contents* are credentials whatever the file is called.
#: ~/.aws/credentials and ~/.ssh/config hold live secrets under names that give
#: nothing away, so the basename list above never sees them. Naming the
#: directory catches every file inside it, at any depth.
CREDENTIAL_DIRS: frozenset[str] = frozenset(
    {
        ".ssh",
        ".gnupg",
        ".aws",
        ".kube",
        ".docker",
    }
)

#: Commands that only read. These may touch SYSTEM_PREFIXES without a prompt.
#: Anything not listed here is treated as mutating: guessing wrong in the safe
#: direction costs a confirmation tap, guessing wrong the other way costs data.
#
#: The list is a heuristic and it has to be, because "reads only" is a property
#: of the arguments as much as the program: ``find`` is read-only until someone
#: hands it ``-delete``. :data:`READ_ONLY_UNLESS` covers the flags that change
#: that, and an unrecognised program is treated as mutating.
READ_ONLY_COMMANDS: frozenset[str] = frozenset(
    {
        "cat",
        "less",
        "more",
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ag",
        "ls",
        "dir",
        "vdir",
        "stat",
        "file",
        "wc",
        "find",
        "fd",
        "tree",
        "du",
        "df",
        "readlink",
        "echo",
        "printf",
        "date",
        "whoami",
        "id",
        "groups",
        "uname",
        "hostname",
        "which",
        "type",
        "command",
        "env",
        "printenv",
        "ps",
        "top",
        "htop",
        "uptime",
        "free",
        "lscpu",
        "lsmem",
        "lsblk",
        "lsof",
        "pwd",
        "realpath",
        "basename",
        "dirname",
        "md5sum",
        "sha256sum",
        "xxd",
        "od",
        "jq",
        "sort",
        "uniq",
        "cut",
        "tr",
        "nl",
        "seq",
        "yes",
        "sleep",
        "test",
        "column",
        "diff",
        "awk",
        "sed",
        "man",
        "help",
        "history",
        "true",
        "false",
        "ping",
        "dig",
        "host",
        "getent",
        "locale",
        "tty",
        "who",
        "w",
    }
)

#: A read-only program stops being read-only when it is given one of these.
#: ``find . -name '*.pyc' -delete`` reads the tree and then unlinks every match,
#: and the only way to tell is to look at the arguments. An empty set would mean
#: "always mutating", but nothing else needs it: programs like ``tee`` and ``dd``
#: are simply absent from :data:`READ_ONLY_COMMANDS` in the first place.
READ_ONLY_UNLESS: dict[str, frozenset[str]] = {
    "find": frozenset(
        {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls", "-fprint", "-fprintf"}
    ),
}

#: Programs that hand their arguments to something else, so the arguments are
#: not really arguments. ``find -exec rm`` and ``xargs rm`` are two spellings of
#: the same delete, and a rule written for one misses the other.
DELEGATING_COMMANDS: frozenset[str] = frozenset(
    {"xargs", "parallel", "env", "timeout", "nohup", "stdbuf"}
)

#: Programs that take a script as an argument, where a path inside a quoted
#: string is something the program will actually open. ``python3 -c "open(
#: '/etc/passwd','w')"`` names no path token at all, so without this the write
#: is invisible to the zone check.
INTERPRETERS: frozenset[str] = frozenset(
    {
        "python",
        "python2",
        "python3",
        "node",
        "deno",
        "bun",
        "perl",
        "ruby",
        "php",
        "lua",
        "sh",
        "bash",
        "zsh",
        "ksh",
    }
)

#: Interpreter flags whose following token is inline code or a module name, not
#: a script path. Without this, ``python3 -c "print(1)"`` would look like an
#: execution of a file called ``print(1)`` and ``python3 -m pytest`` like one
#: called ``pytest``.
INTERPRETER_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "sh": frozenset({"-c", "-o"}),
    "bash": frozenset({"-c", "-o"}),
    "zsh": frozenset({"-c", "-o"}),
    "ksh": frozenset({"-c"}),
    "dash": frozenset({"-c"}),
    "python": frozenset({"-c", "-m", "-W", "-X", "-Q"}),
    "python2": frozenset({"-c", "-m", "-W", "-X"}),
    "python3": frozenset({"-c", "-m", "-W", "-X"}),
    "node": frozenset(
        {"-e", "-p", "-r", "--eval", "--print", "--require", "--import", "--loader"}
    ),
    "deno": frozenset({"-e", "--eval"}),
    "bun": frozenset({"-e", "--eval", "--require"}),
    "perl": frozenset({"-e", "-E", "-M", "-m", "-I"}),
    "ruby": frozenset({"-e", "-r", "-I"}),
    "php": frozenset({"-r", "-f"}),
    "lua": frozenset({"-e", "-l"}),
}

#: Programs that execute a code file named as their first operand. ``source``
#: is ``bash builtin`` with the same effect as running the file.
SCRIPT_RUNNERS: frozenset[str] = INTERPRETERS | frozenset({"source"})

#: Wall-clock start of this process. A script created or changed after this
#: moment was authored during this run; the shell asks the owner before running
#: it. This is the write-then-run half of a prompt injection: fetched content
#: talks the model into writing a program, and the program then executes with
#: no approval of its own.
SESSION_START = time.time()

#: Programs that take a free-text message, where a path in the text is prose and
#: not a target. ``git commit -m "drop the /etc fallback"`` writes to .git and
#: nowhere else, and treating the message as a path made such commits
#: uncommittable.
MESSAGE_COMMANDS: frozenset[str] = frozenset({"git", "hg", "svn"})

#: Flags whose argument is a message rather than a path.
MESSAGE_FLAGS: frozenset[str] = frozenset({"-m", "--message", "-F", "--file"})

#: Programs whose bare, non-flag operands are filesystem paths. This matters
#: because the most dangerous command in the policy is also the shortest:
#: ``rm -rf data`` has no ``/`` in it, so a path-shapedness test sees no path at
#: all and the write zone never gets a say. Restricting the list keeps programs
#: whose arguments are not paths — ``kill 1234`` takes a pid — from growing
#: invented write targets.
PATH_OPERAND_COMMANDS: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "mkdir",
        "touch",
        "cp",
        "mv",
        "ln",
        "install",
        "rsync",
        "gcp",
        "ditto",
        "shred",
        "truncate",
        "tee",
        "chmod",
        "chown",
        "chgrp",
        "chattr",
        "tar",
        "unzip",
        "zip",
        "find",
        "fd",
        "grep",
        "rg",
        "ag",
        "sed",
        "awk",
        "sort",
        "uniq",
        "cut",
        "diff",
        "patch",
        "python",
        "python2",
        "python3",
        "node",
        "deno",
        "bun",
        "perl",
        "ruby",
        "php",
        "echo",
        "printf",
        "cat",
        "dd",
        "stat",
        "file",
        "wc",
        "readlink",
        "realpath",
        "tree",
    }
)

_OPERATORS = {";", "&&", "||", "|", "&", "(", ")", "{", "}", ">", ">>", "<<<", "<"}
_REDIRECTS = {">", ">>"}
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


def _absolute(raw: str, cwd: Path, home: Path | None) -> Path:
    """Turn one path-looking word into an absolute, lexically normalised path."""
    if raw.startswith("~"):
        base = home if home is not None else Path.home()
        return base if raw == "~" else Path(os.path.normpath(base / raw.lstrip("~/")))
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        expanded = cwd / expanded
    return Path(os.path.normpath(expanded))


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
        found.append(_absolute(raw, cwd, home))
    return found


#: Quoted spans, single, double or backtick. ``shlex`` throws the quotes away, so
#: this has to go back to the raw command to see them.
_QUOTED_SPAN = re.compile(r"'([^'\n]*)'|\"([^\"\n]*)\"|`([^`\n]*)`")


def quoted_paths(command: str, cwd: Path, home: Path | None) -> list[Path]:
    """Paths that only appear *inside* a quoted string, for an interpreter.

    ``python3 -c "open('/home/momoi/.bashrc', 'a')"`` has no path token at all:
    the whole program is one argument, and a token walk sees a single opaque
    blob. Reading the quoted spans back out means the zoning check judges the
    literal the same way whether it was typed bare or passed to an interpreter.

    The scan is deliberately narrow, because this is the one heuristic that can
    invent a path out of ordinary prose:

    * only for :data:`INTERPRETERS`, where a string in an argument really is
      something the program opens — ``git commit -m "touch /etc/x"`` is a commit
      message, not a write;
    * only spans with no whitespace, because ``open("a b")`` is a filename with
      a space in it, not two paths.

    Nesting is handled by re-scanning rather than by matching to a fixed depth:
    a span that is not itself a path gets scanned again, so the path inside
    ``"open('/etc/x','w')"`` is found whether or not the wrapper around it
    happens to contain whitespace. ``seen_spans`` is what stops a span that
    quotes itself from looping.
    """
    found: list[Path] = []
    pending = [command]
    seen_spans: set[str] = set()
    while pending:
        text = pending.pop()
        for match in _QUOTED_SPAN.finditer(text):
            span = next(group for group in match.groups() if group is not None).strip()
            if not span or span in seen_spans:
                continue
            seen_spans.add(span)
            raw = _path_like(span)
            looks_like_path = raw is not None and (
                raw.startswith(("/", "~", "./", "../")) or span in {".", ".."}
            )
            if looks_like_path and not any(char.isspace() for char in span):
                found.append(_absolute(raw, cwd, home))
                continue
            # Prose, or a wrapper around a quoted path: look one level in.
            pending.append(span)
    return found


def operand_paths(tokens: list[str], program: str, cwd: Path, home: Path | None) -> list[Path]:
    """Bare, non-flag operands of a path-taking program, resolved to absolute paths.

    ``rm -rf data`` and ``mkdir build`` name their targets without a separator in
    them, so :func:`_path_like` returns None and the write zone is never asked.
    The program gate is what keeps this honest: ``kill 1234`` and ``git commit
    -m wip`` take arguments that are not paths, and resolving those would
    manufacture write targets out of pids and words.
    """
    if program not in PATH_OPERAND_COMMANDS:
        return []
    found: list[Path] = []
    for word in _operands(tokens):
        if word.isdigit():  # a pid, a mode, a count: not a path
            continue
        found.append(_absolute(word, cwd, home))
    return found


def _message_paths(tokens: list[str], base: Path, home: Path | None) -> set[Path]:
    """Paths that appear inside a commit message, and so are not write targets.

    A message is the one argument that routinely *mentions* a path without
    touching it. Reading ``git commit -m 'remove the /etc fallback'`` as a write
    to /etc denies a commit that only edits a file in the workspace.
    """
    paths: set[Path] = set()
    for index, token in enumerate(tokens):
        if token not in MESSAGE_FLAGS:
            continue
        for word in tokens[index + 1 :]:
            if word in _SEPARATORS or word.startswith("-"):
                break
            raw = _path_like(word)
            if raw is not None:
                paths.add(_absolute(raw, base, home))
            break  # the message is a single argument
    return paths


def _operands(tokens: list[str]) -> list[str]:
    """The tokens after the program name, minus flags.

    The program name itself has to come off: in ``cd /etc && rm -rf passwd``
    the segment is ``rm -rf passwd`` and the *word* ``rm`` is the command, not a
    file in /etc. Leading ``VAR=value`` assignments come off for the same reason
    ``_program_name`` skips them.
    """
    words: list[str] = []
    seen_program = False
    for token in tokens:
        if token in _SEPARATORS:
            break
        if not seen_program:
            if "=" in token and not token.startswith(("-", "/", "./", "../")):
                continue  # environment assignment
            seen_program = True
            continue
        if token.startswith("-"):
            continue
        words.append(token)
    return words


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split a token list on shell operators so each command is judged alone."""
    out: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS:
            if current:
                out.append(current)
            current = []
        else:
            current.append(token)
    if current:
        out.append(current)
    return out


#: Commands that move the shell's working directory mid-line.
_CD_COMMANDS: frozenset[str] = frozenset({"cd", "pushd", "popd"})


def effective_cwd(tokens: list[str], cwd: Path, home: Path | None) -> Path | None:
    """The directory the command really runs in, or None if that is unknowable.

    A shell starts in *cwd* and ``cd`` moves it for the rest of the line, so
    ``cd .. && rm -rf data`` deletes a sibling of the workspace. Resolving that
    ``data`` against the workspace would wave it through, which is why this is
    worth resolving properly rather than banning the ``..`` token: the token is
    not the problem, the unfollowed ``cd`` is.

    Returns None for ``cd -``, ``cd $DIR`` and anything else that depends on
    state the policy cannot see. Guessing a base there would silently weaken
    every check that runs afterwards, so the caller escalates instead.
    """
    base = cwd
    for segment in _segments(tokens):
        program = _program_name(segment)
        if program in _CD_COMMANDS:
            if program == "popd":
                return None
            target = next((word for word in segment[1:] if not word.startswith("-")), None)
            if target is None or target == "-":
                return None  # bare `cd` and `cd -` both depend on shell state
            if "$" in target or "`" in target:
                return None
            base = _absolute(target, base, home)
        elif program == "env" and ("-C" in segment or "--chdir" in segment):
            index = segment.index("-C") if "-C" in segment else segment.index("--chdir")
            if index + 1 >= len(segment):
                return None
            base = _absolute(segment[index + 1], base, home)
    return base


def _unwrap_program(tokens: list[str]) -> str:
    """The program that will actually run, looking through thin wrappers.

    ``env cp a /etc/passwd`` is a write to a system path wearing a read-only
    program's name. ``env`` is on the read-only list because ``env`` on its own
    really does only print the environment, so the wrapper has to be peeled off
    before the zone check means anything.
    """
    program = _program_name(tokens)
    if program not in DELEGATING_COMMANDS:
        return program
    for word in tokens[1:]:
        if word.startswith("-") or word.isdigit():
            continue
        if "=" in word and not word.startswith(("/", "~", "./", "../")):
            continue  # `env FOO=bar rm x`
        return Path(word).name
    return program


def _program_token(tokens: list[str]) -> str | None:
    """The program word as typed, skipping leading ``VAR=value`` assignments."""
    for token in tokens:
        if token in _SEPARATORS:
            return None
        if "=" in token and not token.startswith(("-", "/", "./", "../")):
            continue
        return token
    return None


def _interpreter_index(tokens: list[str]) -> int | None:
    """Index of the program that will actually run, looking through wrappers."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _SEPARATORS:
            return None
        if "=" in token and not token.startswith(("-", "/", "./", "../")):
            index += 1
            continue
        if Path(token).name in DELEGATING_COMMANDS:
            index += 1
            while index < len(tokens) and (
                tokens[index].startswith("-")
                or tokens[index].isdigit()
                or (
                    "=" in tokens[index]
                    and not tokens[index].startswith(("/", "~", "./", "../"))
                )
            ):
                index += 1
            continue
        return index
    return None


def script_paths(tokens: list[str], cwd: Path, home: Path | None) -> list[Path]:
    """Files an interpreter (or ``source``) is asked to execute.

    Only the first non-flag operand is a script: operands after it are
    arguments the script reads, and flags such as ``-c``, ``-m`` and ``-e``
    carry inline code or a module name instead of a path.
    """
    index = _interpreter_index(tokens)
    if index is None:
        return []
    program = Path(tokens[index]).name
    if program not in SCRIPT_RUNNERS:
        return []
    value_flags = INTERPRETER_VALUE_FLAGS.get(program, frozenset())
    word_index = index + 1
    while word_index < len(tokens):
        word = tokens[word_index]
        if word in _SEPARATORS:
            break
        if word.startswith("-"):
            word_index += 2 if word in value_flags else 1
            continue
        return [_absolute(word, cwd, home)]
    return []


def _direct_script(tokens: list[str], cwd: Path, home: Path | None) -> Path | None:
    """The program itself when the command executes a path (``./build.sh``)."""
    raw = _program_token(tokens)
    if raw is None:
        return None
    path = _path_like(raw)
    return _absolute(path, cwd, home) if path is not None else None


#: Programs whose bare operands name files they create or replace. Only these
#: feed the same-command authoring check: an interpreter's operands are read,
#: not written, and counting them would call every script self-authored.
CREATING_COMMANDS: frozenset[str] = frozenset({"touch", "mkdir", "tee", "truncate", "shred"})


def _command_write_targets(tokens: list[str], base: Path, home: Path | None) -> set[Path]:
    """Paths this command line creates or writes before later segments run.

    ``echo ... > x.py && python3 x.py`` has no pre-existing file for the
    freshness check to see, so the redirect target itself counts as authored
    by the same command.
    """
    written: set[Path] = set()
    for word in _redirect_destinations(tokens):
        if word:
            written.add(_absolute(word, base, home))
    for segment in _segments(tokens):
        program = _unwrap_program(segment)
        targets = operand_paths(segment, program, base, home)
        if program in DESTINATION_LAST_COMMANDS:
            targets = targets[-1:]
        elif program in CREATING_COMMANDS:
            pass
        else:
            continue
        written.update(targets)
    return written


def _written_since(path: Path, session_start: float) -> bool:
    """True when *path* exists and was created or changed after *session_start*."""
    try:
        return path.stat().st_mtime >= session_start
    except OSError:
        return False


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
    # Checked again after resolution so a symlink that points at /etc is caught
    # as the system path it really is, not as the innocent name it was given.
    resolved_text = os.path.normpath(str(resolved))
    if resolved_text != text:
        for prefix in SYSTEM_PREFIXES:
            normalised = os.path.normpath(prefix)
            if resolved_text == normalised or resolved_text.startswith(normalised + os.sep):
                return "system"
    if _is_within(resolved, project_root):
        return "project"
    if home is not None and _is_within(resolved, home):
        return "home"
    return "outside" if resolved.is_absolute() else "project"


def _has_protected_subpath(path: Path) -> str | None:
    """Name the first :data:`PROTECTED_SUBPATHS` entry contained in *path*, if any."""
    parts = path.parts
    for protected in PROTECTED_SUBPATHS:
        span = len(protected)
        for start in range(len(parts) - span + 1):
            if tuple(parts[start : start + span]) == protected:
                return "/".join(protected)
    return None


#: Sinks that destroy nothing, so writing to them is how you throw output
#: away. Without this the zone check would deny `2>/dev/null` as a write to
#: /dev, which is the single most common redirection there is.
DEV_SINKS: frozenset[str] = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})


def _is_dev_sink(path: Path) -> bool:
    """True for /dev/null and friends, by literal name or after resolution."""
    text = os.path.normpath(str(path))
    if text in DEV_SINKS or text.startswith("/dev/fd/"):
        return True
    try:
        text = os.path.normpath(str(path.resolve()))
    except OSError:  # pragma: no cover - broken symlink loop
        return False
    return text in DEV_SINKS or text.startswith("/dev/fd/")


def _exists(path: Path) -> bool:
    """Does *path* name something that is already there?

    Follows symlinks, and a broken symlink counts: writing through one lands
    on its target, which is exactly the content the overwrite check protects.
    An unstatable path counts too — when in doubt, ask.
    """
    try:
        return path.exists() or path.is_symlink()
    except OSError:  # pragma: no cover - unstatable path
        return True


def _redirect_destinations(tokens: list[str]) -> list[str]:
    """Every token sitting on the right of a ``>`` or ``>>``."""
    out: list[str] = []
    for index, token in enumerate(tokens):
        if token not in _REDIRECTS:
            continue
        if index > 0 and tokens[index - 1] == "-":
            continue  # `ls -> f` is an arrow, not a redirection
        out.append(tokens[index + 1] if index + 1 < len(tokens) else "")
    return out


#: Programs where a destination operand clobbers whatever is already at that
#: name. The check runs against the filesystem, not the command line:
#: ``mv a b`` is only an overwrite when ``b`` exists, and asking on every
#: move would train the owner to tap Confirm without reading.
OVERWRITE_COMMANDS: frozenset[str] = frozenset(
    {"cp", "mv", "ln", "install", "rsync", "gcp", "ditto", "tee"}
)

#: For these, an existing *directory* destination does not mean a clobber:
#: ``mv a existing-dir/`` writes ``existing-dir/a``, and only that name
#: decides. Anything else with an existing directory destination (rsync's
#: merge, ``install -d``) is judged conservatively and asked about.
SINGLE_SOURCE_DEST_COMMANDS: frozenset[str] = frozenset({"cp", "mv", "ln", "install"})


def _flag_values(tokens: list[str], flags: tuple[str, ...]) -> list[str]:
    """The values given to *flags*, in ``-f value`` or ``--flag=value`` form."""
    values: list[str] = []
    for index, token in enumerate(tokens):
        for flag in flags:
            if token == flag and index + 1 < len(tokens):
                values.append(tokens[index + 1])
            elif flag.startswith("--") and token.startswith(flag + "="):
                values.append(token.split("=", 1)[1])
    return values


def _readonly_write_targets(tokens: list[str], base: Path, home: Path | None) -> list[Path]:
    """Files a *read-only* command still writes, under its output flags.

    ``sort``, ``sed``, ``tree``, ``uniq`` and ``xxd`` are in
    :data:`READ_ONLY_COMMANDS` because their common form prints to stdout; with
    ``-o``, ``-i``, a second operand or ``-r`` one of their operands becomes the
    output. Without this, ``sort -o ../config.toml ../config.toml`` was an
    ``allow`` and the state-file guard never saw the write.
    """
    program = _unwrap_program(tokens)
    operands = _operands(tokens)
    # These operands are filenames by definition, including bare names that
    # `_path_like` ignores (`sort -o out.txt ...`). Flag values that are numbers
    # (`uniq -f 2`, `xxd -c 16`) are dropped so they are not mistaken for input.
    words = [word for word in operands if not word.isdigit()]

    def paths(items: list[str]) -> list[Path]:
        return [_absolute(word, base, home) for word in items]

    if program == "sed":
        in_place = any(
            token == "--in-place"
            or token.startswith("--in-place=")
            or (token.startswith("-i") and not token.startswith("--"))
            for token in tokens
        )
        return paths(words) if in_place else []
    if program == "sort":
        return paths(_flag_values(tokens, ("-o", "--output")))
    if program == "tree":
        return paths(_flag_values(tokens, ("-o",)))
    if program == "xxd":
        # `xxd -r [input [output]]`: the last file operand is the write target.
        return paths(words[-1:]) if "-r" in tokens and len(words) > 1 else []
    if program == "uniq":
        # `uniq [input [output]]`: with two files, the last one is the output.
        return paths(words[-1:]) if len(words) > 1 else []
    return []


def _protected_project_path(path: Path, project_root: Path | None) -> str | None:
    """The project-state file a write to *path* would land on, if any.

    Matched after resolution and relative to *project_root*, so a symlink into
    the file is caught the same way the literal name is, and a nested
    ``config.toml`` in some other project is not.
    """
    if project_root is None:
        return None
    try:
        resolved = path.resolve()
        root = project_root.resolve()
    except OSError:  # pragma: no cover - broken symlink loop
        return None
    if not (resolved == root or root in resolved.parents):
        return None
    relative = tuple(resolved.relative_to(root).parts)
    for protected in PROTECTED_PROJECT_PATHS:
        if relative == protected:
            return "/".join(protected)
    return None


def protected_write(path: Path, project_root: Path | None = None) -> str | None:
    """What a write to *path* trips, if anything, under the protected-path rules.

    Returns the protected basename (``.bashrc``, ``id_rsa``, …), subpath
    (``.git/hooks``, …) or project state file (``data/whitelist.json``) that
    makes the write a deny, else ``None``.

    :class:`lumi.tools.files.FilesTool` calls this so both write paths agree
    on what is sacred: a hook file is off-limits through ``run_shell``, so it
    has to be off-limits through the files tool too — otherwise widening
    ``tools.files.writable`` would quietly reopen what the deny list closed.
    *project_root* keeps the shell's scoping rule: a ``.ssh`` directory
    *inside* the project is the owner's own business, one outside it is not.
    """
    if path.name in PROTECTED_BASENAMES:
        return path.name
    subpath = _has_protected_subpath(path)
    if subpath is not None:
        return subpath
    project_path = _protected_project_path(path, project_root)
    if project_path is not None:
        return project_path
    if project_root is not None and not _is_within(path, project_root):
        return next((name for name in path.parts if name in PROTECTED_BASENAMES), None)
    return None


def _is_secret_name(name: str) -> bool:
    """Whether *name* marks a file whose contents are credentials."""
    if name in SECRET_FILE_BASENAMES:
        return True
    # .env.local, .env.production, .envrc — dotenv files under any spelling.
    if name.startswith(".env"):
        return True
    return name.endswith(SECRET_FILE_SUFFIXES)


def credential_dir(path: Path) -> tuple[str, Path] | None:
    """The credential directory *path* lives in, if it lives in one.

    ``~/.aws/credentials`` and ``~/.ssh/config`` hold live secrets under
    names that give nothing away, so the basename list never sees them.
    Naming the directory catches every file inside it, at any depth.

    Returns the directory's name and the directory itself, so a caller can
    ask whether the directory is there.
    """
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - broken symlink loop
        resolved = path
    for candidate in (path, resolved):
        parts = candidate.parts
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] in CREDENTIAL_DIRS:
                return parts[index], Path(*parts[: index + 1])
    return None


def read_guard(path: Path) -> str | None:
    """The credential *path* exposes, if a read of it would expose one.

    Same rules as :func:`protected_read`, but existence-aware, because the two
    kinds of guard differ in what "it is there" means. A secret *file* only
    matters if it exists: ``grep -rn .env .`` names the file without reading
    it, and refusing that would break a normal source-code lookup. A secret
    *directory* matters as soon as the directory is there, because every file
    inside it is a credential — and the file that was named may not be the one
    that exists.
    """
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - broken symlink loop
        resolved = path
    for candidate in (path, resolved):
        if os.path.normpath(str(candidate)) in SYSTEM_SECRET_PATHS:
            return str(candidate)
        if _is_secret_name(candidate.name) and _exists(candidate):
            return candidate.name
    # A credential directory: the guard is the directory being there, not the
    # named file, because the file that exists inside it may be another one.
    found = credential_dir(path)
    if found is not None and _exists(found[1]):
        return f"{found[0]}/"
    return None


def protected_read(path: Path) -> str | None:
    """What a *read* of *path* trips, if anything, under the secret-file rules.

    Returns the name that makes the read a deny (``.env``, ``id_rsa``, …) or
    ``None``. The counterpart to :func:`protected_write`: that one stops a
    secret file being changed, this one stops it being *shown* — the env scrub
    strips ``OPENAI_API_KEY`` from the child process, but ``cat .env`` puts the
    same value straight into the tool result, and only a read-side guard closes
    that hole.

    Checks the path as written as well as the resolved one, so a symlink with
    an innocent name pointing at ``.env`` is caught the same way the write
    guard catches one pointing at ``.bashrc``.
    """
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - broken symlink loop
        resolved = path
    for candidate in (path, resolved):
        if os.path.normpath(str(candidate)) in SYSTEM_SECRET_PATHS:
            return candidate.name
        if _is_secret_name(candidate.name):
            return candidate.name
    # A file inside a credential directory is a credential whatever it is
    # called: ~/.aws/credentials and ~/.ssh/config give nothing away in
    # the filename, so the basename list above never sees them.
    found = credential_dir(path)
    if found is not None:
        return f"{found[0]}/"
    return None


def _clobbered(program: str, tokens: list[str], destinations: list[Path]) -> list[Path]:
    """The subset of *destinations* that already exist and would lose content."""
    hits: list[Path] = []
    for dest in destinations:
        try:
            resolved = dest.resolve()
        except OSError:  # pragma: no cover - broken symlink loop
            resolved = dest
        if not _exists(resolved):
            continue
        if resolved.is_dir() and program in SINGLE_SOURCE_DEST_COMMANDS:
            sources = _operands(tokens)[:-1]  # the last operand is the destination
            for source in sources:
                name = Path(source).name
                candidate = resolved / name if name else resolved
                if _exists(candidate):
                    hits.append(candidate)
        else:
            hits.append(resolved)
    return hits


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #


def _expand_tilde(tokens: list[str], home: Path | None) -> list[str]:
    """Rewrite a leading ``~`` in each token to the directory it means."""
    if home is None:
        return tokens
    out: list[str] = []
    for token in tokens:
        if token == "~":
            out.append(str(home))
        elif token.startswith("~/") or token.startswith("~\\"):
            out.append(str(home) + token[1:])
        else:
            out.append(token)
    return out


def explain(
    command: str,
    *,
    cwd: Path,
    project_root: Path,
    home: Path | None = None,
    write_root: Path | None = None,
    extra_deny: tuple[str, ...] | list[str] = (),
    extra_confirm: tuple[str, ...] | list[str] = (),
    session_start: float | None = None,
    tilde_home: Path | None = None,
) -> list[Verdict]:
    """Every rule that fires for *command*, in table order.

    :func:`classify` folds this into a single verdict. Exposed so the policy can
    be inspected by ``/doctor`` and asserted on in tests.

    :param write_root: the directory writes are allowed to land in. Defaults to
        *project_root*, which is the old behaviour and the right one for a
        library test; the shell tool passes its workspace instead.

    :param session_start: wall-clock start of this run. A script created or
        changed at or after this moment is treated as authored during the
        session, so executing it asks for approval. Defaults to
        :data:`SESSION_START`.

    :param tilde_home: what ``~`` means to the command. The shell pins
        ``HOME`` to the project root, so ``cat ~/notes`` reads a project file,
        not one in the owner's real home. Defaults to *home* (the real home
        directory), which is right for a caller that does not pin it.
    """
    verdicts: list[Verdict] = []
    command = command.strip()
    if not command:
        return [Verdict(Tier.DENY, "empty command", "empty")]

    if write_root is None:
        write_root = project_root

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
                verdicts.append(
                    Verdict(Tier.CONFIRM, f"matches extra_confirm {pattern!r}", "extra_confirm")
                )
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
    # Expand ``~`` the way the command will see it, before any path is
    # resolved. The child process gets HOME pinned to the project root, so
    # judging ``~/.aws/credentials`` against the owner's real home would
    # refuse a read of a file the command cannot even open.
    tokens = _expand_tilde(tokens, tilde_home if tilde_home is not None else home)

    # 3. Structural checks. These beat the regex tables because they see the
    #    actual token structure, so `ls -> f` and `cmd 2>&1` are not mistaken
    #    for redirections the way a `>` pattern would.
    for index, token in enumerate(tokens):
        if token != "|" or index + 1 >= len(tokens):
            continue
        target = _unwrap_program(tokens[index + 1 :])
        if target in PIPE_TARGETS:
            verdicts.append(Verdict(Tier.DENY, f"piping into {target}", "pipe-to-interpreter"))

    # 4. Resolve the directory the command really runs in. `cd .. && rm -rf data`
    #    is the whole reason this exists: judging `data` against the workspace
    #    would miss that the shell is one level up when it is deleted. The
    #    redirection and zoning checks below both judge targets against it.
    base = effective_cwd(tokens, cwd, home)
    if base is None:
        verdicts.append(
            Verdict(
                Tier.CONFIRM,
                "changes directory in a way that cannot be followed, so the target is unknown",
                "cwd:unfollowable",
            )
        )
        base = cwd

    # 5. Redirections write files too, and this is where "creates" and
    #    "clobbers" part ways: `> new.txt` inside the workspace just runs,
    #    `> existing.txt` asks, and anything aimed out of the workspace asks
    #    (or denies) exactly like a path operand would.
    for index, token in enumerate(tokens):
        if token not in _REDIRECTS:
            continue
        # shlex splits "->" into "-" and ">", so an arrow in a human-written
        # command would otherwise look like a redirection. `2>file` is a real
        # redirect and is preceded by the fd number, not by a bare dash.
        if index > 0 and tokens[index - 1] == "-":
            continue
        destination = tokens[index + 1] if index + 1 < len(tokens) else ""
        if not destination:
            verdicts.append(
                Verdict(Tier.CONFIRM, "redirect with no destination", "redirect-write")
            )
            continue
        target_path = _absolute(destination, base, home)
        if _is_dev_sink(target_path):
            continue  # /dev/null destroys nothing
        try:
            resolved = target_path.resolve()
        except OSError:  # pragma: no cover - broken symlink loop
            resolved = target_path
        zone = _zone(target_path, project_root, home)
        if zone == "system":
            verdicts.append(
                Verdict(
                    Tier.DENY, f"would modify the system path {target_path}", "zone:system-write"
                )
            )
        elif not _is_within(resolved, write_root):
            if zone == "home":
                reason = f"would write inside your home directory ({target_path})"
                rule = "zone:home-write"
            elif zone == "outside":
                reason = f"would write outside the project ({target_path})"
                rule = "zone:outside-write"
            else:
                reason = (
                    f"would write outside the workspace ({target_path}); the workspace is "
                    f"{write_root.name or write_root}"
                )
                rule = "zone:outside-workspace"
            verdicts.append(Verdict(Tier.CONFIRM, reason, rule))
        elif token == ">" and _exists(resolved):
            verdicts.append(
                Verdict(
                    Tier.CONFIRM,
                    f"overwriting {target_path.name} by redirecting output into it",
                    "redirect-write",
                )
            )
        # `>>` (append) and a `>` onto a name that does not exist yet, both
        # inside the workspace: nothing to lose, so nothing to ask.

    # 6. Path zoning, one segment at a time. Judging the whole line as a single
    #    command is what lets `cd ..` look like a write target and what makes
    #    `cp a b` ambiguous about which operand is the destination.
    for segment in _segments(tokens):
        segment_program = _unwrap_program(segment)
        if segment_program in _CD_COMMANDS:
            continue  # already accounted for in effective_cwd
        segment_mutating = (
            (segment_program not in READ_ONLY_COMMANDS)
            or any(token in _REDIRECTS for token in segment)
            or not segment_program
        )
        # `find` is a read until someone hands it -delete or -exec.
        if segment_program in READ_ONLY_UNLESS and READ_ONLY_UNLESS[segment_program] & set(segment):
            segment_mutating = True
        if not segment_mutating:
            # Reading is where the env scrub leaks: `cat ../.env` names no
            # write target, so the zoning below never runs for it, yet the
            # output is exactly the secret the scrub exists to hide. Judge the
            # operands of read-only segments against the secret rules before
            # skipping. `read_guard` is existence-aware, so a search *for* the
            # string `.env` (`grep -rn .env .`) stays a search, not a read.
            for word in _operands(segment):
                target = _absolute(word, base, home)
                guard = read_guard(target)
                if guard is not None:
                    verdicts.append(
                        Verdict(
                            Tier.DENY,
                            f"{guard} holds credentials, so its contents are "
                            "not handed to commands",
                            "zone:secret-read",
                        )
                    )
                    continue
                # The read side has the same confinement as the write side: a
                # read-only command may read inside the project, and it may
                # read a system path (that is how `cat /etc/passwd` works),
                # but a read of anything else on the machine is the owner's
                # business, not the model's. The files and explore tools
                # already refuse such a path outright; asking here is the same
                # boundary, with the shell's usual tap instead of an error.
                if not _exists(target):
                    continue
                try:
                    resolved_target = target.resolve()
                except OSError:  # pragma: no cover - broken symlink loop
                    resolved_target = target
                if _is_within(resolved_target, project_root):
                    continue
                if _zone(target, project_root, home) == "system":
                    continue
                verdicts.append(
                    Verdict(
                        Tier.CONFIRM,
                        f"reads a path outside the project ({target})",
                        "zone:outside-read",
                    )
                )
            # A command classified read-only can still name its output as an
            # operand (`sed -i`, `sort -o`, `uniq in out`). Judge only those
            # actual outputs, so the input file beside them stays a read.
            for target in dict.fromkeys(_readonly_write_targets(segment, base, home)):
                try:
                    resolved = target.resolve()
                except OSError:  # pragma: no cover - broken symlink loop
                    resolved = target
                guard = protected_write(resolved, project_root)
                if guard is not None:
                    verdicts.append(
                        Verdict(
                            Tier.DENY,
                            f"{guard} is protected; this command would write it",
                            "zone:protected-write",
                        )
                    )
            continue
        # Per segment, not per line: in `cd .. && rm -rf data` the line's first
        # program is `cd`, and reading the delete as non-recursive because of
        # that would drop it from the deny tier.
        recursive_delete = segment_program == "rm" and any(
            token.startswith("-") and "r" in token for token in segment
        )

        # A target built from a variable or command substitution resolves to
        # something this policy never sees: `mv a $DEST` and
        # `python3 -c "open('$OUT','w')"` both name an in-project path on the
        # command line and a different one at run time. Guessing here would
        # wave them through; escalating costs one tap, the same as an
        # unfollowable `cd`. Read-only segments never reach this, so
        # `grep -r $PATTERN .` still runs free.
        segment_redirects = any(token in _REDIRECTS for token in segment)
        if segment_program in PATH_OPERAND_COMMANDS or segment_redirects:
            for word in [*_operands(segment), *_redirect_destinations(segment)]:
                if "$" in word or "`" in word:
                    verdicts.append(
                        Verdict(
                            Tier.CONFIRM,
                            "the target is built from a variable, so where it writes "
                            "cannot be determined",
                            "target:variable",
                        )
                    )
                    break

        targets = referenced_paths(segment, base, home)
        # Bare operands: `rm -rf data` names its target with no separator in it,
        # so the path-shapedness test above cannot see it. Gated on the program,
        # because `kill 1234` takes a pid and not a path.
        targets += operand_paths(segment, segment_program, base, home)
        # A commit message is prose. `git commit -m 'stop touching /etc'` names
        # a path the same way `rm /etc/passwd` does, and reading it as a write
        # target makes those messages impossible to commit.
        if segment_program in MESSAGE_COMMANDS:
            targets = [p for p in targets if p not in _message_paths(segment, base, home)]
        # Quoted spans, for interpreters: `python3 -c "open('/etc/x','w')"`
        # names no path token at all. Scoped to the interpreter's own segment so
        # a commit message quoting a path stays a commit message.
        if segment_program in INTERPRETERS:
            segment_text = " ".join(segment)
            # The same spans are a *read* vector: `python3 -c "print(open('../.env').read())"`
            # names the secret file inside its script string, where neither the
            # write zoning nor the read-only operand skip ever looks.
            for match in _QUOTED_SPAN.finditer(segment_text):
                span = next(group for group in match.groups() if group is not None).strip()
                if not span:
                    continue
                target = _absolute(span, base, home)
                guard = read_guard(target)
                if guard is not None:
                    verdicts.append(
                        Verdict(
                            Tier.DENY,
                            f"{guard} holds credentials, so its contents are "
                            "not handed to commands",
                            "zone:secret-read",
                        )
                    )
            targets += quoted_paths(segment_text, base, home)
        # For cp/mv/ln the earlier operands are sources, so only the last path
        # is actually written to.
        if segment_program in DESTINATION_LAST_COMMANDS and targets:
            targets = [targets[-1]]

        # Overwrite, not creation, is what asks. `mv a b` clobbers only when
        # `b` is already there; a name that does not exist yet costs the owner
        # nothing to lose, so it runs without a tap.
        if segment_program in OVERWRITE_COMMANDS and targets:
            appending = segment_program == "tee" and bool({"-a", "--append"} & set(segment))
            if not appending:
                destinations = [p for p in targets if not _is_dev_sink(p)]
                for hit in _clobbered(segment_program, segment, destinations):
                    verdicts.append(
                        Verdict(
                            Tier.CONFIRM,
                            f"would overwrite an existing file ({hit})",
                            "overwrite:existing",
                        )
                    )

        for path in dict.fromkeys(targets):
            if _is_dev_sink(path):
                continue  # /dev/null and friends destroy nothing
            zone = _zone(path, project_root, home)
            protected_subpath = _has_protected_subpath(path)
            # The protected-name check runs against the *resolved* path as well
            # as the literal one. A symlink called `notes` pointing at
            # `~/.bashrc` is a write to a login file wearing an innocent name,
            # and only resolution can tell the difference.
            try:
                resolved = path.resolve()
            except OSError:  # pragma: no cover - broken symlink loop
                resolved = path
            protected_name = next(
                (name for name in (path.name, resolved.name) if name in PROTECTED_BASENAMES),
                None,
            )
            protected_part = any(
                name in PROTECTED_BASENAMES for name in (*path.parts, *resolved.parts)
            )
            project_state = _protected_project_path(resolved, project_root)
            if zone == "system":
                verdicts.append(
                    Verdict(Tier.DENY, f"would modify the system path {path}", "zone:system-write")
                )
            elif project_state:
                verdicts.append(
                    Verdict(
                        Tier.DENY,
                        f"{project_state} controls the bot's own access or configuration; "
                        "ask the owner to change it by hand",
                        "zone:protected-state",
                    )
                )
            elif protected_name or (zone != "project" and protected_part):
                verdicts.append(
                    Verdict(
                        Tier.DENY,
                        f"{protected_name or path.name} holds credentials or runs code on login; "
                        "leaving it alone",
                        "zone:protected-file",
                    )
                )
            elif protected_subpath:
                verdicts.append(
                    Verdict(
                        Tier.DENY,
                        f"{protected_subpath} inside the project runs code on someone else's command",
                        "zone:protected-subpath",
                    )
                )
            elif not _is_within(path, write_root):
                # Inside the project but outside the workspace. Escalated to a
                # deny for a recursive delete, which is the irreversible case.
                if recursive_delete:
                    verdicts.append(
                        Verdict(
                            Tier.DENY,
                            f"recursive delete outside the workspace ({path})",
                            "zone:outside-workspace-delete",
                        )
                    )
                else:
                    verdicts.append(
                        Verdict(
                            Tier.CONFIRM,
                            f"would write outside the workspace ({path}); the workspace is "
                            f"{write_root.name or write_root}",
                            "zone:outside-workspace",
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
                    Verdict(
                        Tier.CONFIRM,
                        f"would write outside the project ({path})",
                        "zone:outside-write",
                    )
                )

    # 7. Executing a script authored during this run is the write-then-run half
    #    of a prompt injection: fetched content can talk the model into writing
    #    a program, and without this the shell would run it with no tap of its
    #    own. Scripts that predate this process remain free.
    if session_start is None:
        session_start = SESSION_START
    authored_here = _command_write_targets(tokens, base, home)
    for segment in _segments(tokens):
        candidates = [
            *script_paths(segment, base, home),
            _direct_script(segment, base, home),
        ]
        for script in candidates:
            if script is None:
                continue
            if script in authored_here or _written_since(script, session_start):
                verdicts.append(
                    Verdict(
                        Tier.CONFIRM,
                        f"runs {script.name}, which was created or changed during this session",
                        "exec:recent-script",
                    )
                )

    return verdicts


def classify(
    command: str,
    *,
    cwd: Path,
    project_root: Path,
    home: Path | None = None,
    ask_before_risky: bool = True,
    write_root: Path | None = None,
    extra_deny: tuple[str, ...] | list[str] = (),
    extra_confirm: tuple[str, ...] | list[str] = (),
    session_start: float | None = None,
    tilde_home: Path | None = None,
) -> Verdict:
    """Fold every matching rule into one verdict; the strictest tier wins."""
    verdicts = explain(
        command,
        cwd=cwd,
        project_root=project_root,
        home=home,
        write_root=write_root,
        extra_deny=extra_deny,
        extra_confirm=extra_confirm,
        session_start=session_start,
        tilde_home=tilde_home,
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
        f"{len(PROTECTED_SUBPATHS)} protected subpaths, "
        f"{len(SECRET_FILE_BASENAMES)} secret files refused on read, "
        f"{len(PIPE_TARGETS)} pipe targets blocked"
    )


__all__ = [
    "Tier",
    "Verdict",
    "classify",
    "explain",
    "tokenize",
    "describe_policy",
    "protected_write",
    "protected_read",
    "read_guard",
    "SECRET_FILE_BASENAMES",
    "referenced_paths",
    "quoted_paths",
    "effective_cwd",
    "DENY_RULES",
    "CONFIRM_RULES",
    "SYSTEM_PREFIXES",
    "READ_ONLY_COMMANDS",
    "READ_ONLY_UNLESS",
    "PIPE_TARGETS",
    "PROTECTED_SUBPATHS",
]
