"""The tools themselves: shell execution, file confinement, memory writes."""

from __future__ import annotations

import pytest

from lumi.tools.base import NeedsApproval, ToolContext, ToolError
from lumi.tools.files import FilesTool
from lumi.tools.memory_tool import MemoryTool
from lumi.tools.shell import SECRET_PATTERN, ShellTool


def ctx(**kwargs) -> ToolContext:
    return ToolContext(**kwargs)


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #


@pytest.fixture
def shell(config) -> ShellTool:
    return ShellTool(config)


async def test_runs_a_simple_command(shell: ShellTool) -> None:
    result = await shell.invoke({"command": "echo hello"}, ctx(source="cli"))
    assert result.ok
    assert "hello" in result.text
    assert result.data["exit_code"] == 0


async def test_runs_in_the_workspace_by_default(shell: ShellTool, config) -> None:
    result = await shell.invoke({"command": "pwd"}, ctx(source="cli"))
    assert str(config.shell_cwd) in result.text


async def test_creates_the_workspace_if_absent(config) -> None:
    import shutil

    shutil.rmtree(config.shell_cwd)
    ShellTool(config)
    assert config.shell_cwd.is_dir()


async def test_reports_a_nonzero_exit(shell: ShellTool) -> None:
    result = await shell.invoke({"command": "exit 3"}, ctx(source="cli"))
    assert not result.ok
    assert result.data["exit_code"] == 3


async def test_stderr_is_captured(shell: ShellTool) -> None:
    result = await shell.invoke({"command": "echo oops >&2"}, ctx(source="cli"))
    assert "oops" in result.text


async def test_env_is_scrubbed_of_secrets(shell: ShellTool, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:secret")
    monkeypatch.setenv("MY_DB_PASSWORD", "hunter2")
    result = await shell.invoke(
        {"command": "env | grep -cE 'sk-super-secret|123:secret|hunter2'"}, ctx(source="cli")
    )
    # grep -c prints 0 and exits 1 when there are no matches, which is the point.
    assert "0" in result.text
    # The command string is echoed back, so check the output section specifically.
    stdout = result.text.split("--- stdout ---")[-1]
    assert "sk-super-secret" not in stdout
    assert "hunter2" not in stdout


async def test_home_is_pinned_to_the_project(shell: ShellTool, config) -> None:
    result = await shell.invoke({"command": "echo $HOME"}, ctx(source="cli"))
    assert str(config.root) in result.text


async def test_inherit_home_setting(shell: ShellTool, config) -> None:
    shell.shell_config.home = "inherit"
    result = await shell.invoke({"command": "echo $HOME"}, ctx(source="cli"))
    assert "lumi" not in result.text.split("stdout")[-1].strip()


async def test_timeout_kills_the_process_group(shell: ShellTool, config) -> None:
    config.tools.shell.timeout_seconds = 1
    result = await shell.invoke({"command": "sleep 30"}, ctx(source="cli", confirmed=True))
    assert result.data["timed_out"] is True
    assert "TIMED OUT" in result.text


async def test_output_is_capped(shell: ShellTool, config) -> None:
    config.tools.shell.max_output_chars = 200
    result = await shell.invoke(
        {"command": "for i in $(seq 1 500); do echo line$i; done"}, ctx(source="cli")
    )
    assert "truncated" in result.text
    assert len(result.text) < 1500


async def test_dangerous_command_raises_for_approval(shell: ShellTool) -> None:
    with pytest.raises(NeedsApproval) as caught:
        await shell.invoke({"command": "rm -rf workspace/x"}, ctx(source="cli"))
    assert caught.value.reason
    assert caught.value.preview == "rm -rf workspace/x"


async def test_approved_dangerous_command_runs(shell: ShellTool) -> None:
    result = await shell.invoke(
        {"command": "rm -rf workspace/x"}, ctx(source="cli", confirmed=True)
    )
    assert result.ok


async def test_denied_command_is_refused_not_run(shell: ShellTool) -> None:
    result = await shell.invoke({"command": "rm -rf /"}, ctx(source="cli", confirmed=True))
    assert not result.ok
    assert "Refused" in result.text


async def test_empty_command_is_an_error(shell: ShellTool) -> None:
    with pytest.raises(ToolError):
        await shell.invoke({"command": "   "}, ctx(source="cli"))


async def test_cwd_argument_is_honoured(shell: ShellTool, config) -> None:
    (config.root / "sub").mkdir()
    result = await shell.invoke({"command": "pwd", "cwd": "sub"}, ctx(source="cli"))
    assert str(config.root / "sub") in result.text


async def test_cwd_cannot_escape_the_project(shell: ShellTool) -> None:
    with pytest.raises(ToolError, match="outside the project"):
        await shell.invoke({"command": "pwd", "cwd": "../../etc"}, ctx(source="cli"))


async def test_nonexistent_cwd_is_an_error(shell: ShellTool) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        await shell.invoke({"command": "pwd", "cwd": "nope"}, ctx(source="cli"))


async def test_timeout_override_is_clamped(shell: ShellTool) -> None:
    result = await shell.invoke(
        {"command": "echo ok", "timeout_seconds": 99999}, ctx(source="cli")
    )
    assert result.ok


def test_secret_pattern_catches_the_obvious_names() -> None:
    for name in ("OPENAI_API_KEY", "MY_TOKEN", "DB_PASSWORD", "AWS_SECRET", "SESSION_ID"):
        assert SECRET_PATTERN.search(name), name
    for name in ("PATH", "LANG", "HOME", "TERM"):
        assert not SECRET_PATTERN.search(name), name


def test_shell_reports_its_working_directory(shell: ShellTool) -> None:
    assert "workspace" in shell.summary_line()


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #


@pytest.fixture
def files(config) -> FilesTool:
    return FilesTool(config)


async def test_read_a_project_file(files: FilesTool) -> None:
    result = await files.invoke({"action": "read", "path": "config.toml"}, ctx())
    assert result.ok
    assert "require_owner" in result.text


async def test_read_missing_file_fails_cleanly(files: FilesTool) -> None:
    result = await files.invoke({"action": "read", "path": "nope.md"}, ctx())
    assert not result.ok
    assert "no such file" in result.text


async def test_write_into_the_workspace(files: FilesTool, config) -> None:
    result = await files.invoke(
        {"action": "write", "path": "workspace/notes.md", "content": "hello"}, ctx()
    )
    assert result.ok
    assert (config.shell_cwd / "notes.md").read_text(encoding="utf-8") == "hello"


async def test_write_outside_the_workspace_is_refused(files: FilesTool) -> None:
    # The registry turns this into a failed ToolResult for the model; called
    # directly it surfaces as ToolError.
    with pytest.raises(ToolError, match="not allowed"):
        await files.invoke({"action": "write", "path": "config.toml", "content": "pwned"}, ctx())


async def test_write_escaping_with_dotdot_is_refused(files: FilesTool) -> None:
    with pytest.raises(ToolError, match="not allowed"):
        await files.invoke(
            {"action": "write", "path": "workspace/../../escape.md", "content": "x"}, ctx()
        )


async def test_overwriting_a_non_empty_file_needs_approval(files: FilesTool, config) -> None:
    target = config.shell_cwd / "exists.md"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(NeedsApproval):
        await files.invoke(
            {"action": "write", "path": "workspace/exists.md", "content": "replaced"}, ctx()
        )
    assert target.read_text(encoding="utf-8") == "original"


async def test_overwrite_flag_skips_approval(files: FilesTool, config) -> None:
    target = config.shell_cwd / "exists.md"
    target.write_text("original", encoding="utf-8")
    result = await files.invoke(
        {"action": "write", "path": "workspace/exists.md", "content": "replaced", "overwrite": True},
        ctx(),
    )
    assert result.ok
    assert target.read_text(encoding="utf-8") == "replaced"
    assert "before" in result.text and "after" in result.text  # a diff is shown


async def test_approved_overwrite_runs(files: FilesTool, config) -> None:
    target = config.shell_cwd / "exists.md"
    target.write_text("original", encoding="utf-8")
    await files.invoke(
        {"action": "write", "path": "workspace/exists.md", "content": "replaced"},
        ctx(confirmed=True),
    )
    assert target.read_text(encoding="utf-8") == "replaced"


async def test_write_creates_parent_directories(files: FilesTool, config) -> None:
    result = await files.invoke(
        {"action": "write", "path": "workspace/deep/nested/file.md", "content": "x"}, ctx()
    )
    assert result.ok
    assert (config.shell_cwd / "deep" / "nested" / "file.md").is_file()


async def test_append(files: FilesTool, config) -> None:
    target = config.shell_cwd / "log.md"
    target.write_text("one\n", encoding="utf-8")
    await files.invoke({"action": "append", "path": "workspace/log.md", "content": "two\n"}, ctx())
    assert target.read_text(encoding="utf-8") == "one\ntwo\n"


async def test_append_needs_no_approval_even_when_the_file_exists(files: FilesTool) -> None:
    await files.invoke({"action": "write", "path": "workspace/a.md", "content": "x"}, ctx())
    result = await files.invoke({"action": "append", "path": "workspace/a.md", "content": "y"}, ctx())
    assert result.ok


async def test_list_a_directory(files: FilesTool) -> None:
    result = await files.invoke({"action": "list", "path": "."}, ctx())
    assert result.ok
    assert "config.toml" in result.text


async def test_list_hides_noise(files: FilesTool, config) -> None:
    (config.root / ".git").mkdir(exist_ok=True)
    result = await files.invoke({"action": "list", "path": "."}, ctx())
    assert ".git" not in result.text


async def test_search_with_a_glob(files: FilesTool) -> None:
    result = await files.invoke({"action": "search", "pattern": "**/*.toml"}, ctx())
    assert result.ok
    assert "config.toml" in result.text


async def test_search_with_a_bad_pattern_does_not_crash(files: FilesTool) -> None:
    result = await files.invoke({"action": "search", "pattern": "["}, ctx())
    assert isinstance(result.ok, bool)  # graceful either way


async def test_search_needs_a_pattern(files: FilesTool) -> None:
    with pytest.raises(ToolError, match="pattern is required"):
        await files.invoke({"action": "search"}, ctx())


async def test_stat(files: FilesTool) -> None:
    result = await files.invoke({"action": "stat", "path": "config.toml"}, ctx())
    assert "type: file" in result.text
    assert "size:" in result.text


async def test_read_outside_the_project_is_refused(files: FilesTool) -> None:
    with pytest.raises(ToolError, match="outside the project"):
        await files.invoke({"action": "read", "path": "/etc/passwd"}, ctx())


async def test_read_a_whitelisted_system_path(files: FilesTool) -> None:
    result = await files.invoke({"action": "read", "path": "/etc/os-release"}, ctx())
    assert result.ok
    # /etc/os-release is a symlink to /usr/lib/os-release, so this also proves
    # the whitelist is matched against the requested path, not the resolved one.
    assert result.data["size"] > 0


async def test_unknown_action(files: FilesTool) -> None:
    with pytest.raises(ToolError, match="unknown action"):
        await files.invoke({"action": "teleport"}, ctx())


async def test_write_size_limit(files: FilesTool, config) -> None:
    config.tools.files.max_write_chars = 100
    with pytest.raises(ToolError, match="over the"):
        await files.invoke({"action": "write", "path": "workspace/big.md", "content": "x" * 200}, ctx())


# --------------------------------------------------------------------------- #
# memory tool
# --------------------------------------------------------------------------- #


@pytest.fixture
def memory_tool(config) -> MemoryTool:
    from lumi.memory import MemoryFile

    mem = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    mem.load()
    return MemoryTool(config, mem)


async def test_remember_appends(memory_tool: MemoryTool) -> None:
    result = await memory_tool.invoke({"action": "remember", "fact": "the owner likes tea"}, ctx())
    assert "remembered" in result.text
    assert memory_tool.memory.managed() == ["the owner likes tea"]


async def test_remember_requires_a_fact(memory_tool: MemoryTool) -> None:
    with pytest.raises(ToolError):
        await memory_tool.invoke({"action": "remember"}, ctx())


async def test_recall_searches(memory_tool: MemoryTool) -> None:
    await memory_tool.invoke({"action": "remember", "fact": "the owner likes tea"}, ctx())
    result = await memory_tool.invoke({"action": "recall", "query": "tea"}, ctx())
    assert "likes tea" in result.text


async def test_recall_with_no_match(memory_tool: MemoryTool) -> None:
    result = await memory_tool.invoke({"action": "recall", "query": "zzz"}, ctx())
    assert "no memories match" in result.text


async def test_show_returns_the_file(memory_tool: MemoryTool) -> None:
    result = await memory_tool.invoke({"action": "show"}, ctx())
    assert "## Context" in result.text


async def test_auto_remember_off_refuses(memory_tool: MemoryTool, config) -> None:
    config.tools.memory.auto_remember = False
    result = await memory_tool.invoke({"action": "remember", "fact": "nope"}, ctx())
    assert not result.ok
    assert "auto_remember is off" in result.text


async def test_there_is_no_forget_action(memory_tool: MemoryTool) -> None:
    """The model must not be able to erase what the owner told it."""
    with pytest.raises(ToolError, match="unknown action"):
        await memory_tool.invoke({"action": "forget", "fact": "the security rules"}, ctx())
