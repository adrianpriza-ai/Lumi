from __future__ import annotations

import lumi.tools.mcp.sandbox as sandbox
from lumi.tools.mcp.sandbox import run_code


async def never(*_args):
    raise AssertionError("no tool call should happen")


async def test_refuses_execution_when_isolation_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_bubblewrap_command", lambda _scratch: None)

    run = await run_code(
        code="print('should not run')", tools=[], on_call=never, timeout=1, max_calls=1
    )

    assert not run.ok
    assert "requires Linux, bubblewrap" in run.error
