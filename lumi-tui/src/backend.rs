//! The backend: a `lumi tui` Python subprocess, spoken to over JSON lines.
//!
//! This replaces the old Rust-side agent wholesale (handoff-tui.md §1): Python
//! owns config, the loop, every tool, the safety engine, memory and the LLM
//! client; this file only translates. `main.rs` still talks in `AgentMsg` and
//! `AgentEvent`, so the key routing and the transcript barely change —
//! protocol events are mapped onto the same enum here.
//!
//! Spawn rule (§3): `$LUMI_BACKEND` (a command line, for dev, `tui` appended)
//! → the project's `.venv/bin/lumi` → `lumi` on PATH → `python3 -m lumi`. The
//! child runs with cwd at the root found by walking up for `config.toml`, so
//! it resolves the same project the TUI was launched in. Its stderr goes to
//! `data/logs/lumi-tui-backend.log`; its stdout is protocol-only.
//!
//! The protocol schema is the doc block at the top of `lumi/tui.py`.

use crate::help_text;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use std::process::Stdio;

/// UI → backend.
#[derive(Debug)]
pub enum AgentMsg {
    /// Raw composer input; slash commands are translated here, everything
    /// else goes to the model.
    Submit(String),
    Cancel,
    /// One answered approval, by id — never a `pending[0]` fallback (§4).
    Approval { id: String, approved: bool },
}

/// Backend → UI.
#[derive(Debug)]
pub enum AgentEvent {
    /// Assistant text delta — one piece of a streamed answer.
    Delta(String),
    /// Reasoning delta (dimmed while the model thinks), from the backend's
    /// `thinking_delta` events while `/reasoning` is on.
    ThinkingDelta(String),
    /// A tool call begins.
    ToolStart { name: String, detail: String },
    /// The same call finished; `summary` is one short line.
    ToolEnd { summary: String, ok: bool },
    /// A whole text block is done (streaming cursor off).
    AssistantDone,
    /// The backend-side conversation was reset; the transcript follows suit.
    ConversationCleared,
    /// Markdown-rendered system message block (help, search results, notes).
    System(String),
    /// Plain status line (slash-command replies).
    Notice(String),
    Error(String),
    /// The backend stopped on a dangerous action and is asking. Several can
    /// arrive before one `result`; each is answered by its own id.
    ApprovalRequest {
        id: String,
        tool: String,
        preview: String,
        reason: String,
    },
    Busy(bool),
    /// Header fields, straight from the backend's config — the frontend never
    /// parses `config.toml` or `.env` itself.
    Ready {
        model: String,
        host: String,
        search: String,
    },
    /// An exchange ended and the conversation state is settled. Busy is not
    /// cleared while an approval is still unanswered: the wait is part of the
    /// turn, not the end of it.
    TurnDone,
}

/// Char-boundary truncation with an ellipsis — shared by the transcript and
/// the approval modal.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{}…", cut)
    }
}

/// Walk up from the current directory looking for `config.toml`, exactly like
/// the Python side resolves its paths. The root decides the child's cwd; if
/// the two ever walked differently, the header could describe a backend
/// pointed at another project.
pub fn find_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..6 {
        if dir.join("config.toml").is_file() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    PathBuf::from(".")
}

/// The child's argv. `$LUMI_BACKEND` is treated as a command line and gets
/// `tui` appended — the dev escape hatch (`uv run lumi`, a venv path).
/// Windows `.venv/Scripts/` is deliberately unhandled (§10).
fn resolve_argv(root: &Path) -> Vec<String> {
    let from_env = std::env::var("LUMI_BACKEND")
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if !from_env.is_empty() {
        let mut argv = from_env;
        argv.push("tui".into());
        return argv;
    }
    let venv = root.join(".venv").join("bin").join("lumi");
    if venv.is_file() {
        return vec![venv.display().to_string(), "tui".into()];
    }
    if which("lumi") {
        return vec!["lumi".into(), "tui".into()];
    }
    vec!["python3".into(), "-m".into(), "lumi".into(), "tui".into()]
}

fn which(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| dir.join(name).is_file())
        })
        .unwrap_or(false)
}

/// Status-bar host: scheme and trailing slash off; the key is never in the
/// URL and is never shown.
fn mask_host(base_url: &str) -> String {
    base_url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
}

/// What `map_event` remembers between protocol lines. Plain data, no I/O —
/// which is what makes the whole mapping testable in-process.
#[derive(Default)]
pub struct MapState {
    /// Approvals announced and not yet answered, in arrival order. A second
    /// event for an id already here is the backend re-reporting its pending
    /// queue on the next exchange (`agent.py` hands back the whole list),
    /// not a second question. Busy is not cleared while this is non-empty.
    pub outstanding: Vec<String>,
    /// A tool step has been shown and not yet closed by a later step, a
    /// message or a result.
    pub tool_open: bool,
    /// A `thinking_delta` has opened a trace block that no `AssistantDone`
    /// has closed yet. The turn ends somewhere — answer, error, cap — so
    /// `turn_done` is where the cursor gets switched off, wherever the block
    /// ended up in the transcript.
    pub thinking_open: bool,
}

fn str_field(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

/// One protocol event → the UI events it implies.
///
/// An unknown `ev` maps to nothing: a newer backend must not panic this
/// frontend. The order of the returned events is the order the UI sees them.
pub fn map_event(v: &Value, st: &mut MapState) -> Vec<AgentEvent> {
    let mut out = Vec::new();
    let mut close_tool = |out: &mut Vec<AgentEvent>| {
        if st.tool_open {
            out.push(AgentEvent::ToolEnd { summary: "done".into(), ok: true });
            st.tool_open = false;
        }
    };
    match v.get("ev").and_then(Value::as_str).unwrap_or("") {
        "ready" => out.push(AgentEvent::Ready {
            model: str_field(v, "model"),
            host: mask_host(&str_field(v, "host")),
            search: str_field(v, "search"),
        }),
        "step" => match str_field(v, "kind").as_str() {
            // A tool step means every earlier step — and its tool — is over.
            "tool" => {
                close_tool(&mut out);
                st.tool_open = true;
                out.push(AgentEvent::ToolStart {
                    name: str_field(v, "detail"),
                    detail: String::new(),
                });
            }
            // "thinking" arrives before every model call; the spinner already
            // says that. The trace itself, when shown, streams as its own
            // `thinking_delta` events right after this.
            _ => {}
        },
        "message" => {
            close_tool(&mut out);
            out.push(AgentEvent::Delta(str_field(v, "text")));
            out.push(AgentEvent::AssistantDone);
        }
        // One piece of a streamed answer. The result that follows carries the
        // same text with `streamed: true`, so it must not render it again.
        "delta" => {
            close_tool(&mut out);
            out.push(AgentEvent::Delta(str_field(v, "text")));
        }
        // The trace streams between a tool step and the answer; the tool it
        // follows is already over.
        "thinking_delta" => {
            close_tool(&mut out);
            st.thinking_open = true;
            out.push(AgentEvent::ThinkingDelta(str_field(v, "text")));
        }
        "result" => {
            close_tool(&mut out);
            let text = str_field(v, "text");
            let streamed = v.get("streamed").and_then(Value::as_bool).unwrap_or(false);
            if !text.is_empty() && !streamed {
                out.push(AgentEvent::Delta(text));
                out.push(AgentEvent::AssistantDone);
            }
            // A streamed result closes its own thinking cursor at `turn_done`
            // (below), which follows immediately and knows whether a trace
            // block is still open.
            if let Some(arts) = v.get("artifacts").and_then(Value::as_array) {
                for art in arts {
                    let path = str_field(art, "path");
                    let size = art.get("size").and_then(Value::as_i64).unwrap_or(0);
                    out.push(AgentEvent::Notice(format!("[file: {path} ({size} bytes)]")));
                }
            }
            let error = str_field(v, "error");
            if !error.is_empty() {
                out.push(AgentEvent::Error(error));
            }
        }
        "approval" => {
            let id = str_field(v, "id");
            if !st.outstanding.iter().any(|seen| *seen == id) {
                st.outstanding.push(id.clone());
                out.push(AgentEvent::ApprovalRequest {
                    id,
                    tool: str_field(v, "tool"),
                    preview: str_field(v, "preview"),
                    reason: str_field(v, "reason"),
                });
            }
        }
        "notice" => {
            let text = str_field(v, "text");
            // Multi-line replies (the tool list, the context report, search
            // hits) are markdown documents, not status lines.
            out.push(if text.contains('\n') {
                AgentEvent::System(text)
            } else {
                AgentEvent::Notice(text)
            });
        }
        "error" => out.push(AgentEvent::Error(str_field(v, "text"))),
        "cleared" => {
            st.outstanding.clear();
            st.thinking_open = false;
            out.push(AgentEvent::ConversationCleared);
            out.push(AgentEvent::Notice("conversation cleared".into()));
        }
        "turn_done" => {
            close_tool(&mut out);
            if st.thinking_open {
                // The model call is over: switch the live trace's cursor off.
                out.push(AgentEvent::AssistantDone);
                st.thinking_open = false;
            }
            // Unanswered approvals hold the exchange open: the modal is up
            // and the turn continues the moment it is answered.
            if st.outstanding.is_empty() {
                out.push(AgentEvent::Busy(false));
            }
            out.push(AgentEvent::TurnDone);
        }
        // Forward-compat: an ev this frontend does not know is skipped, not
        // a crash.
        _ => {}
    }
    out
}

/// What one composer line becomes. `Command` goes to the backend as JSON;
/// `Local` is answered here because it belongs to the frontend or is
/// deliberately gone.
pub enum Translation {
    Command(Value),
    Local(Vec<AgentEvent>),
}

/// Whether a protocol command starts a turn and will therefore end with
/// `turn_done`. The backend answers every other command with a bare
/// `notice`/`cleared`/`error`, so claiming busy for those would leave the
/// spinner up until the next real turn.
pub fn starts_turn(value: &Value) -> bool {
    matches!(
        value.get("cmd").and_then(Value::as_str),
        Some("submit") | Some("approve")
    )
}

/// Translate composer input. The Python backend has no command parser: this
/// is the REPL's command table (`__main__.py::_repl`) wearing JSON.
pub fn translate(text: &str) -> Translation {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix('/') else {
        return Translation::Command(json!({"cmd": "submit", "text": text}));
    };
    let mut parts = rest.splitn(2, char::is_whitespace);
    let verb = parts.next().unwrap_or("").to_ascii_lowercase();
    let arg = parts.next().unwrap_or("").trim().to_string();
    let notice = |s: String| {
        Translation::Local(vec![
            AgentEvent::Notice(s),
            AgentEvent::TurnDone,
        ])
    };
    match verb.as_str() {
        "help" | "?" => Translation::Local(vec![
            AgentEvent::System(help_text()),
            AgentEvent::TurnDone,
        ]),
        "reset" | "clear" => Translation::Command(json!({"cmd": "reset"})),
        "tools" => Translation::Command(json!({"cmd": "tools"})),
        "context" => Translation::Command(json!({"cmd": "context"})),
        "memory" => Translation::Command(json!({"cmd": "memory"})),
        "search" if !arg.is_empty() => Translation::Command(json!({"cmd": "search", "query": arg})),
        // The verb is already case-folded; the mode is too, but the *query*
        // above is not — search terms must reach the provider verbatim.
        "reasoning" => match arg.to_ascii_lowercase().as_str() {
            "on" | "off" => Translation::Command(json!({"cmd": "reasoning", "on": arg.eq_ignore_ascii_case("on")})),
            _ => notice("usage: /reasoning on|off".into()),
        },
        // §10: no frontend-side model switch — a `config.llm.model_of()` in
        // Python would contradict it one command later.
        "model" => notice(
            "the model comes from the backend's config; edit config.toml and restart".into(),
        ),
        // Not a known command: the whole line goes to the model verbatim,
        // exactly like the REPL treats "/whatever" as a message.
        _ => Translation::Command(json!({"cmd": "submit", "text": text})),
    }
}

/// Start the backend task. The task owns the child (`kill_on_drop`), so when
/// the UI drops its sender — quit, or a panic on the UI thread — the process
/// is reaped, and when the child dies the event channel closes.
pub fn spawn() -> (mpsc::UnboundedSender<AgentMsg>, mpsc::UnboundedReceiver<AgentEvent>) {
    let (msg_tx, msg_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    tokio::spawn(run(msg_rx, event_tx, find_root()));
    (msg_tx, event_rx)
}

/// A decoded stdout line, or one that failed to decode — skipped, with the
/// reason reported to the transcript rather than silently swallowed.
enum Line {
    Event(Value),
    Garbage(String),
}

/// One stdout line → an event or the garbage report. Split out so the codec's
/// two branches are testable without spawning a child.
fn decode_line(line: &str) -> Line {
    match serde_json::from_str::<Value>(line) {
        Ok(value) => Line::Event(value),
        Err(e) => Line::Garbage(format!("{line} — {e}")),
    }
}

async fn run(
    mut msg_rx: mpsc::UnboundedReceiver<AgentMsg>,
    event_tx: mpsc::UnboundedSender<AgentEvent>,
    root: PathBuf,
) {
    let argv = resolve_argv(&root);
    let log_path = root.join("data/logs/lumi-tui-backend.log");
    let stderr = match open_append(&log_path) {
        Ok(file) => Stdio::from(file),
        Err(_) => Stdio::null(),
    };
    let child = Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(&root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(stderr)
        .kill_on_drop(true)
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(e) => {
            let _ = event_tx.send(AgentEvent::Error(format!(
                "cannot start the lumi backend: {e}\nargv: {}",
                argv.join(" ")
            )));
            return;
        }
    };
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        let _ = event_tx.send(AgentEvent::Error("backend has no stdio pipes".into()));
        return;
    };

    let (line_tx, mut line_rx) = mpsc::unbounded_channel::<Line>();
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }
            let decoded = decode_line(&line);
            if line_tx.send(decoded).is_err() {
                break;
            }
        }
        // EOF on the child's stdout: the backend died. Closing the channel
        // lets the loop below reap it and report the exit status.
    });

    let mut stdin = stdin;
    let mut state = MapState::default();
    let exit: Exit = loop {
        tokio::select! {
            msg = msg_rx.recv() => {
                let Some(msg) = msg else { break Exit::UiQuit };
                let (command, start_turn) = match msg {
                    AgentMsg::Submit(text) => match translate(&text) {
                        Translation::Local(events) => {
                            for event in events {
                                let _ = event_tx.send(event);
                            }
                            continue;
                        }
                        Translation::Command(value) => {
                            let turn = starts_turn(&value);
                            (value, turn)
                        }
                    },
                    AgentMsg::Cancel => (json!({"cmd": "cancel"}), false),
                    AgentMsg::Approval { id, approved } => {
                        // Settled either way: the resolve exchange that
                        // follows ends in a turn_done that may clear busy.
                        state.outstanding.retain(|seen| *seen != id);
                        (json!({"cmd": "approve", "id": id, "ok": approved}), true)
                    }
                };
                if write_line(&mut stdin, &command).await.is_none() {
                    break Exit::ChildGone; // our write hit a dead pipe
                }
                if start_turn {
                    let _ = event_tx.send(AgentEvent::Busy(true));
                }
            }
            line = line_rx.recv() => {
                let Some(line) = line else { break Exit::ChildGone }; // stdout EOF
                match line {
                    Line::Event(value) => {
                        for event in map_event(&value, &mut state) {
                            let _ = event_tx.send(event);
                        }
                    }
                    Line::Garbage(raw) => {
                        let _ = event_tx.send(AgentEvent::Error(
                            format!("backend sent a line that is not JSON: {raw}")));
                    }
                }
            }
        }
    };

    let status = match exit {
        Exit::UiQuit => {
            // Let the loop close its own registry on a clean quit; if the
            // child is already gone the write just fails.
            let _ = write_line(&mut stdin, &json!({"cmd": "quit"})).await;
            drop(stdin);
            let _ = child.wait().await;
            return; // no report: the UI left on purpose
        }
        Exit::ChildGone => {
            drop(stdin);
            child.wait().await
        }
    };
    // §10: EOF on the child's stdout is fatal-but-clean. Report the exit
    // status and the last log lines, then close the channel; main.rs sees
    // the disconnect and restores the terminal before printing.
    let tail = read_tail(&log_path, 6).join("\n");
    let status = status.map(|s| s.to_string()).unwrap_or_else(|e| format!("wait: {e}"));
    let mut note = format!("lumi backend exited ({status})");
    if !tail.is_empty() {
        note.push_str(&format!(" — last log lines:\n{tail}"));
    }
    let _ = event_tx.send(AgentEvent::Error(note));
    // event_tx drops here: the UI sees the channel close and takes the
    // terminal down with it.
}

/// Why the backend loop ended: the UI walked away, or the child did.
enum Exit {
    UiQuit,
    ChildGone,
}

async fn write_line(stdin: &mut tokio::process::ChildStdin, value: &Value) -> Option<()> {
    stdin.write_all(format!("{value}\n").as_bytes()).await.ok()?;
    stdin.flush().await.ok()?;
    Some(())
}

fn open_append(path: &Path) -> Result<std::fs::File> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    Ok(std::fs::OpenOptions::new().create(true).append(true).open(path)?)
}

/// The last few lines of the backend log, for the death report.
fn read_tail(path: &Path, count: usize) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let mut tail: Vec<String> = text.lines().rev().take(count).map(str::to_string).collect();
    tail.reverse();
    tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(lines: &[&str]) -> (Vec<AgentEvent>, MapState) {
        let mut st = MapState::default();
        let mut got = Vec::new();
        for line in lines {
            let value: Value = serde_json::from_str(line).unwrap();
            got.extend(map_event(&value, &mut st));
        }
        (got, st)
    }

    // ---- command translation ------------------------------------------------

    #[test]
    fn submit_is_one_json_line_with_cmd_and_text() {
        let Translation::Command(value) = translate("hello") else {
            panic!("plain text must go to the backend");
        };
        let round: Value = serde_json::from_str(&value.to_string()).unwrap();
        assert_eq!(round["cmd"], "submit");
        assert_eq!(round["text"], "hello");
    }

    #[test]
    fn slash_commands_map_onto_protocol_commands() {
        for (line, cmd) in [
            ("/reset", "reset"),
            ("/clear", "reset"),
            ("/tools", "tools"),
            ("/context", "context"),
            ("/memory", "memory"),
        ] {
            let Translation::Command(value) = translate(line) else {
                panic!("{line} must reach the backend");
            };
            assert_eq!(value["cmd"], cmd, "{line}");
        }
        let Translation::Command(off) = translate("/reasoning off") else { panic!() };
        assert_eq!(off["on"], json!(false));
        let Translation::Command(on) = translate("/REASONING ON") else { panic!() };
        assert_eq!(on["on"], json!(true));
        let Translation::Command(search) = translate("/search rust tokio") else { panic!() };
        assert_eq!(search["cmd"], "search");
        assert_eq!(search["query"], "rust tokio");
    }

    #[test]
    fn unknown_slash_still_goes_to_the_model_verbatim() {
        let Translation::Command(value) = translate("/notacommand hi") else { panic!() };
        assert_eq!(value["cmd"], "submit");
        assert_eq!(value["text"], "/notacommand hi");
    }

    #[test]
    fn frontend_side_commands_never_reach_the_stream() {
        let Translation::Local(events) = translate("/help") else { panic!() };
        assert!(matches!(&events[..], [AgentEvent::System(h), AgentEvent::TurnDone] if h.contains("/help")));
        let Translation::Local(events) = translate("/model gpt-5") else { panic!() };
        assert!(matches!(&events[..], [AgentEvent::Notice(n), AgentEvent::TurnDone] if n.contains("backend")));
        let Translation::Local(events) = translate("/reasoning") else { panic!() };
        assert!(matches!(&events[..], [AgentEvent::Notice(n), AgentEvent::TurnDone] if n.contains("on|off")));
    }

    // ---- event mapping ------------------------------------------------------

    #[test]
    fn a_plain_turn_maps_to_the_documented_sequence() {
        let (got, st) = map(&[
            r#"{"ev":"ready","model":"m","host":"https://h.example/v1","tools":["web"],"search":"on","turns":0}"#,
            r#"{"ev":"step","kind":"thinking","detail":""}"#,
            r#"{"ev":"result","text":"hi","error":"","tools_used":[],"artifacts":[],"elapsed":1.5}"#,
            r#"{"ev":"turn_done"}"#,
        ]);
        assert!(matches!(&got[0], AgentEvent::Ready { model, host, search }
            if model == "m" && host == "h.example/v1" && search == "on"));
        assert!(matches!(&got[1..],
            [AgentEvent::Delta(t), AgentEvent::AssistantDone, AgentEvent::Busy(false), AgentEvent::TurnDone]
            if t == "hi"));
        assert!(st.outstanding.is_empty() && !st.tool_open);
    }

    #[test]
    fn tool_steps_open_and_close_in_order() {
        let (got, _) = map(&[
            r#"{"ev":"step","kind":"tool","detail":"run_shell"}"#,
            r#"{"ev":"step","kind":"tool","detail":"web"}"#,
            r#"{"ev":"result","text":"done","error":"","tools_used":[],"artifacts":[],"elapsed":2.0}"#,
            r#"{"ev":"turn_done"}"#,
        ]);
        assert!(matches!(
            &got[..],
            [
                AgentEvent::ToolStart { name, .. },
                AgentEvent::ToolEnd { .. },
                AgentEvent::ToolStart { name: second, .. },
                AgentEvent::ToolEnd { .. },
                AgentEvent::Delta(_),
                AgentEvent::AssistantDone,
                AgentEvent::Busy(false),
                AgentEvent::TurnDone,
            ]
            if name == "run_shell" && second == "web"
        ));
    }

    #[test]
    fn approvals_are_answered_by_id_and_hold_busy() {
        let (got, st) = map(&[
            r#"{"ev":"approval","id":"a1","tool":"run_shell","preview":"rm x","reason":"destructive"}"#,
            r#"{"ev":"approval","id":"a2","tool":"files","preview":"write y","reason":"outside"}"#,
            r#"{"ev":"result","text":"","error":"","tools_used":[],"artifacts":[],"elapsed":0.1}"#,
            r#"{"ev":"turn_done"}"#,
        ]);
        assert!(matches!(&got[0], AgentEvent::ApprovalRequest { id, .. } if id == "a1"));
        assert!(matches!(&got[1], AgentEvent::ApprovalRequest { id, .. } if id == "a2"));
        assert_eq!(st.outstanding, vec!["a1".to_string(), "a2".to_string()]);
        // turn_done with approvals outstanding must NOT clear busy: the
        // exchange continues when each is answered.
        assert!(matches!(&got.last().unwrap(), AgentEvent::TurnDone));
        assert!(!got.iter().any(|e| matches!(e, AgentEvent::Busy(false))));
    }

    #[test]
    fn a_re_reported_approval_id_is_not_a_second_question() {
        // The next exchange's TurnResult carries the whole pending list; the
        // unanswered second action arrives again with the same id.
        let (got, st) = map(&[
            r#"{"ev":"approval","id":"a2","tool":"files","preview":"p","reason":"r"}"#,
            r#"{"ev":"approval","id":"a2","tool":"files","preview":"p","reason":"r"}"#,
        ]);
        assert_eq!(got.len(), 1);
        assert_eq!(st.outstanding, vec!["a2".to_string()]);
    }

    #[test]
    fn mid_turn_messages_close_the_open_tool_and_land_as_text() {
        let (got, _) = map(&[
            r#"{"ev":"step","kind":"tool","detail":"web"}"#,
            r#"{"ev":"message","text":"phase 1 done"}"#,
        ]);
        assert!(matches!(&got[1..],
            [AgentEvent::ToolEnd { .. }, AgentEvent::Delta(t), AgentEvent::AssistantDone]
            if t == "phase 1 done"));
    }

    #[test]
    fn notices_split_on_newlines_and_errors_and_clears_map() {
        let (got, _) = map(&[
            r#"{"ev":"notice","text":"live steps off"}"#,
            r#"{"ev":"notice","text":"- `a` — x\n- `b` — y"}"#,
            r#"{"ev":"error","text":"malformed line: boom"}"#,
            r#"{"ev":"cleared"}"#,
        ]);
        assert!(matches!(&got[0], AgentEvent::Notice(t) if t == "live steps off"));
        assert!(matches!(&got[1], AgentEvent::System(_)));
        assert!(matches!(&got[2], AgentEvent::Error(t) if t == "malformed line: boom"));
        assert!(matches!(&got[3], AgentEvent::ConversationCleared));
        assert!(matches!(&got[4], AgentEvent::Notice(_)));
    }

    #[test]
    fn only_submit_and_approve_claim_busy() {
        for line in ["/tools", "/context", "/memory", "/reset", "/reasoning on", "/search x"] {
            let Translation::Command(value) = translate(line) else {
                panic!("{line} must reach the backend");
            };
            assert!(!starts_turn(&value), "{line} is not a turn");
        }
        let Translation::Command(value) = translate("hello") else { panic!() };
        assert!(starts_turn(&value));
        assert!(starts_turn(&json!({"cmd": "approve", "id": "a1", "ok": true})));
    }

    #[test]
    fn results_carry_their_errors_and_files() {
        let (got, _) = map(&[
            r#"{"ev":"result","text":"","error":"provider exploded","tools_used":[],"artifacts":[{"path":"p.png","size":12,"mime":"image/png"}],"elapsed":0.3}"#,
        ]);
        assert!(matches!(&got[0], AgentEvent::Notice(t) if t == "[file: p.png (12 bytes)]"));
        assert!(matches!(&got[1], AgentEvent::Error(t) if t == "provider exploded"));
    }

    #[test]
    fn a_streamed_result_does_not_replay_text_the_deltas_already_carried() {
        // §7b: `streamed: true` says `result.text` went out as `delta` events.
        // Rendering both would show the answer twice.
        let (got, _) = map(&[
            r#"{"ev":"delta","text":"Hel"}"#,
            r#"{"ev":"delta","text":"lo"}"#,
            r#"{"ev":"result","text":"Hello","streamed":true,"error":"","tools_used":[],"artifacts":[],"elapsed":1.0}"#,
            r#"{"ev":"turn_done"}"#,
        ]);
        assert!(matches!(&got[..],
            [AgentEvent::Delta(a), AgentEvent::Delta(b), AgentEvent::Busy(false), AgentEvent::TurnDone]
            if a == "Hel" && b == "lo"));
    }

    #[test]
    fn an_unstreamed_result_still_delivers_its_text_whole() {
        // A backend that does not stream (or an older one, without the
        // `streamed` field) must keep working exactly as before.
        let (got, _) = map(&[
            r#"{"ev":"result","text":"hi","error":"","tools_used":[],"artifacts":[],"elapsed":1.0}"#,
        ]);
        assert!(matches!(&got[..],
            [AgentEvent::Delta(t), AgentEvent::AssistantDone] if t == "hi"));
    }

    #[test]
    fn thinking_streams_between_the_tool_step_and_the_answer() {
        let (got, _) = map(&[
            r#"{"ev":"step","kind":"tool","detail":"run_shell"}"#,
            r#"{"ev":"thinking_delta","text":"hmm"}"#,
            r#"{"ev":"delta","text":"done"}"#,
            r#"{"ev":"turn_done"}"#,
        ]);
        assert!(matches!(&got[..],
            [
                AgentEvent::ToolStart { .. },
                AgentEvent::ToolEnd { .. },
                AgentEvent::ThinkingDelta(t),
                AgentEvent::Delta(d),
                AgentEvent::AssistantDone,
                AgentEvent::Busy(false),
                AgentEvent::TurnDone,
            ]
            if t == "hmm" && d == "done"));
    }

    #[test]
    fn a_turn_that_ends_mid_thinking_switches_the_cursor_off() {
        // Error, cap, cancellation: whatever ends the turn, a live trace
        // must not keep its spinner after `turn_done`.
        let (got, _) = map(&[
            r#"{"ev":"thinking_delta","text":"hmm"}"#,
            r#"{"ev":"error","text":"provider exploded"}"#,
            r#"{"ev":"turn_done"}"#,
        ]);
        assert!(matches!(&got[..],
            [
                AgentEvent::ThinkingDelta(_),
                AgentEvent::Error(t),
                AgentEvent::AssistantDone,
                AgentEvent::Busy(false),
                AgentEvent::TurnDone,
            ]
            if t == "provider exploded"));
    }

    #[test]
    fn a_malformed_value_and_an_unknown_ev_never_panic() {
        // Well-formed JSON that is no event at all must be survivable: this
        // is the forward-compat contract with a newer backend.
        for line in [
            r#"{"not":"an event"}"#,
            r#"{"ev":"from_the_future","stuff":{"a":1}}"#,
            r#"{"ev":"result"}"#,
            r#"{"ev":"step","kind":42}"#,
            r#"{"ev":"approval","id":null}"#,
            r#"42"#,
            r#"[1,2,3]"#,
            r#""a string""#,
            r#"null"#,
        ] {
            let value: Value = serde_json::from_str(line).unwrap();
            let mut st = MapState::default();
            let _ = map_event(&value, &mut st);
        }
    }

    #[test]
    fn the_codec_accepts_json_and_reports_what_is_not_json() {
        match decode_line(r#"{"ev":"ready","model":"m"}"#) {
            Line::Event(value) => assert_eq!(value["ev"], "ready"),
            Line::Garbage(raw) => panic!("valid JSON was rejected: {raw}"),
        }
        match decode_line("backend noise, not JSON") {
            Line::Garbage(raw) => {
                assert!(raw.starts_with("backend noise, not JSON"));
                assert!(raw.contains(" — "), "the parse error is kept for the log");
            }
            Line::Event(value) => panic!("invalid JSON was accepted as {value}"),
        }
    }

    #[test]
    fn reset_clears_the_outstanding_queue() {
        let mut st = MapState::default();
        st.outstanding.push("a1".into());
        map_event(&json!({"ev": "cleared"}), &mut st);
        assert!(st.outstanding.is_empty());
    }

    #[test]
    fn truncate_cuts_on_chars_not_bytes() {
        assert_eq!(truncate("héllo", 3), "hél…");
        assert_eq!(truncate("abc", 5), "abc");
    }

    #[test]
    fn the_dev_override_appends_the_subcommand() {
        // resolve_argv reads the real process env, so exercise the rule
        // through a root with no venv: PATH lookup or the last resort.
        let argv = resolve_argv(Path::new("/nonexistent-root-xyz"));
        assert_eq!(argv.last().map(String::as_str), Some("tui"));
        assert!(argv.len() >= 2);
    }
}
