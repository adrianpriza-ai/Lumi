//! The agent task: owns the conversation, drives the model, executes tools.
//! The UI thread talks to it over two channels and never blocks on the model.

use crate::config::Config;
use crate::llm::{ChatMessage, LlmClient, ToolDef};
use crate::search::Search;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Notify};

/// UI → agent.
#[derive(Debug)]
pub enum AgentMsg {
    /// Raw composer input; slash commands are handled agent-side.
    Submit(String),
    Cancel,
}

/// Agent → UI.
#[derive(Debug)]
pub enum AgentEvent {
    /// Assistant text delta.
    Delta(String),
    /// Reasoning delta (dimmed while the model thinks).
    ThinkingDelta(String),
    /// A tool call begins.
    ToolStart { name: String, detail: String },
    /// The same call finished; `summary` is one short line.
    ToolEnd { summary: String, ok: bool },
    /// A whole text block is done (streaming cursor off).
    AssistantDone,
    /// The agent-side conversation was reset; the transcript follows suit.
    ConversationCleared,
    /// Markdown-rendered system message block (help, search results, notes).
    System(String),
    /// Plain status line (slash-command replies).
    Notice(String),
    Error(String),
    /// The `responder` is the direct reply path: the agent is inside the turn
    /// future and not polling its message channel, so the answer cannot come
    /// back through `AgentMsg` — it would deadlock.
    ApprovalRequest { command: String, responder: oneshot::Sender<bool> },
    Busy(bool),
    ModelChanged(String),
    /// Agent finished a turn and the conversation state is settled.
    TurnDone,
}

pub struct Agent {
    config: Config,
    client: LlmClient,
    search: Search,
    conversation: Vec<ChatMessage>,
    model: String,
    show_reasoning: bool,
    event_tx: mpsc::UnboundedSender<AgentEvent>,
    msg_rx: mpsc::UnboundedReceiver<AgentMsg>,
    cancel: Arc<Notify>,
}

const MAX_TOOL_OUTPUT: usize = 8_000;
const MAX_SHELL_OUTPUT: usize = 6_000;

pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "web_search",
            description:
                "Search the web (Tavily, fallback Firecrawl). Returns titled, linked snippets.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" },
                    "max_results": { "type": "integer", "description": "1-10, default 5" }
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "fetch_page",
            description:
                "Fetch one web page and get its content as markdown (via Firecrawl).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Absolute http(s) URL" }
                },
                "required": ["url"]
            }),
        },
        ToolDef {
            name: "shell",
            description:
                "Run a shell command in the project directory. Always asks the user for \
                 approval before running.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run" }
                },
                "required": ["command"]
            }),
        },
        ToolDef {
            name: "read_file",
            description:
                "Read a file from the project directory as UTF-8 text. Paths are relative \
                 to the project root; escaping it is refused.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Relative path in the project" }
                },
                "required": ["path"]
            }),
        },
    ]
}

impl Agent {
    pub fn spawn(
        config: Config,
    ) -> (mpsc::UnboundedSender<AgentMsg>, mpsc::UnboundedReceiver<AgentEvent>) {
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let agent = Self {
            client: LlmClient::new(&config.base_url, config.api_key.clone()),
            search: Search::new(config.tavily_keys.clone(), config.firecrawl_keys.clone()),
            conversation: Vec::new(),
            model: config.model.clone(),
            show_reasoning: config.show_reasoning,
            event_tx,
            msg_rx,
            cancel: Arc::new(Notify::new()),
            config,
        };
        tokio::spawn(async move { agent.run().await });
        (msg_tx, event_rx)
    }

    async fn run(mut self) {
        self.conversation.push(ChatMessage::system(self.system_prompt()));
        while let Some(msg) = self.msg_rx.recv().await {
            match msg {
                AgentMsg::Submit(input) => {
                    if let Some(cmd) = input.strip_prefix('/') {
                        if self.handle_command(cmd).await {
                            continue; // consumed as a command
                        }
                    }
                    let _ = self.event_tx.send(AgentEvent::Busy(true));
                    let cancel = self.cancel.clone();
                    let turn = self.turn(&input);
                    tokio::select! {
                        _ = cancel.notified() => {
                            self.seal_dangling_tool_calls();
                            let _ = self.event_tx.send(AgentEvent::Notice("turn cancelled".into()));
                        }
                        result = turn => {
                            if let Err(e) = result {
                                let _ = self.event_tx.send(AgentEvent::Error(format!("{e:#}")));
                            }
                        }
                    }
                    self.trim_history();
                    let _ = self.event_tx.send(AgentEvent::Busy(false));
                    let _ = self.event_tx.send(AgentEvent::TurnDone);
                }
                AgentMsg::Cancel => {
                    // only meaningful mid-turn; the Notify wakes the select
                    self.cancel.notify_waiters();
                }
            }
        }
    }

    fn system_prompt(&self) -> String {
        let mut prompt = String::new();
        let personality = self.config.root.join("PERSONALITY.md");
        if let Ok(text) = std::fs::read_to_string(&personality) {
            prompt.push_str(text.trim());
            prompt.push_str("\n\n");
        }
        let memory = self
            .config
            .root
            .join("data/memory/cli/MEMORY.md");
        if let Ok(text) = std::fs::read_to_string(&memory) {
            if !text.trim().is_empty() {
                prompt.push_str("# Memory (cli scope)\n\n");
                prompt.push_str(text.trim());
                prompt.push_str("\n\n");
            }
        }
        prompt.push_str(&format!(
            "# Environment\n\
             - You are Lumi, running in the `lumi-tui` terminal UI.\n\
             - Project root: {}\n\
             - Today: {}\n\
             - Tools: web_search (Tavily/Firecrawl), fetch_page (Firecrawl),\n\
             shell (asks the user for approval first), read_file (project root only).\n\
             Use web_search for anything time-sensitive or outside your knowledge;\n\
             use fetch_page to read a page the search surfaced; prefer answering\n\
             from search snippets before scraping full pages.\n\
             Format answers in markdown.",
            self.config.root.display(),
            chrono_today(),
        ));
        prompt
    }

    /// Returns true when the slash command was consumed.
    async fn handle_command(&mut self, cmd: &str) -> bool {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let verb = parts.next().unwrap_or("").trim().to_ascii_lowercase();
        let rest = parts.next().unwrap_or("").trim().to_string();
        let _ = self.event_tx.send(AgentEvent::Busy(true));
        match verb.as_str() {
            "quit" | "exit" | "q" => {
                let _ = self.event_tx.send(AgentEvent::Notice("bye — press ctrl-c to close"
                    .into()));
            }
            "help" | "?" => {
                let _ = self.event_tx.send(AgentEvent::System(crate::help_text()));
            }
            "reset" | "clear" => {
                self.conversation.clear();
                self.conversation.push(ChatMessage::system(self.system_prompt()));
                let _ = self.event_tx.send(AgentEvent::ConversationCleared);
                let _ = self
                    .event_tx
                    .send(AgentEvent::Notice("conversation cleared".into()));
            }
            "model" => {
                if rest.is_empty() {
                    let _ = self
                        .event_tx
                        .send(AgentEvent::Notice(format!("model: {}", self.model)));
                } else {
                    self.model = rest.clone();
                    let _ = self.event_tx.send(AgentEvent::ModelChanged(rest));
                }
            }
            "reasoning" => match rest.as_str() {
                "on" => {
                    self.show_reasoning = true;
                    let _ = self.event_tx.send(AgentEvent::Notice("live thinking on".into()));
                }
                "off" => {
                    self.show_reasoning = false;
                    let _ = self.event_tx.send(AgentEvent::Notice("live thinking off".into()));
                }
                _ => {
                    let _ = self.event_tx.send(AgentEvent::Notice(format!(
                        "live thinking is {} — /reasoning on|off",
                        if self.show_reasoning { "on" } else { "off" }
                    )));
                }
            },
            "tools" => {
                let names: Vec<&str> = tool_defs().iter().map(|t| t.name).collect();
                let _ = self.event_tx.send(AgentEvent::Notice(format!(
                    "tools: {} · search: {}",
                    names.join(", "),
                    self.config.search_status()
                )));
            }
            "search" => {
                if rest.is_empty() {
                    let _ = self
                        .event_tx
                        .send(AgentEvent::Notice("usage: /search <query>".into()));
                } else {
                    match self.run_search_async(&rest, 5).await {
                        Ok(markdown) => {
                            let _ = self.event_tx.send(AgentEvent::System(markdown));
                        }
                        Err(e) => {
                            let _ = self.event_tx.send(AgentEvent::Error(format!("{e:#}")));
                        }
                    }
                }
            }
            _ => {
                // Not a known command: treat the whole thing as a message that
                // happens to start with '/' (the model still sees it verbatim).
                let _ = self.event_tx.send(AgentEvent::Busy(false));
                return false;
            }
        }
        let _ = self.event_tx.send(AgentEvent::Busy(false));
        let _ = self.event_tx.send(AgentEvent::TurnDone);
        true
    }

    /// One full agent turn: stream the model, run tools, repeat until the
    /// model answers without tool calls (or the iteration cap trips).
    async fn turn(&mut self, input: &str) -> Result<()> {
        self.conversation.push(ChatMessage::user(input));

        for _ in 0..self.config.max_tool_iterations {
            let mut streamed_text = false;
            let mut streamed_thinking = false;
            let message = self
                .client
                .stream(&self.model, &self.conversation, &tool_defs(), &mut |update| {
                    if !update.content_delta.is_empty() {
                        streamed_text = true;
                        let _ = self.event_tx.send(AgentEvent::Delta(update.content_delta));
                    }
                    if !update.reasoning_delta.is_empty() && self.show_reasoning {
                        streamed_thinking = true;
                        let _ =
                            self.event_tx
                                .send(AgentEvent::ThinkingDelta(update.reasoning_delta));
                    }
                    let _ = (streamed_text, streamed_thinking);
                })
                .await?;

            match message.tool_calls.clone() {
                Some(calls) if !calls.is_empty() => {
                    // Keep whatever text preceded the calls, then record them.
                    self.conversation.push(message);
                    for call in calls {
                        let result = self.execute_tool(&call).await;
                        self.conversation.push(ChatMessage::tool_result(
                            &call.id,
                            &call.function.name,
                            result,
                        ));
                    }
                }
                _ => {
                    let text = message.content.trim().to_string();
                    self.conversation.push(message);
                    let _ = self.event_tx.send(AgentEvent::AssistantDone);
                    if text.is_empty() {
                        return Err(anyhow!("empty answer from {}", self.model));
                    }
                    return Ok(());
                }
            }
        }
        Err(anyhow!(
            "gave up after {} tool round trips",
            self.config.max_tool_iterations
        ))
    }

    async fn execute_tool(&mut self, call: &crate::llm::ToolCall) -> String {
        let name = call.function.name.clone();
        let args: Value = serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));
        let detail = match name.as_str() {
            "web_search" => args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            "fetch_page" => args.get("url").and_then(Value::as_str).unwrap_or("").to_string(),
            "shell" => args
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            "read_file" => args.get("path").and_then(Value::as_str).unwrap_or("").to_string(),
            _ => String::new(),
        };
        let _ = self
            .event_tx
            .send(AgentEvent::ToolStart { name: name.clone(), detail: detail.clone() });

        let started = std::time::Instant::now();
        let outcome: Result<String> = match name.as_str() {
            "web_search" => {
                let query = detail.clone();
                let max = args
                    .get("max_results")
                    .and_then(Value::as_u64)
                    .unwrap_or(5) as usize;
                self.run_search_async(&query, max).await
            }
            "fetch_page" => self.run_fetch(&detail).await,
            "shell" => self.run_shell(&detail).await,
            "read_file" => self.run_read(&detail),
            other => Err(anyhow!("unknown tool {other:?}")),
        };
        let elapsed = started.elapsed().as_secs_f32();

        let (ok, summary, payload) = match outcome {
            Ok(text) => {
                let first_line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("ok");
                let summary = match name.as_str() {
                    "web_search" => format!(
                        "{} hit{} · {:.1}s",
                        text.lines()
                            .filter(|l| {
                                let mut chars = l.chars();
                                matches!(chars.next(), Some('0'..='9'))
                                    && matches!(chars.next(), Some('.'))
                                    && matches!(chars.next(), Some(' '))
                            })
                            .count(),
                        "s",
                        elapsed
                    ),
                    "fetch_page" => format!("{} chars · {:.1}s", text.len(), elapsed),
                    _ => crate::llm::truncate(first_line, 60),
                };
                (true, summary, text)
            }
            Err(e) => (
                false,
                format!("failed · {:.1}s", elapsed),
                {
                    // A few lines of the real error reach both the model and
                    // the tool line, so a bad URL is visible without digging.
                    let text = format!("error: {e:#}");
                    text.lines().take(3).collect::<Vec<_>>().join("\n")
                },
            ),
        };
        let _ = self.event_tx.send(AgentEvent::ToolEnd { summary, ok });
        if payload.len() > MAX_TOOL_OUTPUT {
            let cut = crate::llm::truncate(&payload, MAX_TOOL_OUTPUT);
            format!("{cut}\n[truncated]")
        } else {
            payload
        }
    }

    async fn run_search_async(&self, query: &str, max: usize) -> Result<String> {
        let hits = self.search.web_search(query, max).await?;
        let mut out = String::new();
        for (i, hit) in hits.iter().enumerate() {
            out.push_str(&format!(
                "{}. [{}]({})\n   {}\n\n",
                i + 1,
                crate::llm::truncate(&hit.title, 90),
                hit.url,
                crate::llm::truncate(hit.snippet.replace('\n', " ").trim(), 280)
            ));
        }
        if out.is_empty() {
            out.push_str("no results\n");
        }
        Ok(out)
    }

    async fn run_fetch(&self, url: &str) -> Result<String> {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(anyhow!("url must start with http:// or https://"));
        }
        let markdown = self.search.fetch_page(url).await?;
        Ok(markdown)
    }

    async fn run_shell(&mut self, command: &str) -> Result<String> {
        if command.trim().is_empty() {
            return Err(anyhow!("empty command"));
        }
        let (tx, rx) = oneshot::channel();
        let _ = self
            .event_tx
            .send(AgentEvent::ApprovalRequest { command: command.to_string(), responder: tx });

        let approved = rx
            .await
            .map_err(|_| anyhow!("approval channel dropped"))?;
        if !approved {
            return Ok("the user declined to run this command".to_string());
        }

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.config.root)
            .output()
            .await
            .map_err(|e| anyhow!("spawn: {e}"))?;
        let mut text = format!(
            "exit code: {}\nstdout:\n{}\nstderr:\n{}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if text.len() > MAX_SHELL_OUTPUT {
            text = format!("{}\n[truncated]", crate::llm::truncate(&text, MAX_SHELL_OUTPUT));
        }
        Ok(text)
    }

    fn run_read(&self, path: &str) -> Result<String> {
        let full = crate::config::safe_join(&self.config.root, path)?;
        let text = std::fs::read_to_string(&full).map_err(|e| anyhow!("{}: {e}", path))?;
        let mut text = text;
        if text.len() > MAX_TOOL_OUTPUT {
            text = format!("{}\n[truncated]", crate::llm::truncate(&text, MAX_TOOL_OUTPUT));
        }
        Ok(text)
    }

    /// After a mid-turn cancellation the future may have died between pushing
    /// an assistant message with `tool_calls` and pushing its results. Replay
    /// would then be protocol-invalid, so close any unanswered calls with a
    /// synthetic result.
    fn seal_dangling_tool_calls(&mut self) {
        let Some(idx) = self
            .conversation
            .iter()
            .rposition(|m| m.tool_calls.as_ref().is_some_and(|c| !c.is_empty()))
        else {
            return;
        };
        let answered = self.conversation[idx + 1..]
            .iter()
            .filter_map(|m| m.tool_call_id.clone())
            .collect::<Vec<_>>();
        let missing: Vec<(String, String)> = self.conversation[idx]
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .filter(|c| !answered.iter().any(|id| id == &c.id))
            .map(|c| (c.id.clone(), c.function.name.clone()))
            .collect();
        for (id, name) in missing {
            self.conversation
                .push(ChatMessage::tool_result(&id, &name, "cancelled by user"));
        }
    }

    /// Drop oldest non-system turns when the replay grows huge, mirroring the
    /// Python context manager's oldest-first policy in a cruder form.
    fn trim_history(&mut self) {
        const MAX_MESSAGES: usize = 120;
        if self.conversation.len() <= MAX_MESSAGES {
            return;
        }
        let excess = self.conversation.len() - MAX_MESSAGES;
        // Never drop index 0 (the system prompt) and never orphan a tool
        // result away from its call: cut from the first user/assistant pair.
        let start = 1 + excess.min(self.conversation.len() - 2);
        self.conversation.drain(1..start);
    }
}

fn chrono_today() -> String {
    // Avoid pulling chrono for one date string.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let (year, month, day) = civil_from_days(days as i64);
    format!("{year:04}-{month:02}-{day:02}")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}
