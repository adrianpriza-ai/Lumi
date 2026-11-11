//! UI-side state: the transcript as raw blocks plus a cached rendering, the
//! multi-line composer, scroll tracking, and the pending approval prompt.

use crate::backend::{self, AgentEvent};
use crate::markdown::{render, Role};
use crate::theme;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use std::collections::VecDeque;
use std::time::Instant;

/// One question from the backend: a tool that stopped, waiting for yes or no.
/// Answered by id — several can be queued, and the modal shows the head.
#[derive(Debug, Clone)]
pub struct Approval {
    pub id: String,
    pub tool: String,
    pub preview: String,
    pub reason: String,
}

/// One entry inside a trace: a chunk of reasoning, or a tool call.
#[derive(Debug, Clone)]
pub enum TraceItem {
    Thinking(String),
    Tool { name: String, detail: String, done: Option<(String, bool)> },
}

/// One transcript entry, kept raw so it can be re-wrapped on resize.
#[derive(Debug, Clone)]
pub enum Block {
    Message { role: Role, text: String },
    /// A run of reasoning and tool calls between two messages. The whole run
    /// is one line by default — a long turn should read as `3 steps ·
    /// thinking, web_search`, not as a wall of italic. `ctrl+t` opens it.
    Trace { items: Vec<TraceItem>, live: bool },
}

#[derive(Debug, Clone)]
pub enum Submit {
    Send(String),
    Nothing,
}

pub struct Composer {
    pub lines: Vec<String>,
    pub row: usize,
    pub col: usize,
    history: Vec<String>,
    history_idx: Option<usize>,
    stash: Option<Vec<String>>,
}

impl Default for Composer {
    fn default() -> Self {
        Self {
            lines: vec![String::new()],
            row: 0,
            col: 0,
            history: Vec::new(),
            history_idx: None,
            stash: None,
        }
    }
}

impl Composer {
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(|l| l.trim().is_empty())
    }

    pub fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.row = 0;
        self.col = 0;
        self.history_idx = None;
        self.stash = None;
    }

    /// Insert a hard newline (Shift+Enter, Alt+Enter, Ctrl+J).
    pub fn newline(&mut self) {
        let line = &mut self.lines[self.row];
        let col = char_col(line, self.col);
        let rest = line[col..].to_string();
        line.truncate(col);
        self.lines.insert(self.row + 1, rest);
        self.row += 1;
        self.col = 0;
    }

    /// Insert pasted text; newlines in the paste become real lines. The line
    /// under the cursor becomes `head + paste + tail`, split on the paste's
    /// own newlines — the same model every text editor uses.
    pub fn insert_str(&mut self, text: &str) {
        let chunk = text.replace('\r', "");
        let trailing = chunk.ends_with('\n');
        let chunk = chunk.trim_end_matches('\n');
        let parts: Vec<String> = chunk.split('\n').map(str::to_string).collect();

        let mut lines = self.lines.clone();
        let col = char_col(&lines[self.row], self.col);
        let mut joined = lines[self.row][..col].to_string();
        joined.push_str(&parts.join("\n"));
        if trailing {
            joined.push('\n');
        }
        joined.push_str(&lines[self.row][col..]);
        let new_lines: Vec<String> = joined.split('\n').map(str::to_string).collect();
        let row = self.row;
        lines.splice(row..=row, new_lines);
        self.lines = lines;

        let inserted = parts.len() + trailing as usize;
        self.row = row + inserted - 1;
        self.col = if trailing {
            0
        } else {
            parts.last().map(|p| p.chars().count()).unwrap_or(0)
        };
    }

    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> Submit {
        use crossterm::event::{KeyCode, KeyModifiers};
        let newline_mod = key
            .modifiers
            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT);
        match key.code {
            KeyCode::Enter if newline_mod => {
                self.newline();
                Submit::Nothing
            }
            KeyCode::Enter => {
                if self.is_empty() {
                    return Submit::Nothing;
                }
                let text = self.text();
                self.history.push(text.clone());
                self.clear();
                Submit::Send(text)
            }
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.newline();
                Submit::Nothing
            }
            // Emacs-ish control keys; anything else with Ctrl is swallowed so
            // shortcuts never leak literal letters into the composer.
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.col = 0;
                Submit::Nothing
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.col = grapheme_count(&self.lines[self.row]);
                Submit::Nothing
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let col = char_col(&self.lines[self.row], self.col);
                self.lines[self.row].replace_range(..col, "");
                self.col = 0;
                Submit::Nothing
            }
            KeyCode::Char(_) if key.modifiers.contains(KeyModifiers::CONTROL) => Submit::Nothing,
            KeyCode::Char(c) => {
                let line = &mut self.lines[self.row];
                let col = char_col(line, self.col);
                line.insert(col, c);
                self.col += 1;
                Submit::Nothing
            }
            KeyCode::Backspace => {
                self.backspace();
                Submit::Nothing
            }
            KeyCode::Delete => {
                self.delete();
                Submit::Nothing
            }
            KeyCode::Left => {
                if self.col > 0 {
                    self.col -= 1;
                } else if self.row > 0 {
                    self.row -= 1;
                    self.col = grapheme_count(&self.lines[self.row]);
                }
                Submit::Nothing
            }
            KeyCode::Right => {
                let len = grapheme_count(&self.lines[self.row]);
                if self.col < len {
                    self.col += 1;
                } else if self.row + 1 < self.lines.len() {
                    self.row += 1;
                    self.col = 0;
                }
                Submit::Nothing
            }
            KeyCode::Up => {
                if self.row > 0 {
                    self.row -= 1;
                    self.col = self.col.min(grapheme_count(&self.lines[self.row]));
                } else {
                    self.history_prev();
                }
                Submit::Nothing
            }
            KeyCode::Down => {
                if self.row + 1 < self.lines.len() {
                    self.row += 1;
                    self.col = self.col.min(grapheme_count(&self.lines[self.row]));
                } else {
                    self.history_next();
                }
                Submit::Nothing
            }
            KeyCode::Home => {
                self.col = 0;
                Submit::Nothing
            }
            KeyCode::End => {
                self.col = grapheme_count(&self.lines[self.row]);
                Submit::Nothing
            }
            _ => Submit::Nothing,
        }
    }

    fn backspace(&mut self) {
        if self.col > 0 {
            let line = &mut self.lines[self.row];
            let col = char_col(line, self.col);
            // Step back to the start of the previous char; col - 1 is a byte
            // offset that lands inside multi-byte chars.
            let prev = line[..col]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            line.remove(prev);
            self.col -= 1;
        } else if self.row > 0 {
            let rest = self.lines.remove(self.row);
            self.row -= 1;
            self.col = grapheme_count(&self.lines[self.row]);
            self.lines[self.row].push_str(&rest);
        }
    }

    fn delete(&mut self) {
        let len = self.lines[self.row].len();
        let col = char_col(&self.lines[self.row], self.col);
        if col < len {
            self.lines[self.row].remove(col);
        } else if self.row + 1 < self.lines.len() {
            let next = self.lines.remove(self.row + 1);
            self.lines[self.row].push_str(&next);
        }
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.history_idx {
            None => {
                if !self.is_empty() {
                    self.stash = Some(self.lines.clone());
                }
                self.history.len() - 1
            }
            Some(i) if i > 0 => i - 1,
            Some(i) => i,
        };
        self.history_idx = Some(idx);
        self.lines = vec![self.history[idx].clone()];
        self.row = 0;
        self.col = grapheme_count(&self.lines[0]);
    }

    fn history_next(&mut self) {
        let Some(idx) = self.history_idx else { return };
        if idx + 1 < self.history.len() {
            self.history_idx = Some(idx + 1);
            self.lines = vec![self.history[idx + 1].clone()];
        } else {
            self.history_idx = None;
            self.lines = self.stash.take().unwrap_or_else(|| vec![String::new()]);
        }
        self.row = 0;
        self.col = grapheme_count(&self.lines[0]);
    }
}

/// Byte offset of char column `col` in `line`.
fn char_col(line: &str, col: usize) -> usize {
    line.char_indices().nth(col).map(|(i, _)| i).unwrap_or(line.len())
}

fn grapheme_count(s: &str) -> usize {
    s.chars().count()
}

pub struct App {
    pub blocks: Vec<Block>,
    /// Rendered lines per block at `render_width`; flattened each frame.
    pub rendered: Vec<Vec<Line<'static>>>,
    pub render_width: u16,
    /// Absolute top row of the scroll viewport; None = pin to bottom.
    pub scroll_top: Option<u16>,
    /// Visible transcript height, updated every frame; used by scrolling.
    pub viewport: u16,
    pub composer: Composer,
    pub busy: bool,
    /// When the current turn started, for the status row's elapsed clock.
    pub busy_started_at: Option<Instant>,
    pub spinner: usize,
    /// Approvals the backend has asked and the user has not answered; the
    /// modal shows the front one (§4: queue, answer by explicit id).
    pub approvals: VecDeque<Approval>,
    pub model: String,
    pub host: String,
    pub search_status: String,
    pub turns: usize,
    pub quit: bool,
    /// Draw every trace in full; toggled by `ctrl+t`. Off by default so the
    /// reasoning stays out of the way while a turn runs.
    pub show_trace: bool,
}

impl App {
    pub fn new(model: String, host: String, search_status: String) -> Self {
        Self {
            blocks: Vec::new(),
            rendered: Vec::new(),
            render_width: 80,
            scroll_top: None,
            viewport: 24,
            composer: Composer::default(),
            busy: false,
            busy_started_at: None,
            spinner: 0,
            approvals: VecDeque::new(),
            model,
            host,
            search_status,
            turns: 0,
            quit: false,
            show_trace: false,
        }
    }

    /// Expand or collapse every trace (`ctrl+t`).
    pub fn toggle_trace(&mut self) {
        self.show_trace = !self.show_trace;
        for i in 0..self.blocks.len() {
            if matches!(&self.blocks[i], Block::Trace { .. }) {
                self.rerender_block(i);
            }
        }
    }

    pub fn set_width(&mut self, width: u16) {
        let w = width.saturating_sub(4).max(20);
        if w == self.render_width {
            return;
        }
        self.render_width = w;
        for i in 0..self.blocks.len() {
            self.rerender_block(i);
        }
    }

    fn rerender_block(&mut self, index: usize) {
        let lines = match &self.blocks[index] {
            Block::Message { role, text } => render(text, *role, self.render_width),
            Block::Trace { items, live } => self.render_trace(items, *live),
        };
        if self.rendered.len() <= index {
            self.rendered.resize(index + 1, Vec::new());
        }
        self.rendered[index] = lines;
    }

    /// Codex's exec line: the mark carries the only colour, the detail sits
    /// in gray, the summary stays dim.
    fn tool_line(name: &str, detail: &str, done: &Option<(String, bool)>) -> Line<'static> {
        let (mark, mark_style) = match done {
            Some((_, true)) => ("✓", Style::default().fg(theme::SUCCESS)),
            Some((_, false)) => ("✗", Style::default().fg(theme::FAILURE)),
            None => ("✳", Style::default().fg(theme::SECONDARY)),
        };
        let detail_short = backend::truncate(detail.replace('\n', " ").trim(), 72);
        let rest = if detail_short.is_empty() {
            name.to_string()
        } else {
            format!("{name}({detail_short})")
        };
        let suffix = match done {
            Some((summary, _)) => format!(" · {summary}"),
            None => " …".to_string(),
        };
        Line::from(vec![
            Span::styled(format!("{mark} "), mark_style),
            Span::styled(rest, Style::default().fg(theme::SECONDARY)),
            Span::styled(suffix, Style::default().fg(theme::MUTED)),
        ])
    }

    /// A trace renders as one summary line, plus the reasoning and tool lines
    /// when `show_trace` is on. A lone tool call has nothing to summarise, so
    /// it keeps its plain exec line.
    fn render_trace(&self, items: &[TraceItem], live: bool) -> Vec<Line<'static>> {
        if let [TraceItem::Tool { name, detail, done }] = items {
            return vec![Self::tool_line(name, detail, done)];
        }
        if items.is_empty() {
            return Vec::new();
        }

        // `3 steps · thinking×2, web_search` — repeats collapse into a count
        // in first-seen order, so a search-heavy turn still fits on one line.
        let mut names: Vec<(String, usize)> = Vec::new();
        for item in items {
            let name = match item {
                TraceItem::Thinking(_) => "thinking",
                TraceItem::Tool { name, .. } => name.as_str(),
            };
            match names.iter_mut().find(|(seen, _)| seen == name) {
                Some((_, count)) => *count += 1,
                None => names.push((name.to_string(), 1)),
            }
        }
        let joined = names
            .iter()
            .map(|(name, count)| {
                if *count > 1 {
                    format!("{name}×{count}")
                } else {
                    name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        // The transcript clips rather than wraps, so keep the summary short
        // enough that the live tail after it still fits on one line.
        let joined = backend::truncate(
            &joined,
            (self.render_width as usize).saturating_sub(26).max(16),
        );
        let steps = if items.len() == 1 {
            "1 step".to_string()
        } else {
            format!("{} steps", items.len())
        };
        let failed = items
            .iter()
            .any(|item| matches!(item, TraceItem::Tool { done: Some((_, false)), .. }));

        let (mark, mark_style) = if live {
            ("✳", Style::default().fg(theme::SECONDARY))
        } else if self.show_trace {
            ("▾", Style::default().fg(theme::MUTED))
        } else if failed {
            ("✗", Style::default().fg(theme::FAILURE))
        } else {
            ("▸", Style::default().fg(theme::MUTED))
        };
        let mut header = vec![
            Span::styled(format!("{mark} "), mark_style),
            Span::styled(steps, Style::default().fg(theme::SECONDARY)),
            Span::styled(format!(" · {joined}"), Style::default().fg(theme::MUTED)),
        ];
        if !self.show_trace {
            // While collapsed, the newest thought is still worth a glance —
            // dimmed onto the same line rather than streamed line by line.
            if let Some(tail) = items.iter().rev().find_map(|item| match item {
                TraceItem::Thinking(text) => text.lines().rev().find(|l| !l.trim().is_empty()),
                _ => None,
            }) {
                let room = (self.render_width as usize).saturating_sub(48).max(12);
                header.push(Span::styled(
                    format!(" · {}", backend::truncate(tail.trim(), room)),
                    Style::default().fg(theme::MUTED).add_modifier(Modifier::ITALIC),
                ));
            }
        }
        let mut lines = vec![Line::from(header)];
        if !self.show_trace {
            return lines;
        }

        let thinking_style = theme::role_style(Role::Thinking);
        for item in items {
            match item {
                TraceItem::Thinking(text) => {
                    for raw in text.trim_end().lines() {
                        if raw.trim().is_empty() {
                            continue;
                        }
                        lines.extend(crate::markdown::span_line(
                            &format!("  {}", raw.trim_end()),
                            thinking_style,
                            self.render_width as usize,
                        ));
                    }
                }
                TraceItem::Tool { name, detail, done } => {
                    lines.push(Self::tool_line(name, detail, done));
                }
            }
        }
        if live {
            if let Some(last) = lines.last_mut() {
                last.spans
                    .push(Span::styled(" ▍", Style::default().fg(theme::MUTED)));
            }
        }
        lines
    }

    /// Append a fresh block (already rendered) to the transcript.
    fn push_block(&mut self, block: Block) {
        self.blocks.push(block);
        let i = self.blocks.len() - 1;
        self.rerender_block(i);
    }

    /// Stop the live cursor on every open trace. Called when real output
    /// starts, wherever the trace sits in the transcript.
    fn close_live_traces(&mut self) {
        let open: Vec<usize> = (0..self.blocks.len())
            .filter(|i| matches!(&self.blocks[*i], Block::Trace { live: true, .. }))
            .collect();
        for i in open {
            if let Block::Trace { live, .. } = &mut self.blocks[i] {
                *live = false;
            }
            self.rerender_block(i);
        }
    }

    fn last_message_block(&self, role: Role) -> Option<usize> {
        match self.blocks.last() {
            Some(Block::Message { role: r, .. }) if *r == role => Some(self.blocks.len() - 1),
            _ => None,
        }
    }

    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Delta(text) => {
                // Any text delta stops the live thinking cursor — wherever the
                // trace sits, since a tool step may have been pushed after it
                // opened.
                self.close_live_traces();
                let idx = match self.last_message_block(Role::Assistant) {
                    Some(i) => i,
                    None => {
                        self.blocks.push(Block::Message {
                            role: Role::Assistant,
                            text: String::new(),
                        });
                        self.blocks.len() - 1
                    }
                };
                if let Block::Message { text: t, .. } = &mut self.blocks[idx] {
                    t.push_str(&text);
                }
                self.rerender_block(idx);
            }
            AgentEvent::ThinkingDelta(text) => {
                let idx = match self.blocks.last() {
                    Some(Block::Trace { .. }) => self.blocks.len() - 1,
                    _ => {
                        self.push_block(Block::Trace { items: Vec::new(), live: true });
                        self.blocks.len() - 1
                    }
                };
                if let Block::Trace { items, live } = &mut self.blocks[idx] {
                    *live = true;
                    // One thinking item per uninterrupted stream; a tool call
                    // in between starts a new one.
                    match items.last_mut() {
                        Some(TraceItem::Thinking(t)) => t.push_str(&text),
                        _ => items.push(TraceItem::Thinking(text)),
                    }
                }
                self.rerender_block(idx);
            }
            AgentEvent::AssistantDone => {
                self.close_live_traces();
            }
            AgentEvent::ToolStart { name, detail } => {
                let idx = match self.blocks.last() {
                    Some(Block::Trace { .. }) => self.blocks.len() - 1,
                    _ => {
                        self.push_block(Block::Trace { items: Vec::new(), live: false });
                        self.blocks.len() - 1
                    }
                };
                if let Block::Trace { items, .. } = &mut self.blocks[idx] {
                    items.push(TraceItem::Tool { name, detail, done: None });
                }
                self.rerender_block(idx);
            }
            AgentEvent::ToolEnd { summary, ok } => {
                // Close the newest tool still running, walking back through the
                // traces: an answer may already sit on top of it.
                let mut hit = None;
                'outer: for i in (0..self.blocks.len()).rev() {
                    if let Block::Trace { items, .. } = &mut self.blocks[i] {
                        for item in items.iter_mut().rev() {
                            if let TraceItem::Tool { done, .. } = item {
                                if done.is_none() {
                                    *done = Some((summary.clone(), ok));
                                    hit = Some(i);
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                if let Some(i) = hit {
                    self.rerender_block(i);
                }
            }
            AgentEvent::System(markdown) => {
                self.push_block(Block::Message { role: Role::System, text: markdown });
            }
            AgentEvent::Notice(text) => {
                self.push_block(Block::Message {
                    role: Role::System,
                    text: format!("*{text}*"),
                });
            }
            AgentEvent::Error(text) => {
                self.push_block(Block::Message {
                    role: Role::Error,
                    text: format!("**error** — {text}"),
                });
            }
            AgentEvent::ApprovalRequest { id, tool, preview, reason } => {
                self.approvals.push_back(Approval { id, tool, preview, reason });
            }
            AgentEvent::Busy(on) => {
                self.busy = on;
                // The clock restarts with every turn; it stops when the turn ends.
                self.busy_started_at = if on { Some(Instant::now()) } else { None };
            }
            AgentEvent::ConversationCleared => {
                self.blocks.clear();
                self.rendered.clear();
                self.scroll_top = None;
                self.turns = 0;
                self.approvals.clear();
            }
            AgentEvent::Ready { model, host, search } => {
                self.model = model;
                self.host = host;
                self.search_status = search;
            }
            AgentEvent::TurnDone => {
                self.turns = self
                    .blocks
                    .iter()
                    .filter(|b| matches!(b, Block::Message { role: Role::User, .. }))
                    .count();
            }
        }
    }

    pub fn push_user(&mut self, text: &str) {
        self.push_block(Block::Message { role: Role::User, text: text.to_string() });
        self.scroll_top = None;
    }

    pub fn total_lines(&self) -> usize {
        self.rendered.iter().map(|b| b.len()).sum()
    }

    /// Take the question the modal is showing; the caller answers it by id.
    pub fn pop_approval(&mut self) -> Option<Approval> {
        self.approvals.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    fn key(code: crossterm::event::KeyCode, mods: crossterm::event::KeyModifiers) -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::new(code, mods)
    }

    #[test]
    fn ctrl_j_and_shift_enter_insert_newlines() {
        let mut c = Composer::default();
        for ch in "one".chars() {
            c.handle_key(key(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        c.handle_key(key(KeyCode::Char('j'), KeyModifiers::CONTROL));
        c.handle_key(key(KeyCode::Enter, KeyModifiers::SHIFT));
        for ch in "two".chars() {
            c.handle_key(key(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        assert_eq!(c.lines, vec!["one".to_string(), String::new(), "two".to_string()]);
        assert_eq!(c.row, 2);
    }

    #[test]
    fn plain_enter_submits_multiline() {
        let mut c = Composer::default();
        for ch in "a\nb".chars() {
            match ch {
                '\n' => c.newline(),
                x => {
                    c.handle_key(key(KeyCode::Char(x), KeyModifiers::NONE));
                }
            }
        }
        match c.handle_key(key(KeyCode::Enter, KeyModifiers::NONE)) {
            Submit::Send(text) => assert_eq!(text, "a\nb"),
            _ => panic!("expected submit"),
        }
        assert!(c.is_empty());
    }

    #[test]
    fn paste_splits_lines_and_keeps_tail() {
        let mut c = Composer::default();
        for ch in "hello".chars() {
            c.handle_key(key(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        c.handle_key(key(KeyCode::Left, KeyModifiers::NONE));
        c.handle_key(key(KeyCode::Left, KeyModifiers::NONE)); // cursor between 'l' and 'o'
        c.insert_str("X\nY\n");
        assert_eq!(c.lines, vec!["helX".to_string(), "Y".to_string(), "lo".to_string()]);
        assert_eq!((c.row, c.col), (2, 0));
    }

    #[test]
    fn backspace_joins_lines_upwards() {
        let mut c = Composer::default();
        for ch in "ab".chars() {
            c.handle_key(key(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        c.newline();
        c.backspace();
        assert_eq!(c.lines, vec!["ab".to_string()]);
    }

    #[test]
    fn history_recall_then_release() {
        let mut c = Composer::default();
        for ch in "first".chars() {
            c.handle_key(key(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        let _ = c.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        c.handle_key(key(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.text(), "first");
        c.handle_key(key(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.text(), "");
    }

    #[test]
    fn backspace_handles_multibyte() {
        let mut c = Composer::default();
        for ch in "héllo".chars() {
            c.handle_key(key(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        c.handle_key(key(KeyCode::Backspace, KeyModifiers::NONE));
        c.handle_key(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(c.text(), "hél");
        c.handle_key(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(c.text(), "hé");
    }

    #[test]
    fn control_keys_edit_instead_of_inserting() {
        let mut c = Composer::default();
        for ch in "abc".chars() {
            c.handle_key(key(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        c.handle_key(key(KeyCode::Char('a'), KeyModifiers::CONTROL));
        assert_eq!((c.row, c.col), (0, 0));
        c.handle_key(key(KeyCode::Char('e'), KeyModifiers::CONTROL));
        assert_eq!(c.col, 3);
        c.handle_key(key(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert_eq!(c.text(), "");
        // Unknown Ctrl combos must not leak letters into the composer.
        c.handle_key(key(KeyCode::Char('x'), KeyModifiers::CONTROL));
        assert_eq!(c.text(), "");
    }

    fn text_of(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A thinking chunk, a search, more thinking, then the answer: one turn
    /// with three trace steps folds into a single line.
    fn run_a_turn(app: &mut App) {
        app.handle_agent_event(AgentEvent::ThinkingDelta("look up the docs".into()));
        app.handle_agent_event(AgentEvent::ToolStart {
            name: "web_search".into(),
            detail: "ratatui block".into(),
        });
        app.handle_agent_event(AgentEvent::ToolEnd {
            summary: "5 results".into(),
            ok: true,
        });
        app.handle_agent_event(AgentEvent::ThinkingDelta("now answer".into()));
        app.handle_agent_event(AgentEvent::AssistantDone);
        app.handle_agent_event(AgentEvent::Delta("here you go".into()));
    }

    #[test]
    fn thinking_and_tools_fold_into_one_collapsed_line() {
        let mut app = App::new(String::new(), String::new(), String::new());
        run_a_turn(&mut app);

        assert_eq!(app.blocks.len(), 2, "trace + answer: {:?}", app.blocks);
        let trace = text_of(&app.rendered[0]);
        assert_eq!(app.rendered[0].len(), 1, "collapsed trace is one line: {trace}");
        assert!(trace.starts_with("▸ 3 steps · thinking×2, web_search"), "{trace}");
        assert!(!trace.contains("look up the docs"), "reasoning hidden: {trace}");

        // The live cursor closed when the answer started streaming.
        assert!(matches!(&app.blocks[0], Block::Trace { live: false, .. }));
    }

    #[test]
    fn ctrl_t_expands_the_trace_in_place() {
        let mut app = App::new(String::new(), String::new(), String::new());
        run_a_turn(&mut app);
        app.toggle_trace();

        let trace = text_of(&app.rendered[0]);
        assert!(trace.starts_with("▾ 3 steps · thinking×2, web_search"), "{trace}");
        assert!(trace.contains("look up the docs"), "reasoning shown: {trace}");
        assert!(trace.contains("✓ web_search(ratatui block) · 5 results"), "{trace}");

        app.toggle_trace();
        assert_eq!(app.rendered[0].len(), 1, "collapses back to one line");
    }

    #[test]
    fn a_single_tool_call_keeps_its_own_exec_line() {
        let mut app = App::new(String::new(), String::new(), String::new());
        app.handle_agent_event(AgentEvent::ToolStart {
            name: "shell".into(),
            detail: "cargo test".into(),
        });
        app.handle_agent_event(AgentEvent::ToolEnd {
            summary: "14 passed".into(),
            ok: true,
        });

        let trace = text_of(&app.rendered[0]);
        assert_eq!(trace, "✓ shell(cargo test) · 14 passed");
    }

    #[test]
    fn a_new_message_starts_a_fresh_trace() {
        let mut app = App::new(String::new(), String::new(), String::new());
        run_a_turn(&mut app);
        app.push_user("and again");
        app.handle_agent_event(AgentEvent::ThinkingDelta("second look".into()));
        app.handle_agent_event(AgentEvent::AssistantDone);

        let traces = app
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Trace { .. }))
            .count();
        assert_eq!(traces, 2, "one trace per run of activity: {:?}", app.blocks);
    }

    #[test]
    fn busy_tracks_when_the_turn_started() {
        let mut app = App::new(String::new(), String::new(), String::new());
        assert!(app.busy_started_at.is_none());
        app.handle_agent_event(AgentEvent::Busy(true));
        assert!(app.busy);
        assert!(app.busy_started_at.is_some());
        app.handle_agent_event(AgentEvent::Busy(false));
        assert!(!app.busy);
        assert!(app.busy_started_at.is_none());
    }
}
