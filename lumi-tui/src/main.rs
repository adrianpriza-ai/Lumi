//! lumi-tui — a Codex-style terminal UI for Lumi.
//!
//! A frontend, not a second agent: the whole brain — model, tools, safety,
//! memory — lives in Python (`lumi/tui.py`), served over JSON lines by a
//! `lumi tui-serve` child process this binary spawns and owns. Rust keeps the
//! terminal: raw mode, the composer, scrollback, markdown, the approval modal.
//!
//! Users do not run this binary: `lumi` with no arguments execs it (see
//! `cmd_ui` in `lumi/__main__.py`), the way `vibe` runs `vibe-rs`.

mod app;
mod backend;
mod markdown;
mod ui;

use anyhow::{anyhow, Result};
use app::{App, Submit};
use backend::{AgentEvent, AgentMsg};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::stdout;

pub fn help_text() -> String {
    "\
## Commands
- `/help` — this list
- `/search <query>` — run a web search now, through the backend
- `/reasoning on|off` — show the model's thinking steps live
- `/tools` — the tools the backend has loaded
- `/context` — the context window: tokens in use, compactions
- `/memory` — this chat's long-term memory as it enters the prompt
- `/reset` — clear the conversation (memory file stays)
- `/quit` — leave

## Keys
- `enter` send · `shift+enter` (or `ctrl+j`, `alt+enter`) new line
- `↑/↓` input history and multi-line cursor · `pgup/pgdn`, wheel scrollback
- `y`/`n` answer the approval modal · `esc` cancel the running turn · `ctrl+c` cancel, then quit on empty input"
        .to_string()
}

fn main() -> Result<()> {
    // A panic in raw mode with the alternate screen up would leave the shell
    // unusable; restore the terminal before the default handler prints.
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(
            stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            DisableBracketedPaste
        );
        default_panic(info);
    }));

    // Multi-threaded: the backend task must keep reading its child while this
    // thread runs the blocking terminal loop below.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let (msg_tx, mut event_rx) = runtime.block_on(async { backend::spawn() });

    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture, EnableBracketedPaste)?;
    // Enhanced keys make Shift+Enter a distinct event where the terminal
    // supports the Kitty keyboard protocol; elsewhere Ctrl+J still works.
    let _ = crossterm::execute!(
        out,
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        )
    );

    let backend_ui = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend_ui)?;

    let result = run_ui(&mut terminal, &msg_tx, &mut event_rx);

    let _ = crossterm::execute!(
        terminal.backend_mut(),
        crossterm::event::PopKeyboardEnhancementFlags
    );
    let _ = execute!(terminal.backend_mut(), DisableBracketedPaste, DisableMouseCapture);
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);

    result
}

fn run_ui(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    msg_tx: &tokio::sync::mpsc::UnboundedSender<AgentMsg>,
    event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEvent>,
) -> Result<()> {
    // The header starts blank; the backend's `ready` event fills it in the
    // moment the child is up. No local config is parsed — Python owns it.
    let mut app = App::new(String::new(), String::new(), String::new());

    let mut needs_draw = true;
    loop {
        if needs_draw {
            terminal.draw(|frame| ui::draw(frame, &mut app))?;
            needs_draw = false;
        }

        // Drain everything the backend produced, then wait for input briefly
        // so the spinner keeps ticking while a turn runs.
        let mut got_events = false;
        loop {
            match event_rx.try_recv() {
                Ok(event) => {
                    app.handle_agent_event(event);
                    got_events = true;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    // The backend task ended: the child died and its fate is
                    // already on the transcript. Leave via a terminal-safe
                    // path so main() restores the screen before reporting.
                    return Err(anyhow!(
                        "the lumi backend went away — see data/logs/lumi-tui-backend.log"
                    ));
                }
            }
        }
        if got_events {
            needs_draw = true;
        }
        let timeout = if app.busy || got_events {
            std::time::Duration::from_millis(80)
        } else {
            std::time::Duration::from_millis(250)
        };

        if crossterm::event::poll(timeout)? {
            match crossterm::event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if handle_key(&mut app, msg_tx, key) {
                        return Ok(());
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => scroll(&mut app, -8),
                    MouseEventKind::ScrollDown => scroll(&mut app, 8),
                    _ => {}
                },
                Event::Paste(text) if app.approvals.is_empty() => {
                    app.composer.insert_str(&text)
                }
                Event::Resize(w, _h) => app.set_width(w),
                _ => {}
            }
            needs_draw = true;
        }
        if app.busy {
            app.spinner = app.spinner.wrapping_add(1);
            needs_draw = true;
        }
        if app.quit {
            return Ok(());
        }
    }
}

/// Adjust the scroll viewport. `delta` lines, negative = towards history.
fn scroll(app: &mut App, delta: i32) {
    let total = app.total_lines() as i32;
    let viewport = app.viewport as i32;
    let max_top = (total - viewport).max(0);
    let current = app.scroll_top.unwrap_or(max_top as u16) as i32;
    let next = (current + delta).clamp(0, max_top);
    app.scroll_top = if next >= max_top { None } else { Some(next as u16) };
}

/// Returns true when the UI should close.
fn handle_key(
    app: &mut App,
    msg_tx: &tokio::sync::mpsc::UnboundedSender<AgentMsg>,
    key: crossterm::event::KeyEvent,
) -> bool {
    use crossterm::event::KeyCode;

    // The approval modal swallows everything while it is up. Each answer
    // goes to the backend by its id; the next queued one takes its place.
    if !app.approvals.is_empty() {
        let approved = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => Some(true),
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc | KeyCode::Enter => Some(false),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Some(false),
            _ => None,
        };
        if let Some(approved) = approved {
            if let Some(answer) = app.pop_approval() {
                let _ = msg_tx.send(AgentMsg::Approval { id: answer.id, approved });
            }
        }
        return false;
    }

    // Ctrl+C: cancel the turn, clear the composer, then quit.
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if app.busy {
            let _ = msg_tx.send(AgentMsg::Cancel);
            return false;
        }
        if !app.composer.is_empty() {
            app.composer.clear();
            return false;
        }
        return true;
    }
    // Ctrl+D quits on an empty composer.
    if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if app.composer.is_empty() {
            return true;
        }
        return false;
    }

    match key.code {
        KeyCode::Esc if app.busy => {
            let _ = msg_tx.send(AgentMsg::Cancel);
            return false;
        }
        KeyCode::Esc => {
            app.composer.clear();
            return false;
        }
        KeyCode::PageUp => {
            scroll(app, -20);
            return false;
        }
        KeyCode::PageDown => {
            scroll(app, 20);
            return false;
        }
        KeyCode::Up if key.modifiers.contains(KeyModifiers::ALT) => {
            scroll(app, -2);
            return false;
        }
        KeyCode::Down if key.modifiers.contains(KeyModifiers::ALT) => {
            scroll(app, 2);
            return false;
        }
        _ => {}
    }

    match app.composer.handle_key(key) {
        Submit::Send(text) => {
            let lowered = text.trim().to_ascii_lowercase();
            if lowered == "/quit" || lowered == "/exit" || lowered == "/q" || lowered == ":q" {
                return true;
            }
            app.push_user(&text);
            app.scroll_top = None;
            let _ = msg_tx.send(AgentMsg::Submit(text));
        }
        Submit::Nothing => {}
    }
    false
}
