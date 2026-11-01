//! lumi-tui — a Codex-style terminal UI for Lumi.
//!
//! Talks to any OpenAI-compatible endpoint with SSE streaming, runs tools
//! (web search via Tavily/Firecrawl, page fetch, approved shell, project file
//! reads), and reuses Lumi's `.env` / `config.toml` / `PERSONALITY.md`.

mod agent;
mod app;
mod config;
mod llm;
mod markdown;
mod search;
mod ui;

use agent::{AgentEvent, AgentMsg};
use anyhow::Result;
use app::{App, Submit};
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
- `/search <query>` — run a web search now (Tavily, fallback Firecrawl)
- `/model [name]` — show or switch the model
- `/reasoning on|off` — show the model's thinking live
- `/tools` — list the tools and which search provider is live
- `/reset` — clear the conversation (memory file stays)
- `/quit` — leave

## Keys
- `enter` send · `shift+enter` (or `ctrl+j`, `alt+enter`) new line
- `↑/↓` input history and multi-line cursor · `pgup/pgdn`, wheel scrollback
- `esc` cancel the running turn · `ctrl+c` cancel, then quit on empty input"
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

    let config = config::Config::load()?;
    // Multi-threaded: the agent task must keep streaming while this thread
    // runs the blocking terminal loop below.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let (msg_tx, mut event_rx) = runtime.block_on(async { agent::Agent::spawn(config.clone()) });

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

    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;

    let result = run_ui(&mut terminal, &msg_tx, &mut event_rx, &config);

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
    config: &config::Config,
) -> Result<()> {
    let mut app = App::new(
        config.model.clone(),
        config.host(),
        config.search_status(),
    );

    let mut needs_draw = true;
    loop {
        if needs_draw {
            terminal.draw(|frame| ui::draw(frame, &mut app))?;
            needs_draw = false;
        }

        // Drain everything the agent produced, then wait for input briefly
        // so the spinner keeps ticking while a turn runs.
        let mut got_events = false;
        while let Ok(event) = event_rx.try_recv() {
            app.handle_agent_event(event);
            got_events = true;
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
                Event::Paste(text) if app.approval.is_none() => {
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

    // The approval modal swallows everything while it is up.
    if app.approval.is_some() {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => app.answer_approval(true),
            // Decline and kill the turn; Busy(false) also clears the modal.
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.answer_approval(false);
                let _ = msg_tx.send(AgentMsg::Cancel);
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc | KeyCode::Enter => {
                app.answer_approval(false)
            }
            _ => {}
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
