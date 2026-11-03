//! Frame drawing: header, scrollable transcript, status line, composer box,
//! and the shell-approval overlay. Codex-CLI-flavoured: rounded boxes, small
//! glyph labels, everything else quiet.

use crate::app::App;
use crate::backend;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use unicode_width::UnicodeWidthChar;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block as TuiBlock, BorderType, Borders, Clear, Paragraph};
use ratatui::Frame;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    app.set_width(area.width);

    let composer_height = composer_height(app);
    let status_height = 1usize;
    let header_height = 1usize;

    let rows = Layout::vertical([
        Constraint::Length(header_height as u16),
        Constraint::Min(3),
        Constraint::Length(status_height as u16),
        Constraint::Length(composer_height),
    ])
    .split(area);

    draw_header(frame, app, rows[0]);
    draw_transcript(frame, app, rows[1]);
    draw_status(frame, app, rows[2]);
    draw_composer(frame, app, rows[3]);
    if let Some(approval) = app.approvals.front() {
        draw_approval(frame, approval, area, app.approvals.len());
    }
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let left = Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(Color::Cyan)),
        Span::styled(
            "lumi",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  chat", Style::default().fg(Color::DarkGray)),
    ]);
    let right = Line::from(vec![
        Span::styled(app.model.clone(), Style::default().fg(Color::DarkGray)),
        Span::styled(" · ", Style::default().fg(Color::DarkGray)),
        Span::styled(app.host.clone(), Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
    ]);
    frame.render_widget(Paragraph::new(left), area);
    frame.render_widget(
        Paragraph::new(right).alignment(Alignment::Right),
        area,
    );
}

fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    app.viewport = area.height;
    let total = app.total_lines() as u16;
    let height = area.height;
    let max_top = total.saturating_sub(height);

    // Resolve the scroll position: None means glued to the newest line.
    let top = match app.scroll_top {
        None => max_top,
        Some(top) => top.min(max_top),
    };
    app.scroll_top = if top >= max_top { None } else { Some(top) };

    let mut lines: Vec<Line> = Vec::with_capacity(total as usize);
    for block_lines in &app.rendered {
        lines.extend(block_lines.iter().cloned());
    }

    if lines.is_empty() {
        let mut empty = Vec::new();
        empty.push(Line::from(""));
        empty.push(Line::from(Span::styled(
            "  lumi chat",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        empty.push(Line::from(Span::styled(
            "  type a message, or /help for commands",
            Style::default().fg(Color::DarkGray),
        )));
        empty.push(Line::from(Span::styled(
            "  enter sends · shift+enter (or ctrl+j) adds a line",
            Style::default().fg(Color::DarkGray),
        )));
        frame.render_widget(Paragraph::new(empty), area);
        return;
    }

    // Lines are pre-wrapped in the renderer, so no Paragraph wrap here —
    // anything longer than the pane just clips instead of double-wrapping.
    let paragraph = Paragraph::new(lines).scroll((top, 0));
    frame.render_widget(paragraph, area);
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let left = if let Some(head) = app.approvals.front() {
        let more = if app.approvals.len() > 1 {
            format!(" · {} more waiting", app.approvals.len() - 1)
        } else {
            String::new()
        };
        let label = if head.tool.is_empty() {
            "approval needed".to_string()
        } else {
            format!("{} asks", head.tool)
        };
        Line::from(Span::styled(
            format!(" ⚠ {label} — y to run, n to decline{more}"),
            Style::default().fg(Color::Yellow),
        ))
    } else if app.busy {
        let glyph = SPINNER[app.spinner % SPINNER.len()];
        Line::from(vec![
            Span::styled(format!(" {glyph} "), Style::default().fg(Color::Cyan)),
            Span::styled("working — esc to cancel", Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(Span::styled(
            " enter send · shift+enter newline · ↑ history · /help commands",
            Style::default().fg(Color::DarkGray),
        ))
    };

    let right = Line::from(vec![
        Span::styled(
            format!("search: {}", app.search_status),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(" · ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{} turns", app.turns), Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
    ]);

    frame.render_widget(Paragraph::new(left), area);
    frame.render_widget(Paragraph::new(right).alignment(Alignment::Right), area);
}

fn composer_height(app: &App) -> u16 {
    // Border(2) + content lines, capped so a wall of pasted text stays sane.
    let needed = app.composer.lines.len() as u16;
    (needed + 2).clamp(3, 12)
}

fn draw_composer(frame: &mut Frame, app: &App, area: Rect) {
    let accent = if app.busy { Color::DarkGray } else { Color::Cyan };
    let label = if app.composer.is_empty() {
        "you — type a message"
    } else {
        "you"
    };
    let block = TuiBlock::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .title(Span::styled(
            format!(" {label} "),
            Style::default().fg(accent),
        ));

    let visible = area.height.saturating_sub(2) as usize;
    // Stateless scroll: show a window ending at the cursor row once the text
    // outgrows the box, so the caret never falls off the bottom.
    let offset = app
        .composer
        .row
        .saturating_sub(visible.saturating_sub(1))
        .min(app.composer.lines.len().saturating_sub(visible.min(app.composer.lines.len())));
    let mut lines: Vec<Line> = Vec::new();
    for (i, text) in app.composer.lines.iter().skip(offset).take(visible).enumerate() {
        if i + offset == app.composer.row {
            // Show the caret as a reverse-space; the real cursor is placed too.
            let (head, caret, tail) = split_at_col(text, app.composer.col);
            lines.push(Line::from(vec![
                Span::raw(format!(" {head}")),
                Span::styled(caret, Style::default().add_modifier(Modifier::REVERSED)),
                Span::raw(tail),
            ]));
        } else {
            lines.push(Line::from(Span::raw(format!(" {text}"))));
        }
    }
    if lines.is_empty() {
        lines.push(Line::from(" "));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);

    // Terminal cursor at the caret position: border (1) + leading pad (1) +
    // display width of everything before it, adjusted for the scroll window.
    let row_text = app
        .composer
        .lines
        .get(app.composer.row)
        .map(String::as_str)
        .unwrap_or("");
    let head_w: usize = row_text
        .chars()
        .take(app.composer.col)
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(1))
        .sum();
    let inner_x = area.x + 2 + head_w as u16;
    let display_col = inner_x.min(area.x + area.width.saturating_sub(2));
    let cursor_y = area.y + 1 + (app.composer.row - offset) as u16;
    frame.set_cursor_position((display_col, cursor_y.min(area.y + area.height.saturating_sub(2))));
}

fn split_at_col(text: &str, col: usize) -> (String, String, String) {
    let mut head = String::new();
    let mut caret = " ".to_string();
    let mut tail = String::new();
    for (i, c) in text.chars().enumerate() {
        if i < col {
            head.push(c);
        } else if i == col {
            caret = c.to_string();
        } else {
            tail.push(c);
        }
    }
    (head, caret, tail)
}

fn draw_approval(frame: &mut Frame, approval: &crate::app::Approval, area: Rect, waiting: usize) {
    let width = (area.width.saturating_sub(8)).min(72).max(24) as usize;
    let command_lines = backend::truncate(&approval.preview, 600);

    let block = TuiBlock::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            format!(
                " ⚠ run {}?{} ",
                if approval.tool.is_empty() { "this" } else { &approval.tool },
                if waiting > 1 { format!(" ({waiting} waiting)") } else { String::new() }
            ),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ));

    let mut lines: Vec<Line> = Vec::new();
    if !approval.reason.is_empty() {
        for raw in approval.reason.lines() {
            lines.extend(crate::markdown::span_line(
                &format!("  {raw}"),
                Style::default().fg(Color::Yellow),
                width.saturating_sub(2),
            ));
        }
    }
    for raw in command_lines.lines() {
        lines.extend(crate::markdown::span_line(
            &format!("  $ {raw}"),
            Style::default().fg(Color::Gray),
            width.saturating_sub(2),
        ));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  [y]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::raw(" approve   "),
        Span::styled("[n]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::raw(" decline"),
    ]));

    let height = (lines.len() + 2) as u16;
    let box_area = centered_rect(area, width as u16, height);
    frame.render_widget(Clear, box_area);
    frame.render_widget(Paragraph::new(lines).block(block), box_area);
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    }
}
