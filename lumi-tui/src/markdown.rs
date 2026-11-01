//! A small markdown → styled-line renderer: fences, headings, bullets,
//! quotes, inline code and bold. No syntax highlighting dependency — a chat
//! transcript needs shape, not a compiler.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
    System,
    Thinking,
    Error,
}

fn theme(role: Role) -> (Color, Style) {
    match role {
        Role::User => (Color::Cyan, Style::default().fg(Color::Cyan)),
        Role::Assistant => (Color::Green, Style::default().fg(Color::Green)),
        Role::System => (Color::Blue, Style::default().fg(Color::Blue)),
        Role::Thinking => (Color::DarkGray, Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
        Role::Error => (Color::Red, Style::default().fg(Color::Red)),
    }
}

/// Render a markdown-ish chunk plus its speaker label into full-width lines.
pub fn render(text: &str, role: Role, width: u16) -> Vec<Line<'static>> {
    let w = width.max(20) as usize;
    let (accent, base) = theme(role);
    let mut out: Vec<Line<'static>> = Vec::new();

    let label = match role {
        Role::User => "you",
        Role::Assistant => "lumi",
        Role::System => "lumi ·",
        Role::Thinking => "thinking",
        Role::Error => "error",
    };
    out.push(Line::from(vec![
        Span::styled("● ".to_string(), Style::default().fg(accent)),
        Span::styled(
            label.to_string(),
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ),
    ]));

    if text.trim().is_empty() {
        return out;
    }

    let mut in_code = false;
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            // Show the fence as a subtle divider; the language tag is noise
            // for a reader but keep it visible in dim.
            let tag = line.trim_start().trim_start_matches("```");
            if !tag.is_empty() {
                out.extend(span_line(
                    &format!("  ┆ {tag}"),
                    Style::default().fg(Color::DarkGray),
                    w,
                ));
            }
            continue;
        }
        if in_code {
            out.extend(span_line(
                &format!("  │ {line}"),
                Style::default().fg(Color::Gray),
                w,
            ));
            continue;
        }

        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('#') {
            let level = rest.chars().take_while(|c| *c == '#').count().min(3);
            let body = rest.trim_start_matches('#').trim();
            let style = Style::default()
                .fg(accent)
                .add_modifier(Modifier::BOLD)
                .add_modifier(if level == 1 { Modifier::UNDERLINED } else { Modifier::empty() });
            out.extend(span_line(body, style, w));
            continue;
        }
        if trimmed.starts_with("> ") {
            out.extend(span_line(
                &format!("▏ {}", trimmed[2..].trim()),
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                w,
            ));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- ").or(trimmed.strip_prefix("* ")) {
            out.extend(span_line(
                &format!("  • {rest}"),
                base,
                w.saturating_sub(3),
            ));
            continue;
        }
        if trimmed.is_empty() {
            out.push(Line::from(""));
            continue;
        }
        out.extend(span_line(line, base, w));
    }
    out.push(Line::from(""));
    out
}

/// Wrap one source line into as many display lines as needed, honouring
/// inline `code` and **bold** markers, and unicode display widths.
pub fn span_line(text: &str, base: Style, width: usize) -> Vec<Line<'static>> {
    let styled = inline_styles(text, base);
    wrap_spans(styled, width)
}

/// Split `text` into spans with inline markers applied.
fn inline_styles(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut plain = String::new();
    let mut chars = text.chars().peekable();
    let code_style = Style::default().fg(Color::Cyan);
    let bold_style = base.add_modifier(Modifier::BOLD);

    let push_plain = |spans: &mut Vec<Span<'static>>, buf: &mut String| {
        if !buf.is_empty() {
            spans.push(Span::styled(buf.clone(), base));
            buf.clear();
        }
    };

    while let Some(c) = chars.next() {
        if c == '`' {
            // inline code: consume to the next backtick
            let mut code = String::new();
            let mut closed = false;
            for c2 in chars.by_ref() {
                if c2 == '`' {
                    closed = true;
                    break;
                }
                code.push(c2);
            }
            push_plain(&mut spans, &mut plain);
            if closed {
                spans.push(Span::styled(code, code_style));
            } else {
                plain.push('`');
                plain.push_str(&code);
            }
        } else if c == '*' && chars.peek() == Some(&'*') {
            chars.next();
            // bold: consume to the next **
            let mut bold = String::new();
            let mut closed = false;
            let mut prev_star = false;
            for c2 in chars.by_ref() {
                if c2 == '*' {
                    if prev_star {
                        closed = true;
                        break;
                    }
                    prev_star = true;
                    continue;
                }
                if prev_star {
                    bold.push('*');
                    prev_star = false;
                }
                bold.push(c2);
            }
            push_plain(&mut spans, &mut plain);
            if closed {
                spans.push(Span::styled(bold, bold_style));
            } else {
                plain.push_str("**");
                plain.push_str(&bold);
            }
        } else {
            plain.push(c);
        }
    }
    push_plain(&mut spans, &mut plain);
    spans
}

/// Greedy word-wrap over styled spans. Spans keep their styles across breaks.
pub fn wrap_spans(spans: Vec<Span<'static>>, width: usize) -> Vec<Line<'static>> {
    if spans.is_empty() {
        return vec![Line::from("")];
    }
    let mut lines: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    let mut current_width = 0usize;

    for span in spans {
        let style = span.style;
        let flush_word = |lines: &mut Vec<Vec<Span<'static>>>,
                              current_width: &mut usize,
                              word: &mut String,
                              word_width: &mut usize| {
            if word.is_empty() {
                return;
            }
            // Break before the word when it does not fit.
            if *current_width + *word_width > width && *current_width > 0 {
                if *word_width <= width {
                    lines.push(Vec::new());
                    *current_width = 0;
                }
            }
            // Hard-split a single word longer than a full line.
            while *word_width > width.saturating_sub(*current_width) {
                let room = width.saturating_sub(*current_width).max(1);
                let head: String = word.chars().take(room).collect();
                let tail: String = word.chars().skip(room).collect();
                lines.last_mut().unwrap().push(Span::styled(head, style));
                lines.push(Vec::new());
                *current_width = 0;
                *word = tail;
                *word_width = word.chars().count();
            }
            lines.last_mut().unwrap().push(Span::styled(word.clone(), style));
            *current_width += *word_width;
            word.clear();
            *word_width = 0;
        };

        let mut word = String::new();
        let mut word_width = 0usize;
        for c in span.content.chars() {
            let cw = unicode_width::UnicodeWidthChar::width(c).unwrap_or(1);
            if c == ' ' {
                flush_word(&mut lines, &mut current_width, &mut word, &mut word_width);
                if current_width + 1 <= width {
                    lines.last_mut().unwrap().push(Span::styled(" ".to_string(), style));
                    current_width += 1;
                } else {
                    lines.push(Vec::new());
                    current_width = 0;
                }
            } else {
                word.push(c);
                word_width += cw;
            }
        }
        flush_word(&mut lines, &mut current_width, &mut word, &mut word_width);
    }

    lines.into_iter().map(Line::from).collect()
}
