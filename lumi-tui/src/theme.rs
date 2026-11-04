//! The palette: a Codex-CLI-style near-monochrome — white content, gray
//! chrome, dark background — with green/red/yellow reserved for what they
//! *mean* (success, failure, attention), the way Codex's style.rs does it.
//! Every colour the UI uses lives here, so the look is one file, not thirty
//! scattered `Style` sites.

use crate::markdown::Role;
use ratatui::style::{Color, Modifier, Style};

/// Primary content: assistant prose, the header wordmark, the active composer.
pub const TEXT: Color = Color::White;
/// Secondary text: labels, code, command lines, the working indicator.
pub const SECONDARY: Color = Color::Gray;
/// Chrome: hints, borders, thinking text, dim summaries.
pub const MUTED: Color = Color::DarkGray;
/// Background wash behind the user's own turn — the subtle fill Codex puts
/// behind echoed prompts, so your words read as a block, not as prose.
pub const USER_FILL: Color = Color::DarkGray;

/// Semantic status colours, bold where they are used.
pub const SUCCESS: Color = Color::Green;
pub const FAILURE: Color = Color::Red;
pub const ATTENTION: Color = Color::Yellow;

/// Base style for a transcript block's role.
pub fn role_style(role: Role) -> Style {
    match role {
        Role::User => Style::default().fg(TEXT),
        Role::Assistant => Style::default().fg(TEXT),
        Role::System => Style::default().fg(SECONDARY),
        Role::Thinking => Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
        Role::Error => Style::default().fg(FAILURE),
    }
}

/// The glyph and speaker label shown at the top of each transcript block.
pub fn role_marker(role: Role) -> (&'static str, &'static str) {
    match role {
        Role::User => ("●", "you"),
        Role::Assistant => ("●", "lumi"),
        Role::System => ("●", "lumi ·"),
        Role::Thinking => ("●", "thinking"),
        Role::Error => ("●", "error"),
    }
}

/// Codex-style asterisk spinner frames.
pub const SPINNER: [&str; 4] = ["✳", "✲", "✶", "✱"];

/// Codex's compact elapsed format: `0s`, `59s`, `1m 05s`, `1h 00m 00s` —
/// seconds under a minute, zero-padded seconds under an hour, then zero-padded
/// minutes and seconds (ported from `status_indicator_widget.rs`).
pub fn fmt_elapsed_compact(seconds: u64) -> String {
    if seconds < 60 {
        return format!("{seconds}s");
    }
    let minutes = seconds / 60;
    let secs = seconds % 60;
    if minutes < 60 {
        return format!("{minutes}m {secs:02}s");
    }
    let hours = minutes / 60;
    let minutes = minutes % 60;
    format!("{hours}h {minutes:02}m {secs:02}s")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_formats_like_codex() {
        assert_eq!(fmt_elapsed_compact(0), "0s");
        assert_eq!(fmt_elapsed_compact(59), "59s");
        assert_eq!(fmt_elapsed_compact(60), "1m 00s");
        assert_eq!(fmt_elapsed_compact(65), "1m 05s");
        assert_eq!(fmt_elapsed_compact(3599), "59m 59s");
        assert_eq!(fmt_elapsed_compact(3600), "1h 00m 00s");
        assert_eq!(fmt_elapsed_compact(9012), "2h 30m 12s");
    }

    #[test]
    fn spinner_frames_are_single_column_glyphs() {
        for frame in SPINNER {
            assert_eq!(frame.chars().count(), 1, "{frame} is not one char");
        }
    }
}
