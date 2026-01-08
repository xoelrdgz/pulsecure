//! UI module: View components for the TUI.

pub mod analytics;
pub mod dashboard;
pub mod inference;
pub mod patient;

use ratatui::{
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::tui::styles::MedicalTheme;

pub fn render_disclaimer(f: &mut Frame, area: Rect) {
    let text = vec![
		Line::from(vec![Span::styled(
			"DISCLAIMER: This tool provides indicative estimates and does not replace professional medical evaluation.",
			MedicalTheme::text_muted(),
		)]),
		Line::from(vec![Span::styled(
			"The model tends toward false positives.",
			MedicalTheme::text_muted(),
		)]),
	];

    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(MedicalTheme::border());

    let p = Paragraph::new(text).block(block).wrap(Wrap { trim: true });

    f.render_widget(p, area);
}
