//! Analytics view: Privacy-preserving statistics.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame,
};

use crate::ports::PrivateStatistics;
use crate::tui::styles::MedicalTheme;

/// Analytics state
pub struct AnalyticsState {
    pub statistics: Option<PrivateStatistics>,
    pub epsilon_used: f64,
    pub budget_remaining: f64,
    pub loading: bool,
    pub error: Option<String>,
}

impl Default for AnalyticsState {
    fn default() -> Self {
        Self {
            statistics: None,
            epsilon_used: 0.0,
            budget_remaining: 1.0,
            loading: false,
            error: None,
        }
    }
}

/// Render the analytics view
pub fn render_analytics(f: &mut Frame, area: Rect, state: &AnalyticsState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Footer
        ])
        .split(area);

    render_analytics_header(f, chunks[0]);
    render_analytics_content(f, chunks[1], state);
    render_analytics_footer(f, chunks[2], state);
}

fn render_analytics_header(f: &mut Frame, area: Rect) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" ", MedicalTheme::text()),
        Span::styled("Analytics", MedicalTheme::title()),
        Span::styled(" │ ε-Differential Privacy Protected", MedicalTheme::text_secondary()),
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(header, area);
}

fn render_analytics_content(f: &mut Frame, area: Rect, state: &AnalyticsState) {
    if state.loading {
        render_loading(f, area);
        return;
    }

    if let Some(err) = &state.error {
        render_analytics_error(f, area, err);
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .margin(1)
        .split(area);

    render_statistics(f, chunks[0], state);
    render_privacy_info(f, chunks[1], state);
}

fn render_loading(f: &mut Frame, area: Rect) {
    let content = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled("Loading statistics...", MedicalTheme::text_muted())),
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(content, area);
}

fn render_analytics_error(f: &mut Frame, area: Rect, message: &str) {
    let content = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled("! Cannot Load Statistics", MedicalTheme::danger())),
        Line::from(""),
        Line::from(Span::styled(message, MedicalTheme::text())),
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(MedicalTheme::danger()),
    );

    f.render_widget(content, area);
}

fn render_statistics(f: &mut Frame, area: Rect, state: &AnalyticsState) {
    let block = Block::default()
        .title(Span::styled(" Aggregate Statistics ", MedicalTheme::subtitle()))
        .borders(Borders::ALL)
        .border_style(MedicalTheme::border());

    let inner = block.inner(area);
    f.render_widget(block, area);

    match &state.statistics {
        Some(stats) => {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Total count
                    Constraint::Length(4), // Positive rate
                    Constraint::Length(4), // Avg confidence
                    Constraint::Min(0),    // Padding
                ])
                .margin(1)
                .split(inner);

            // Total diagnoses (noisy)
            let count_text = Paragraph::new(Line::from(vec![
                Span::styled("Total Diagnoses: ", MedicalTheme::text_secondary()),
                Span::styled(format!("~{:.0}", stats.total_count), MedicalTheme::text()),
                Span::styled(" (±noise)", MedicalTheme::text_muted()),
            ]));
            f.render_widget(count_text, chunks[0]);

            // Positive rate gauge
            let rate_gauge = Gauge::default()
                .block(
                    Block::default()
                        .title(Span::styled(
                            " Positive Diagnosis Rate ",
                            MedicalTheme::text_secondary(),
                        ))
                        .borders(Borders::ALL)
                        .border_style(MedicalTheme::border()),
                )
                .gauge_style(MedicalTheme::gauge(1.0 - stats.positive_rate))
                .percent((stats.positive_rate * 100.0).clamp(0.0, 100.0) as u16)
                .label(format!("~{:.1}%", stats.positive_rate * 100.0));
            f.render_widget(rate_gauge, chunks[1]);

            // Average confidence
            let conf_gauge = Gauge::default()
                .block(
                    Block::default()
                        .title(Span::styled(
                            " Average Confidence ",
                            MedicalTheme::text_secondary(),
                        ))
                        .borders(Borders::ALL)
                        .border_style(MedicalTheme::border()),
                )
                .gauge_style(MedicalTheme::info())
                .percent((stats.avg_confidence * 100.0).clamp(0.0, 100.0) as u16)
                .label(format!("~{:.1}%", stats.avg_confidence * 100.0));
            f.render_widget(conf_gauge, chunks[2]);
        }
        None => {
            let no_data = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "No statistics available",
                    MedicalTheme::text_muted(),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "Press [R] to refresh",
                    MedicalTheme::text_secondary(),
                )),
            ])
            .alignment(Alignment::Center);
            f.render_widget(no_data, inner);
        }
    }
}

fn render_privacy_info(f: &mut Frame, area: Rect, state: &AnalyticsState) {
    let block = Block::default()
        .title(Span::styled(" Privacy Status ", MedicalTheme::subtitle()))
        .borders(Borders::ALL)
        .border_style(MedicalTheme::border());

    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // Budget gauge
            Constraint::Length(3), // Epsilon used
            Constraint::Min(0),    // Info text
        ])
        .margin(1)
        .split(inner);

    // Privacy budget gauge
    let budget_gauge = Gauge::default()
        .block(
            Block::default()
                .title(Span::styled(" ε Budget Remaining ", MedicalTheme::text_secondary()))
                .borders(Borders::ALL)
                .border_style(MedicalTheme::border()),
        )
        .gauge_style(MedicalTheme::gauge(state.budget_remaining))
        .percent((state.budget_remaining * 100.0).clamp(0.0, 100.0) as u16)
        .label(format!("{:.0}%", state.budget_remaining * 100.0));
    f.render_widget(budget_gauge, chunks[0]);

    // Epsilon used
    let epsilon_text = Paragraph::new(Line::from(vec![
        Span::styled("ε Used: ", MedicalTheme::text_secondary()),
        Span::styled(format!("{:.3}", state.epsilon_used), MedicalTheme::text()),
    ]));
    f.render_widget(epsilon_text, chunks[1]);

    // Info text
    let info = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "Statistics are protected with",
            MedicalTheme::text_muted(),
        )),
        Line::from(Span::styled(
            "differential privacy (Laplace)",
            MedicalTheme::text_muted(),
        )),
    ]);
    f.render_widget(info, chunks[2]);
}

fn render_analytics_footer(f: &mut Frame, area: Rect, state: &AnalyticsState) {
    let content = if state.budget_remaining <= 0.0 {
        Line::from(vec![
            Span::styled("! ", MedicalTheme::danger()),
            Span::styled("Privacy budget exhausted", MedicalTheme::danger()),
            Span::styled(" [Esc] ", MedicalTheme::key_hint()),
            Span::styled("Back", MedicalTheme::key_desc()),
        ])
    } else {
        Line::from(vec![
            Span::styled("[R] ", MedicalTheme::key_hint()),
            Span::styled("Refresh (costs ε) ", MedicalTheme::key_desc()),
            Span::styled("[Esc] ", MedicalTheme::key_hint()),
            Span::styled("Back", MedicalTheme::key_desc()),
        ])
    };

    let footer = Paragraph::new(content).block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(footer, area);
}
