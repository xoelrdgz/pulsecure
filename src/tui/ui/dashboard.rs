//! Dashboard view: Main overview screen.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame,
};

use crate::domain::RiskLevel;
use crate::tui::styles::MedicalTheme;

#[derive(Debug, Clone, Copy, Default)]
pub struct RecentSummary {
    pub total: usize,
    pub low: u32,
    pub moderate: u32,
    pub high: u32,
    pub encrypted: u32,
}

/// Dashboard state for rendering.
pub struct DashboardState {
    pub keys_generated: bool,
    pub show_client_fingerprint: bool,
    pub model_loaded: bool,
    pub diagnosis_count: usize,
    pub privacy_budget: f64,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            keys_generated: false,
            show_client_fingerprint: false,
            model_loaded: false,
            diagnosis_count: 0,
            privacy_budget: 1.0,
        }
    }
}

/// Render the main dashboard view.
pub fn render_dashboard(
    f: &mut Frame,
    area: Rect,
    state: &DashboardState,
    recent: RecentSummary,
    client_fingerprint: Option<&str>,
) {
    // Split into header and main content
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Main content
        ])
        .split(area);

    render_header(f, chunks[0]);
    render_main_content(f, chunks[1], state, recent, client_fingerprint);
}

fn render_header(f: &mut Frame, area: Rect) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" ", MedicalTheme::text()),
        Span::styled("Pulsecure", MedicalTheme::title()),
        Span::styled(" │ ", MedicalTheme::text_muted()),
        Span::styled("Privacy-Preserving Medical Diagnostics", MedicalTheme::text_secondary()),
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(header, area);
}

fn render_main_content(
    f: &mut Frame,
    area: Rect,
    state: &DashboardState,
    recent: RecentSummary,
    client_fingerprint: Option<&str>,
) {
    // Split into left (status) and right (recent diagnoses)
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Status panels
            Constraint::Percentage(60), // Recent diagnoses
        ])
        .split(area);

    render_status_panels(f, chunks[0], state, client_fingerprint);
    render_recent_summary(f, chunks[1], recent);
}

fn render_status_panels(
    f: &mut Frame,
    area: Rect,
    state: &DashboardState,
    client_fingerprint: Option<&str>,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6), // System status
            Constraint::Length(5), // Privacy budget
            Constraint::Min(0),    // Quick actions
        ])
        .margin(1)
        .split(area);

    // System Status
    let mut status_items = vec![
        format_status_item("FHE Keys", state.keys_generated),
        format_status_item("Model Loaded", state.model_loaded),
        Line::from(vec![
            Span::styled("  Diagnoses: ", MedicalTheme::text_secondary()),
            Span::styled(state.diagnosis_count.to_string(), MedicalTheme::text()),
        ]),
    ];

    if state.show_client_fingerprint {
        let fp = client_fingerprint.unwrap_or("<unavailable>");
        status_items.push(Line::from(vec![
            Span::styled("  Client FP: ", MedicalTheme::text_secondary()),
            Span::styled(fp.to_string(), MedicalTheme::text_muted()),
        ]));
    } else {
        status_items.push(Line::from(vec![
            Span::styled("  Client FP: ", MedicalTheme::text_secondary()),
            Span::styled("hidden", MedicalTheme::text_muted()),
            Span::styled(" (press [F])", MedicalTheme::text_secondary()),
        ]));
    }

    let status_block = Block::default()
        .title(Span::styled(" System Status ", MedicalTheme::subtitle()))
        .borders(Borders::ALL)
        .border_style(MedicalTheme::border());

    let status_list = Paragraph::new(status_items).block(status_block);
    f.render_widget(status_list, chunks[0]);

    // Privacy Budget Gauge
    let privacy_block = Block::default()
        .title(Span::styled(" ε Privacy Budget ", MedicalTheme::subtitle()))
        .borders(Borders::ALL)
        .border_style(MedicalTheme::border());

    let privacy_gauge = Gauge::default()
        .block(privacy_block)
        .gauge_style(MedicalTheme::gauge(state.privacy_budget))
        .percent((state.privacy_budget * 100.0) as u16)
        .label(format!("{:.0}%", state.privacy_budget * 100.0));

    f.render_widget(privacy_gauge, chunks[1]);

    // Quick Actions
    let actions = vec![
        Line::from(vec![
            Span::styled("[N] ", MedicalTheme::key_hint()),
            Span::styled("New Diagnosis", MedicalTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("[A] ", MedicalTheme::key_hint()),
            Span::styled("Analytics", MedicalTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("[K] ", MedicalTheme::key_hint()),
            Span::styled("Regenerate Keys", MedicalTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("[F] ", MedicalTheme::key_hint()),
            Span::styled("Toggle Fingerprint", MedicalTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("[Q] ", MedicalTheme::key_hint()),
            Span::styled("Quit", MedicalTheme::key_desc()),
        ]),
    ];

    let actions_block = Block::default()
        .title(Span::styled(" Quick Actions ", MedicalTheme::subtitle()))
        .borders(Borders::ALL)
        .border_style(MedicalTheme::border());

    let actions_list = Paragraph::new(actions).block(actions_block);
    f.render_widget(actions_list, chunks[2]);
}

fn format_status_item(label: &str, ok: bool) -> Line<'static> {
    let (icon, style) = if ok {
        ("OK", MedicalTheme::success())
    } else {
        ("FAIL", MedicalTheme::danger())
    };

    Line::from(vec![
        Span::styled(format!("  {icon} "), style),
        Span::styled(label.to_string(), MedicalTheme::text()),
    ])
}

fn render_recent_summary(f: &mut Frame, area: Rect, recent: RecentSummary) {
    let block = Block::default()
        .title(Span::styled(
            " Recent Activity (Aggregated) ",
            MedicalTheme::subtitle(),
        ))
        .borders(Borders::ALL)
        .border_style(MedicalTheme::border());

    if recent.total == 0 {
        let empty_msg = Paragraph::new(Line::from(vec![Span::styled(
            "No diagnoses yet. Press [N] to start.",
            MedicalTheme::text_muted(),
        )]))
        .block(block);
        f.render_widget(empty_msg, area);
        return;
    }

    let total = recent.total;

    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines = vec![
        Line::from(vec![
            Span::styled("Last ", MedicalTheme::text_secondary()),
            Span::styled(total.to_string(), MedicalTheme::text()),
            Span::styled(" diagnoses (details hidden)", MedicalTheme::text_muted()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Low: ", MedicalTheme::text_secondary()),
            Span::styled(recent.low.to_string(), MedicalTheme::risk_level(RiskLevel::Low)),
            Span::styled("  ", MedicalTheme::text()),
            Span::styled("Moderate: ", MedicalTheme::text_secondary()),
            Span::styled(
                recent.moderate.to_string(),
                MedicalTheme::risk_level(RiskLevel::Moderate),
            ),
        ]),
        Line::from(vec![
            Span::styled("High: ", MedicalTheme::text_secondary()),
            Span::styled(recent.high.to_string(), MedicalTheme::risk_level(RiskLevel::High)),
            Span::styled("  ", MedicalTheme::text()),
            Span::styled("Encrypted: ", MedicalTheme::text_secondary()),
            Span::styled(recent.encrypted.to_string(), MedicalTheme::info()),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "For privacy, the dashboard does not display probabilities or timestamps.",
            MedicalTheme::text_muted(),
        )]),
    ];

    let p = Paragraph::new(lines).block(Block::default());
    f.render_widget(p, inner);
}
