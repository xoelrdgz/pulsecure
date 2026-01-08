//! Inference visualization view.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame,
};

use crate::domain::{Diagnosis, RiskLevel};
use crate::tui::styles::MedicalTheme;

/// Inference state
#[derive(Debug, Clone)]
pub enum InferenceState {
    /// Not started
    Idle,
    /// Encrypting patient data
    Encrypting { progress: f64 },
    /// Running FHE computation
    Computing { progress: f64 },
    /// Decrypting result
    Decrypting { progress: f64 },
    /// Completed with result
    Complete { diagnosis: Diagnosis },
    /// Error occurred
    Error { message: String },
}

impl Default for InferenceState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Render the inference visualization
pub fn render_inference(f: &mut Frame, area: Rect, state: &InferenceState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Footer
        ])
        .split(area);

    render_inference_header(f, chunks[0]);
    render_inference_content(f, chunks[1], state);
    render_inference_footer(f, chunks[2], state);
}

fn render_inference_header(f: &mut Frame, area: Rect) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" ", MedicalTheme::text()),
        Span::styled("Encrypted Inference", MedicalTheme::title()),
        Span::styled(" │ Homomorphic Computation", MedicalTheme::text_secondary()),
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(header, area);
}

fn render_inference_content(f: &mut Frame, area: Rect, state: &InferenceState) {
    match state {
        InferenceState::Idle => render_idle(f, area),
        InferenceState::Encrypting { progress } => {
            render_progress(f, area, "Encrypting", *progress, "Encrypting patient data...")
        }
        InferenceState::Computing { progress } => {
            render_progress(f, area, "Computing", *progress, "Running FHE inference (blind compute)...")
        }
        InferenceState::Decrypting { progress } => {
            render_progress(f, area, "Decrypting", *progress, "Decrypting diagnosis result...")
        }
        InferenceState::Complete { diagnosis } => render_result(f, area, diagnosis),
        InferenceState::Error { message } => render_error(f, area, message),
    }
}

fn render_idle(f: &mut Frame, area: Rect) {
    let content = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "Ready to perform encrypted inference",
            MedicalTheme::text_secondary(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Enter patient data to begin",
            MedicalTheme::text_muted(),
        )),
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(content, area);
}

fn render_progress(f: &mut Frame, area: Rect, stage: &str, progress: f64, description: &str) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .margin(2)
        .split(area);

    // Stage label
    let stage_text = Paragraph::new(Line::from(vec![
        Span::styled("Stage: ", MedicalTheme::text_secondary()),
        Span::styled(stage, MedicalTheme::focused()),
    ]))
    .alignment(Alignment::Center);
    f.render_widget(stage_text, chunks[0]);

    // Progress bar
    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(MedicalTheme::border()),
        )
        .gauge_style(MedicalTheme::info())
        .percent((progress * 100.0) as u16)
        .label(format!("{:.0}%", progress * 100.0));
    f.render_widget(gauge, chunks[1]);

    // Description
    let desc = Paragraph::new(Line::from(Span::styled(
        description,
        MedicalTheme::text_muted(),
    )))
    .alignment(Alignment::Center);
    f.render_widget(desc, chunks[2]);
}

fn render_result(f: &mut Frame, area: Rect, diagnosis: &Diagnosis) {
    let block = Block::default()
        .title(Span::styled(" Diagnosis Result ", MedicalTheme::subtitle()))
        .borders(Borders::ALL)
        .border_style(MedicalTheme::border_focused());

    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Risk level
            Constraint::Length(4), // Probability
            Constraint::Length(3), // Confidence
            Constraint::Min(0),    // Padding
        ])
        .margin(1)
        .split(inner);

    // Risk level (big display)
    let risk_style = MedicalTheme::risk_level(diagnosis.risk_level);
    let risk_icon = match diagnosis.risk_level {
        RiskLevel::Low => "OK",
        RiskLevel::Moderate => "!",
        RiskLevel::High => "!",
    };

    let risk_display = Paragraph::new(vec![
        Line::from(Span::styled(
            format!("{} {}", risk_icon, diagnosis.risk_level),
            risk_style.add_modifier(ratatui::style::Modifier::BOLD),
        )),
        Line::from(Span::styled(
            diagnosis.risk_level.description(),
            MedicalTheme::text_secondary(),
        )),
    ])
    .alignment(Alignment::Center);
    f.render_widget(risk_display, chunks[0]);

    // Probability bar
    let prob_gauge = Gauge::default()
        .block(
            Block::default()
                .title(Span::styled(" Disease Probability ", MedicalTheme::text_secondary()))
                .borders(Borders::ALL)
                .border_style(MedicalTheme::border()),
        )
        .gauge_style(risk_style)
        .percent((diagnosis.result.probability * 100.0) as u16)
        .label(format!("{:.1}%", diagnosis.result.probability * 100.0));
    f.render_widget(prob_gauge, chunks[1]);

    // Confidence
    let confidence = Paragraph::new(Line::from(vec![
        Span::styled("Confidence: ", MedicalTheme::text_secondary()),
        Span::styled(
            format!("{:.1}%", diagnosis.result.confidence * 100.0),
            MedicalTheme::text(),
        ),
    ]))
    .alignment(Alignment::Center);
    f.render_widget(confidence, chunks[2]);
}

fn render_error(f: &mut Frame, area: Rect, message: &str) {
    let content = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled("! Error", MedicalTheme::danger())),
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

fn render_inference_footer(f: &mut Frame, area: Rect, state: &InferenceState) {
    let content = match state {
        InferenceState::Complete { .. } => Line::from(vec![
            Span::styled("[Enter] ", MedicalTheme::key_hint()),
            Span::styled("Save & Return ", MedicalTheme::key_desc()),
            Span::styled("[N] ", MedicalTheme::key_hint()),
            Span::styled("New Diagnosis", MedicalTheme::key_desc()),
        ]),
        InferenceState::Error { .. } => Line::from(vec![
            Span::styled("[Enter] ", MedicalTheme::key_hint()),
            Span::styled("Retry ", MedicalTheme::key_desc()),
            Span::styled("[Esc] ", MedicalTheme::key_hint()),
            Span::styled("Cancel", MedicalTheme::key_desc()),
        ]),
        _ => Line::from(vec![Span::styled(
            "Processing...",
            MedicalTheme::text_muted(),
        )]),
    };

    let footer = Paragraph::new(content).block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(footer, area);
}
