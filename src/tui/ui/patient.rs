//! Patient data input form.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::domain::PatientFeatures;
use crate::tui::styles::MedicalTheme;
use zeroize::Zeroize;

/// Form field definition
#[derive(Debug, Clone)]
pub struct FormField {
    pub label: &'static str,
    pub hint: &'static str,
    pub value: String,
    pub min: f64,
    pub max: f64,
}

/// Patient form state
pub struct PatientFormState {
    pub fields: Vec<FormField>,
    pub selected_field: usize,
    pub error_message: Option<String>,
}

impl Default for PatientFormState {
    fn default() -> Self {
        Self {
            fields: vec![
                FormField {
                    label: "Age",
                    hint: "years (18-120)",
                    value: String::new(),
                    min: 18.0,
                    max: 120.0,
                },
                FormField {
                    label: "Hypertension",
                    hint: "0=no, 1=yes (diagnosed)",
                    value: String::new(),
                    min: 0.0,
                    max: 1.0,
                },
                FormField {
                    label: "Systolic BP",
                    hint: "mmHg (50-250)",
                    value: String::new(),
                    min: 50.0,
                    max: 250.0,
                },
                FormField {
                    label: "Smoking",
                    hint: "0=no, 1=yes (100+ cigs ever)",
                    value: String::new(),
                    min: 0.0,
                    max: 1.0,
                },
                FormField {
                    label: "HDL Cholesterol",
                    hint: "mg/dL (5-200)",
                    value: String::new(),
                    min: 5.0,
                    max: 200.0,
                },
                FormField {
                    label: "Creatinine",
                    hint: "mg/dL (0.1-20)",
                    value: String::new(),
                    min: 0.1,
                    max: 20.0,
                },
                FormField {
                    label: "Waist Circ",
                    hint: "cm (30-200)",
                    value: String::new(),
                    min: 30.0,
                    max: 200.0,
                },
                FormField {
                    label: "Diabetes",
                    hint: "0=no, 1=yes (diagnosed)",
                    value: String::new(),
                    min: 0.0,
                    max: 1.0,
                },
                FormField {
                    label: "HbA1c",
                    hint: "% (3.0-20)",
                    value: String::new(),
                    min: 3.0,
                    max: 20.0,
                },
            ],
            selected_field: 0,
            error_message: None,
        }
    }
}

impl PatientFormState {
    /// Move to the next field
    pub fn next_field(&mut self) {
        self.selected_field = (self.selected_field + 1) % self.fields.len();
    }

    /// Move to the previous field
    pub fn prev_field(&mut self) {
        if self.selected_field == 0 {
            self.selected_field = self.fields.len() - 1;
        } else {
            self.selected_field -= 1;
        }
    }

    /// Add a character to the current field
    pub fn input_char(&mut self, c: char) {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            self.fields[self.selected_field].value.push(c);
            self.error_message = None;
        }
    }

    /// Delete the last character
    pub fn delete_char(&mut self) {
        self.fields[self.selected_field].value.pop();
    }

    /// Clear the current field
    pub fn clear_field(&mut self) {
        self.fields[self.selected_field].value.clear();
    }

    /// Wipe all field buffers from memory and clear values.
    ///
    /// Intended to be called immediately after starting inference so plaintext
    /// inputs do not persist in the UI state.
    pub fn clear_sensitive(&mut self) {
        for field in self.fields.iter_mut() {
            field.value.zeroize();
        }
        self.error_message = None;
        self.selected_field = 0;
    }

    /// Validate and convert to PatientFeatures
    pub fn to_patient_features(&self) -> Result<PatientFeatures, String> {
        let mut values = Vec::with_capacity(9);

        for field in self.fields.iter() {
            let value: f64 = field
                .value
                .parse()
                .map_err(|_| format!("{}: Invalid number", field.label))?;

            if value < field.min || value > field.max {
                return Err(format!(
                    "{}: Value must be between {} and {}",
                    field.label, field.min, field.max
                ));
            }

            values.push(value);
        }

        PatientFeatures::from_vec(&values)
    }

    /// Load sample data for testing (typical CVD risk patient)
    pub fn load_sample_data(&mut self) {
        // Sample: 55yo, hypertension, prediabetic, smoker (9 features)
        let sample = [
            "55",  // age (years)
            "1",   // hypertension (diagnosed)
            "142", // sys_bp (mmHg)
            "1",   // smoking (yes)
            "45",  // hdl_chol (mg/dL)
            "1.1", // creatinine (mg/dL)
            "102", // waist_circ (cm)
            "0",   // diabetes (no)
            "5.9", // hba1c (%)
        ];
        for (i, val) in sample.iter().enumerate() {
            self.fields[i].value = val.to_string();
        }
    }
}

/// Render the patient data input form
pub fn render_patient_form(f: &mut Frame, area: Rect, state: &PatientFormState) {
    // Split into header and form
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Form
            Constraint::Length(3), // Footer/error
        ])
        .split(area);

    render_form_header(f, chunks[0]);
    render_form_fields(f, chunks[1], state);
    render_form_footer(f, chunks[2], state);
}

fn render_form_header(f: &mut Frame, area: Rect) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" ", MedicalTheme::text()),
        Span::styled("Patient Data Entry", MedicalTheme::title()),
        Span::styled(
            " │ NHANES Cardiovascular Features",
            MedicalTheme::text_secondary(),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(header, area);
}

fn render_form_fields(f: &mut Frame, area: Rect, state: &PatientFormState) {
    // Create a two-column layout
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .margin(1)
        .split(area);

    let mid = (state.fields.len() + 1) / 2;

    // Left column
    render_field_column(f, columns[0], &state.fields[..mid], 0, state.selected_field);

    // Right column
    render_field_column(
        f,
        columns[1],
        &state.fields[mid..],
        mid,
        state.selected_field,
    );
}

fn render_field_column(
    f: &mut Frame,
    area: Rect,
    fields: &[FormField],
    offset: usize,
    selected: usize,
) {
    let field_height = 3;
    let constraints: Vec<Constraint> = fields
        .iter()
        .map(|_| Constraint::Length(field_height))
        .chain(std::iter::once(Constraint::Min(0)))
        .collect();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    for (i, field) in fields.iter().enumerate() {
        let is_selected = offset + i == selected;
        let border_style = if is_selected {
            MedicalTheme::border_focused()
        } else {
            MedicalTheme::border()
        };

        let title_style = if is_selected {
            MedicalTheme::focused()
        } else {
            MedicalTheme::text_secondary()
        };

        let block = Block::default()
            .title(Span::styled(format!(" {} ", field.label), title_style))
            .borders(Borders::ALL)
            .border_style(border_style);

        let value_display = if field.value.is_empty() {
            Span::styled(field.hint, MedicalTheme::text_muted())
        } else {
            Span::styled(&field.value, MedicalTheme::text())
        };

        let content = Paragraph::new(Line::from(vec![
            Span::raw(" "),
            value_display,
            if is_selected {
                Span::styled("▌", MedicalTheme::primary_cursor())
            } else {
                Span::raw("")
            },
        ]))
        .block(block);

        f.render_widget(content, chunks[i]);
    }
}

fn render_form_footer(f: &mut Frame, area: Rect, state: &PatientFormState) {
    let content = if let Some(err) = &state.error_message {
        Line::from(vec![
            Span::styled("! ", MedicalTheme::danger()),
            Span::styled(err.clone(), MedicalTheme::danger()),
        ])
    } else {
        Line::from(vec![
            Span::styled("[↑↓] ", MedicalTheme::key_hint()),
            Span::styled("Navigate ", MedicalTheme::key_desc()),
            Span::styled("[Enter] ", MedicalTheme::key_hint()),
            Span::styled("Submit ", MedicalTheme::key_desc()),
            Span::styled("[S] ", MedicalTheme::key_hint()),
            Span::styled("Sample Data ", MedicalTheme::key_desc()),
            Span::styled("[Esc] ", MedicalTheme::key_hint()),
            Span::styled("Cancel", MedicalTheme::key_desc()),
        ])
    };

    let footer = Paragraph::new(content).block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(MedicalTheme::border()),
    );

    f.render_widget(footer, area);
}

// Helper trait extension
trait MedicalThemeExt {
    fn primary_cursor() -> ratatui::style::Style;
}

impl MedicalThemeExt for MedicalTheme {
    fn primary_cursor() -> ratatui::style::Style {
        ratatui::style::Style::default().fg(MedicalTheme::PRIMARY_LIGHT)
    }
}
