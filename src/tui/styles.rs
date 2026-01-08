//! Medical-themed color palette and styles.
//!
//! Colors chosen for:
//! - Professional healthcare appearance
//! - High contrast for accessibility
//! - Trust and calm associations

use ratatui::style::{Color, Modifier, Style};

/// Medical theme color palette.
pub struct MedicalTheme;

impl MedicalTheme {
    // === Primary Colors ===

    /// Deep teal - Primary color (trust, medical)
    pub const PRIMARY: Color = Color::Rgb(13, 148, 136); // #0D9488

    /// Lighter teal for highlights
    pub const PRIMARY_LIGHT: Color = Color::Rgb(45, 212, 191); // #2DD4BF

    /// Darker teal for accents
    pub const PRIMARY_DARK: Color = Color::Rgb(15, 118, 110); // #0F766E

    // === Secondary Colors ===

    /// Slate blue - Secondary (professionalism)
    pub const SECONDARY: Color = Color::Rgb(71, 85, 105); // #475569

    /// Light slate for borders
    pub const SECONDARY_LIGHT: Color = Color::Rgb(148, 163, 184); // #94A3B8

    // === Semantic Colors ===

    /// Emerald - Success/healthy
    pub const SUCCESS: Color = Color::Rgb(16, 185, 129); // #10B981

    /// Amber - Warning/moderate risk
    pub const WARNING: Color = Color::Rgb(251, 191, 36); // #FBBF24

    /// Rose - Error/high risk
    pub const DANGER: Color = Color::Rgb(244, 63, 94); // #F43F5E

    /// Blue - Info
    pub const INFO: Color = Color::Rgb(59, 130, 246); // #3B82F6

    // === Background Colors ===

    /// Near-black with blue tint
    pub const BG_DARK: Color = Color::Rgb(15, 23, 42); // #0F172A

    /// Slightly lighter background
    pub const BG_SURFACE: Color = Color::Rgb(30, 41, 59); // #1E293B

    /// Card/panel background
    pub const BG_CARD: Color = Color::Rgb(51, 65, 85); // #334155

    // === Text Colors ===

    /// Primary text (white)
    pub const TEXT_PRIMARY: Color = Color::Rgb(248, 250, 252); // #F8FAFC

    /// Secondary text (gray)
    pub const TEXT_SECONDARY: Color = Color::Rgb(148, 163, 184); // #94A3B8

    /// Muted text
    pub const TEXT_MUTED: Color = Color::Rgb(100, 116, 139); // #64748B

    // === Preset Styles ===

    /// Style for titles
    #[must_use]
    pub fn title() -> Style {
        Style::default()
            .fg(Self::TEXT_PRIMARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for subtitles
    #[must_use]
    pub fn subtitle() -> Style {
        Style::default()
            .fg(Self::PRIMARY_LIGHT)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for normal text
    #[must_use]
    pub fn text() -> Style {
        Style::default().fg(Self::TEXT_PRIMARY)
    }

    /// Style for secondary text
    #[must_use]
    pub fn text_secondary() -> Style {
        Style::default().fg(Self::TEXT_SECONDARY)
    }

    /// Style for muted text
    #[must_use]
    pub fn text_muted() -> Style {
        Style::default().fg(Self::TEXT_MUTED)
    }

    /// Style for success messages
    #[must_use]
    pub fn success() -> Style {
        Style::default().fg(Self::SUCCESS)
    }

    /// Style for warning messages
    #[must_use]
    pub fn warning() -> Style {
        Style::default().fg(Self::WARNING)
    }

    /// Style for danger/error messages
    #[must_use]
    pub fn danger() -> Style {
        Style::default().fg(Self::DANGER)
    }

    /// Style for info messages
    #[must_use]
    pub fn info() -> Style {
        Style::default().fg(Self::INFO)
    }

    /// Style for selected items
    #[must_use]
    pub fn selected() -> Style {
        Style::default()
            .fg(Self::BG_DARK)
            .bg(Self::PRIMARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for focused elements
    #[must_use]
    pub fn focused() -> Style {
        Style::default()
            .fg(Self::PRIMARY_LIGHT)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for borders
    #[must_use]
    pub fn border() -> Style {
        Style::default().fg(Self::SECONDARY_LIGHT)
    }

    /// Style for focused borders
    #[must_use]
    pub fn border_focused() -> Style {
        Style::default().fg(Self::PRIMARY)
    }

    /// Style for the header
    #[must_use]
    pub fn header() -> Style {
        Style::default()
            .fg(Self::TEXT_PRIMARY)
            .bg(Self::PRIMARY_DARK)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for key hints
    #[must_use]
    pub fn key_hint() -> Style {
        Style::default()
            .fg(Self::PRIMARY_LIGHT)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for key descriptions
    #[must_use]
    pub fn key_desc() -> Style {
        Style::default().fg(Self::TEXT_SECONDARY)
    }

    /// Get risk level style
    #[must_use]
    pub fn risk_level(level: crate::domain::RiskLevel) -> Style {
        match level {
            crate::domain::RiskLevel::Low => Self::success(),
            crate::domain::RiskLevel::Moderate => Self::warning(),
            crate::domain::RiskLevel::High => Self::danger(),
        }
    }

    /// Get gauge style based on percentage
    #[must_use]
    pub fn gauge(percentage: f64) -> Style {
        if percentage >= 0.7 {
            Self::success()
        } else if percentage >= 0.3 {
            Self::warning()
        } else {
            Self::danger()
        }
    }
}

/// ASCII art logo for Pulsecure
#[allow(dead_code)]
pub const LOGO: &str = r#"
╔╦╗┌─┐┌┬┐╦  ╦┌─┐┬ ┬┬ ┌┬┐
║║║├┤  ││╚╗╔╝├─┤│ ││  │ 
╩ ╩└─┘─┴┘ ╚╝ ┴ ┴└─┘┴─┘┴ 
"#;

/// Smaller inline logo
#[allow(dead_code)]
pub const LOGO_SMALL: &str = "Pulsecure";
