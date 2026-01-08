//! TUI module: Terminal User Interface using Ratatui.
//!
//! Provides a professional medical-themed interface for:
//! - Dashboard with system status
//! - Patient data input
//! - Encrypted inference visualization
//! - Privacy-preserving analytics

mod app;
mod styles;
mod ui;
mod worker;

pub use app::App;
pub use styles::MedicalTheme;
pub use worker::{InferenceProgress, InferenceWorker, InferenceWorkerHandle};

