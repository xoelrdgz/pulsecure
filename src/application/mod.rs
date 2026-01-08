//! Application layer: Use cases and services.
//!
//! This module orchestrates domain logic with ports to implement
//! the core use cases of the application.

mod analytics;
mod inference;

pub use analytics::AnalyticsService;
pub use inference::InferenceService;
