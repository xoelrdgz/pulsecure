//! Differential Privacy port: Trait for privacy-preserving analytics.
//!
//! This trait abstracts the DP library (OpenDP) from the application logic.

use crate::domain::Diagnosis;

/// Errors that can occur during differential privacy operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum DpError {
    #[error("Invalid epsilon: {0}")]
    InvalidEpsilon(f64),

    #[error("Invalid sensitivity: {0}")]
    InvalidSensitivity(f64),

    #[error("Privacy budget exhausted")]
    BudgetExhausted,

    #[error("DP RNG unavailable")]
    RngUnavailable,
}

/// Statistics with differential privacy protection.
#[derive(Debug, Clone)]
pub struct PrivateStatistics {
    /// Total number of diagnoses (noisy)
    pub total_count: f64,

    /// Percentage with positive diagnosis (noisy)
    pub positive_rate: f64,

    /// Average confidence score (noisy)
    pub avg_confidence: f64,

    /// Epsilon spent for this query
    pub epsilon_spent: f64,

    /// Remaining privacy budget
    pub budget_remaining: f64,
}

/// Trait for differential privacy operations.
///
/// Implementations provide mechanisms for adding calibrated noise
/// to aggregate statistics, preventing re-identification attacks.
pub trait DifferentialPrivacy: Send + Sync {
    /// Add Laplacian noise to a single value.
    ///
    /// # Arguments
    /// * `value` - The true value to protect
    /// * `sensitivity` - Maximum change from one person's data
    /// * `epsilon` - Privacy budget for this query
    ///
    /// # Returns
    /// The noisy value.
    fn add_laplace_noise(&self, value: f64, sensitivity: f64, epsilon: f64) -> Result<f64, DpError>;

    /// Aggregate diagnoses with differential privacy.
    ///
    /// Computes statistics over a set of diagnoses with privacy protection.
    ///
    /// # Arguments
    /// * `diagnoses` - List of diagnoses to aggregate
    /// * `epsilon` - Privacy budget for this aggregation
    ///
    /// # Returns
    /// Private statistics that are safe to release.
    fn aggregate(&self, diagnoses: &[Diagnosis], epsilon: f64) -> Result<PrivateStatistics, DpError>;

    /// Get the total privacy budget consumed so far.
    fn total_epsilon_spent(&self) -> f64;

    /// Get the remaining privacy budget.
    fn budget_remaining(&self) -> f64;

    /// Check if a query with the given epsilon can be performed.
    fn can_query(&self, epsilon: f64) -> bool;
}
