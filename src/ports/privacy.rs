use crate::domain::Diagnosis;

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

#[derive(Debug, Clone)]
pub struct PrivateStatistics {
    pub total_count: f64,

    pub positive_rate: f64,

    pub avg_confidence: f64,

    pub epsilon_spent: f64,

    pub budget_remaining: f64,
}

pub trait DifferentialPrivacy: Send + Sync {
    fn add_laplace_noise(&self, value: f64, sensitivity: f64, epsilon: f64)
        -> Result<f64, DpError>;

    fn aggregate(
        &self,
        diagnoses: &[Diagnosis],
        epsilon: f64,
    ) -> Result<PrivateStatistics, DpError>;

    fn total_epsilon_spent(&self) -> f64;

    fn budget_remaining(&self) -> f64;

    fn can_query(&self, epsilon: f64) -> bool;
}
