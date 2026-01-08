//! OpenDP adapter: Implementation of DifferentialPrivacy.
//!
//! Provides Laplacian noise mechanism for privacy-preserving analytics.
//!
//! # Mutex Behavior
//!
//! This adapter uses `Mutex` for thread-safe RNG access. A poisoned mutex
//! (from a panic in another thread) fails closed by returning an error.
//! The application should treat this as a privacy-critical failure.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::domain::Diagnosis;
use crate::ports::{DpError, DifferentialPrivacy, PrivateStatistics};

/// Configuration for differential privacy.
#[derive(Debug, Clone)]
pub struct PrivacyConfig {
    /// Maximum total epsilon budget
    pub max_epsilon: f64,

    /// Default epsilon per query
    pub default_epsilon: f64,

    /// Relative weights for splitting an aggregation epsilon across metrics.
    ///
    /// Order: (count, positive_rate, avg_confidence).
    /// Values are normalized at use time.
    pub aggregation_epsilon_weights: [f64; 3],
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            max_epsilon: 1.0,
            default_epsilon: 0.1,
            aggregation_epsilon_weights: [1.0, 1.0, 1.0],
        }
    }
}

impl PrivacyConfig {
    /// Load config overrides from environment (best-effort).
    ///
    /// Supported:
    /// - PULSECURE_DP_MAX_EPSILON
    /// - PULSECURE_DP_DEFAULT_EPSILON
    /// - PULSECURE_DP_AGG_EPS_WEIGHTS="count,rate,conf"
    fn from_env_or_default() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("PULSECURE_DP_MAX_EPSILON") {
            if let Ok(x) = v.trim().parse::<f64>() {
                if x.is_finite() && x > 0.0 {
                    cfg.max_epsilon = x;
                }
            }
        }

        if let Ok(v) = std::env::var("PULSECURE_DP_DEFAULT_EPSILON") {
            if let Ok(x) = v.trim().parse::<f64>() {
                if x.is_finite() && x > 0.0 {
                    cfg.default_epsilon = x;
                }
            }
        }

        if let Ok(v) = std::env::var("PULSECURE_DP_AGG_EPS_WEIGHTS") {
            let parts: Vec<_> = v.split(',').map(str::trim).collect();
            if parts.len() == 3 {
                let parsed = [
                    parts[0].parse::<f64>(),
                    parts[1].parse::<f64>(),
                    parts[2].parse::<f64>(),
                ];
                if let [Ok(a), Ok(b), Ok(c)] = parsed {
                    if a.is_finite() && b.is_finite() && c.is_finite() && a > 0.0 && b > 0.0 && c > 0.0 {
                        cfg.aggregation_epsilon_weights = [a, b, c];
                    }
                }
            }
        }

        cfg
    }
}

/// OpenDP adapter for differential privacy.
///
/// Implements the Laplacian mechanism for adding calibrated noise
/// to aggregate statistics.
///
/// # Security
///
/// - Uses fixed-point arithmetic for epsilon tracking (no IEEE 754 attacks)
/// - Global sensitivity bounds independent of actual data size
pub struct OpenDpAdapter {
    config: PrivacyConfig,

    /// Total epsilon spent, scaled by EPSILON_SCALE for precise atomic ops
    epsilon_spent_scaled: Arc<AtomicU64>,

    /// CSPRNG for noise generation
    rng: Arc<std::sync::Mutex<ChaCha20Rng>>,
}

/// Scale factor for fixed-point epsilon arithmetic.
/// Epsilon is stored as (epsilon * EPSILON_SCALE) to avoid IEEE 754 precision issues.
const EPSILON_SCALE: f64 = 1_000_000_000.0;

// NOTE: For DP correctness, sensitivities must never be underestimated.
// The aggregation code uses conservative (data-independent) global sensitivities.

impl OpenDpAdapter {
    /// Create a new OpenDP adapter with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(PrivacyConfig::from_env_or_default())
    }

    /// Create a new OpenDP adapter with custom configuration.
    #[must_use]
    pub fn with_config(config: PrivacyConfig) -> Self {
        let rng = ChaCha20Rng::from_entropy();

        Self {
            config,
            epsilon_spent_scaled: Arc::new(AtomicU64::new(0)),
            rng: Arc::new(std::sync::Mutex::new(rng)),
        }
    }

    #[cfg(test)]
    fn with_test_seed(config: PrivacyConfig, seed: [u8; 32]) -> Self {
        let rng = ChaCha20Rng::from_seed(seed);
        Self {
            config,
            epsilon_spent_scaled: Arc::new(AtomicU64::new(0)),
            rng: Arc::new(std::sync::Mutex::new(rng)),
        }
    }

    /// Sample from Laplace distribution.
    fn sample_laplace(&self, scale: f64) -> Result<f64, DpError> {
        let mut rng = self.rng.lock().map_err(|_| DpError::RngUnavailable)?;

        // Laplace distribution via inverse CDF
        // Laplace(0, b) where b = scale
        // IMPORTANT: avoid exact endpoints that would yield ln(0) => +/-inf.
        let mut u01: f64 = rng.gen(); // [0, 1)
        if u01 == 0.0 {
            u01 = f64::MIN_POSITIVE;
        }
        let u: f64 = u01 - 0.5; // (-0.5, 0.5)

        let inner: f64 = 1.0 - 2.0 * u.abs();
        Ok(-scale * u.signum() * inner.ln())
    }

    fn add_laplace_noise_without_consuming_budget(
        &self,
        value: f64,
        sensitivity: f64,
        epsilon: f64,
    ) -> Result<f64, DpError> {
        if sensitivity == 0.0 {
            return Ok(value);
        }
        let scale = sensitivity / epsilon;
        let noise = self.sample_laplace(scale)?;
        Ok(value + noise)
    }

    fn max_epsilon_scaled(&self) -> u64 {
        if !self.config.max_epsilon.is_finite() || self.config.max_epsilon <= 0.0 {
            // Fail-safe: a non-positive max budget would break DP accounting.
            tracing::error!("Invalid max_epsilon configured: {}", self.config.max_epsilon);
            return 0;
        }
        (self.config.max_epsilon * EPSILON_SCALE).round().max(0.0) as u64
    }

    /// Atomically consume epsilon from the global budget.
    ///
    /// This prevents concurrent queries from overspending the privacy budget.
    fn try_consume_epsilon(&self, epsilon: f64) -> bool {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return false;
        }

        let max_scaled = self.max_epsilon_scaled();
        let delta = (epsilon * EPSILON_SCALE).round().max(0.0) as u64;

        loop {
            let current = self.epsilon_spent_scaled.load(Ordering::SeqCst);
            if current > max_scaled {
                return false;
            }
            if max_scaled - current < delta {
                return false;
            }
            let next = current + delta;
            match self.epsilon_spent_scaled.compare_exchange_weak(
                current,
                next,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(_) => continue,
            }
        }
    }

    fn split_epsilon_for_aggregate(&self, epsilon: f64) -> [f64; 3] {
        let [w0, w1, w2] = self.config.aggregation_epsilon_weights;
        let sum = w0 + w1 + w2;

        if !(sum.is_finite()) || sum <= 0.0 || w0 <= 0.0 || w1 <= 0.0 || w2 <= 0.0 {
            let e = epsilon / 3.0;
            return [e, e, e];
        }

        [epsilon * (w0 / sum), epsilon * (w1 / sum), epsilon * (w2 / sum)]
    }
}

impl Default for OpenDpAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentialPrivacy for OpenDpAdapter {
    fn add_laplace_noise(&self, value: f64, sensitivity: f64, epsilon: f64) -> Result<f64, DpError> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            tracing::error!("Invalid epsilon: {epsilon}. Refusing to release statistic.");
            return Err(DpError::InvalidEpsilon(epsilon));
        }

        if !sensitivity.is_finite() || sensitivity < 0.0 {
            tracing::error!("Invalid sensitivity: {sensitivity}. Refusing to release statistic.");
            return Err(DpError::InvalidSensitivity(sensitivity));
        }

        // Consume privacy budget first (fail-closed if exhausted).
        if !self.try_consume_epsilon(epsilon) {
            tracing::warn!(
                "Privacy budget exhausted: need {epsilon}, remaining {:.9}",
                self.budget_remaining()
            );
            return Err(DpError::BudgetExhausted);
        }

        self.add_laplace_noise_without_consuming_budget(value, sensitivity, epsilon)
    }

    fn aggregate(&self, diagnoses: &[Diagnosis], epsilon: f64) -> Result<PrivateStatistics, DpError> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            tracing::error!("Invalid epsilon for aggregate: {epsilon}");
            return Err(DpError::InvalidEpsilon(epsilon));
        }

        // Reserve the entire epsilon for this aggregation as a single atomic operation.
        // This prevents concurrent overspending.
        if !self.try_consume_epsilon(epsilon) {
            tracing::warn!(
                "Privacy budget exhausted: need {epsilon}, remaining {:.9}",
                self.budget_remaining()
            );
            return Err(DpError::BudgetExhausted);
        }

        // Split epsilon budget among queries
        let [eps_count, eps_rate, eps_conf] = self.split_epsilon_for_aggregate(epsilon);

        // True values
        let true_count = diagnoses.len() as f64;
        let true_positive_count = diagnoses
            .iter()
            .filter(|d| d.result.prediction == 1)
            .count() as f64;
        let true_confidence_sum: f64 = diagnoses.iter().map(|d| d.result.confidence).sum();

        // Add noise to each statistic
        // Sensitivity for count = 1 (one person changes count by 1)
        let noisy_count = if true_count == 0.0 {
            0.0
        } else {
            self.add_laplace_noise_without_consuming_budget(true_count, 1.0, eps_count)?
                .max(0.0)
        };

        // Conservative global sensitivity bounds (do not depend on actual dataset size).
        // For values bounded in [0,1] (rates/means), worst-case change between neighboring
        // datasets can be up to 1.0.
        let sensitivity_rate = 1.0;

        // Avoid division-by-zero leakage: define a canonical value when there is no data,
        // and still apply noise.
        let true_rate = if true_count > 0.0 {
            true_positive_count / true_count
        } else {
            0.0
        };
        let noisy_rate = self
            .add_laplace_noise_without_consuming_budget(true_rate, sensitivity_rate, eps_rate)?
            .clamp(0.0, 1.0);

        // Sensitivity for average confidence (bounded in [0,1])
        let true_avg_conf = if true_count > 0.0 {
            true_confidence_sum / true_count
        } else {
            0.0
        };
        let noisy_avg_conf = self
            .add_laplace_noise_without_consuming_budget(true_avg_conf, sensitivity_rate, eps_conf)?
            .clamp(0.0, 1.0);

        Ok(PrivateStatistics {
            total_count: noisy_count,
            positive_rate: noisy_rate,
            avg_confidence: noisy_avg_conf,
            epsilon_spent: epsilon,
            budget_remaining: self.budget_remaining(),
        })
    }

    fn total_epsilon_spent(&self) -> f64 {
        self.epsilon_spent_scaled.load(Ordering::SeqCst) as f64 / EPSILON_SCALE
    }

    fn budget_remaining(&self) -> f64 {
        let max_scaled = self.max_epsilon_scaled();
        let spent_scaled = self.epsilon_spent_scaled.load(Ordering::SeqCst);
        max_scaled
            .saturating_sub(spent_scaled) as f64
            / EPSILON_SCALE
    }

    fn can_query(&self, epsilon: f64) -> bool {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return false;
        }
        let max_scaled = self.max_epsilon_scaled();
        let spent_scaled = self.epsilon_spent_scaled.load(Ordering::SeqCst);
        let needed_scaled = (epsilon * EPSILON_SCALE).round().max(0.0) as u64;
        spent_scaled <= max_scaled && max_scaled - spent_scaled >= needed_scaled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DiagnosisResult;

    fn sample_diagnoses() -> Vec<Diagnosis> {
        (0..10)
            .map(|i| {
                let prob = (i as f64) / 10.0;
                Diagnosis::new(DiagnosisResult::new(prob), true)
            })
            .collect()
    }

    #[test]
    fn test_laplace_noise() {
        let adapter = OpenDpAdapter::with_test_seed(PrivacyConfig::default(), [7u8; 32]);

        let noisy = adapter
            .add_laplace_noise(100.0, 1.0, 0.01)
            .expect("DP noise should work");
        assert!(noisy.is_finite());
        assert_ne!(noisy, 100.0);
    }

    #[test]
    fn test_aggregate() {
        let adapter = OpenDpAdapter::new();
        let diagnoses = sample_diagnoses();

        let stats = adapter.aggregate(&diagnoses, 0.1).expect("DP aggregate should work");

        // Noisy values should be in reasonable range
        assert!(stats.total_count >= 0.0);
        assert!(stats.positive_rate >= 0.0 && stats.positive_rate <= 1.0);
        assert!(stats.avg_confidence >= 0.0 && stats.avg_confidence <= 1.0);
    }

    #[test]
    fn test_budget_tracking() {
        let adapter = OpenDpAdapter::new();

        assert!(adapter.can_query(0.1));
        assert_eq!(adapter.total_epsilon_spent(), 0.0);

        adapter
            .add_laplace_noise(1.0, 1.0, 0.5)
            .expect("DP noise should work");
        assert!(adapter.total_epsilon_spent() > 0.0);
    }
}
