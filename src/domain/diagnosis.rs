//! Diagnosis result types.
//!
//! Represents the output of the FHE-based heart disease prediction.

use serde::{Deserialize, Serialize};

/// Risk level classification for heart disease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk of heart disease
    Low,
    /// Moderate risk, monitoring recommended
    Moderate,
    /// High risk, intervention recommended
    High,
}

impl RiskLevel {
    /// Get a human-readable description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Low => "Low risk - No significant indicators",
            Self::Moderate => "Moderate risk - Follow-up recommended",
            Self::High => "High risk - Immediate consultation advised",
        }
    }

    /// Get the associated color for TUI display (RGB).
    #[must_use]
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Self::Low => (16, 185, 129),    // Emerald (#10B981)
            Self::Moderate => (251, 191, 36), // Amber (#FBBF24)
            Self::High => (244, 63, 94),     // Rose (#F43F5E)
        }
    }
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Moderate => write!(f, "MODERATE"),
            Self::High => write!(f, "HIGH"),
        }
    }
}

/// Result of the ML model prediction (before interpretation).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiagnosisResult {
    /// Raw prediction probability (0.0 to 1.0)
    pub probability: f64,

    /// Binary prediction (0 = no disease, 1 = disease present)
    pub prediction: u8,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

impl DiagnosisResult {
    /// Create a new diagnosis result.
    #[must_use]
    pub fn new(probability: f64) -> Self {
        let prediction = if probability >= 0.5 { 1 } else { 0 };
        let confidence = if probability >= 0.5 {
            probability
        } else {
            1.0 - probability
        };

        Self {
            probability,
            prediction,
            confidence,
        }
    }

    /// Get the risk level based on probability thresholds.
    #[must_use]
    pub fn risk_level(&self) -> RiskLevel {
        if self.probability < 0.3 {
            RiskLevel::Low
        } else if self.probability < 0.7 {
            RiskLevel::Moderate
        } else {
            RiskLevel::High
        }
    }
}

/// Complete diagnosis record including metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    /// Unique identifier
    pub id: String,

    /// Reference to patient (if available)
    pub patient_id: Option<String>,

    /// The ML prediction result
    pub result: DiagnosisResult,

    /// Risk classification
    pub risk_level: RiskLevel,

    /// Whether this was computed via FHE (encrypted)
    pub encrypted_computation: bool,

    /// Timestamp of diagnosis
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Diagnosis {
    /// Create a new diagnosis from a result.
    #[must_use]
    pub fn new(result: DiagnosisResult, encrypted: bool) -> Self {
        Self {
            id: uuid_v4(),
            patient_id: None,
            risk_level: result.risk_level(),
            result,
            encrypted_computation: encrypted,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a diagnosis with patient reference.
    #[must_use]
    pub fn with_patient(result: DiagnosisResult, patient_id: impl Into<String>, encrypted: bool) -> Self {
        Self {
            id: uuid_v4(),
            patient_id: Some(patient_id.into()),
            risk_level: result.risk_level(),
            result,
            encrypted_computation: encrypted,
            created_at: chrono::Utc::now(),
        }
    }
}

/// Generate a simple UUID v4 (random) using CSPRNG.
///
/// Uses ChaCha20Rng seeded from OS entropy to ensure cryptographic randomness
/// on all platforms. This prevents UUID prediction attacks.
fn uuid_v4() -> String {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand::Rng;
    
    // Use CSPRNG instead of thread_rng() for guaranteed cryptographic security
    let mut rng = ChaCha20Rng::from_entropy();
    let bytes: [u8; 16] = rng.gen();

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        (bytes[6] & 0x0f) | 0x40, bytes[7],
        (bytes[8] & 0x3f) | 0x80, bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_from_probability() {
        assert_eq!(DiagnosisResult::new(0.1).risk_level(), RiskLevel::Low);
        assert_eq!(DiagnosisResult::new(0.5).risk_level(), RiskLevel::Moderate);
        assert_eq!(DiagnosisResult::new(0.9).risk_level(), RiskLevel::High);
    }

    #[test]
    fn test_diagnosis_creation() {
        let result = DiagnosisResult::new(0.75);
        let diagnosis = Diagnosis::new(result, true);

        assert_eq!(diagnosis.risk_level, RiskLevel::High);
        assert!(diagnosis.encrypted_computation);
        assert!(diagnosis.patient_id.is_none());
    }

    #[test]
    fn test_uuid_generation() {
        let id1 = uuid_v4();
        let id2 = uuid_v4();
        assert_ne!(id1, id2);
        assert_eq!(id1.len(), 36); // UUID format with dashes
    }
}
