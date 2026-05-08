use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,

    Moderate,

    High,
}

impl RiskLevel {
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Low => "Low risk - No significant indicators",
            Self::Moderate => "Moderate risk - Follow-up recommended",
            Self::High => "High risk - Immediate consultation advised",
        }
    }

    #[must_use]
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Self::Low => (16, 185, 129),
            Self::Moderate => (251, 191, 36),
            Self::High => (244, 63, 94),
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiagnosisResult {
    pub probability: f64,

    pub prediction: u8,

    #[serde(default)]
    pub screening_positive: bool,

    #[serde(default = "default_clinical_threshold")]
    pub threshold_used: f64,

    pub confidence: f64,
}

fn default_clinical_threshold() -> f64 {
    0.5
}

impl DiagnosisResult {
    #[must_use]
    pub fn new(probability: f64) -> Self {
        Self::with_threshold(probability, default_clinical_threshold())
    }

    #[must_use]
    pub fn with_threshold(probability: f64, threshold: f64) -> Self {
        let screening_positive = probability >= threshold;
        let prediction = u8::from(screening_positive);
        let confidence = if screening_positive {
            probability
        } else {
            1.0 - probability
        };

        Self {
            probability,
            prediction,
            screening_positive,
            threshold_used: threshold,
            confidence,
        }
    }

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    pub id: String,

    pub patient_id: Option<String>,

    pub result: DiagnosisResult,

    pub risk_level: RiskLevel,

    pub encrypted_computation: bool,

    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Diagnosis {
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

    #[must_use]
    pub fn with_patient(
        result: DiagnosisResult,
        patient_id: impl Into<String>,
        encrypted: bool,
    ) -> Self {
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

fn uuid_v4() -> String {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

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
    fn test_screening_threshold_is_not_hardcoded_to_point_five() {
        let result = DiagnosisResult::with_threshold(0.2, 0.1);
        assert!(result.screening_positive);
        assert_eq!(result.prediction, 1);
        assert!((result.threshold_used - 0.1).abs() < f64::EPSILON);

        let result = DiagnosisResult::with_threshold(0.2, 0.3);
        assert!(!result.screening_positive);
        assert_eq!(result.prediction, 0);
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
        assert_eq!(id1.len(), 36);
    }
}
