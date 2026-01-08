//! Patient data types for cardiovascular risk prediction.
//!
//! Based on NHANES (CDC National Health and Nutrition Examination Survey) features.

use serde::{Deserialize, Serialize};

/// Raw patient data input from the TUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientData {
    /// Patient identifier (local only, never transmitted)
    pub id: Option<String>,

    /// Clinical features for prediction
    pub features: PatientFeatures,

    /// Timestamp of data entry
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Clinical features from NHANES dataset for CVD risk prediction.
///
/// 9 features matching the calibrated model (calibrated_model.json):
/// RIDAGEYR, BPQ020, BPXSY1, SMQ020, LBDHDD, LBXSCR, BMXWAIST, DIQ010, LBXGH
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatientFeatures {
    /// Age in years (RIDAGEYR, 18-85 typical range)
    pub age: f64,

    /// Hypertension: 0 = no, 1 = yes (BPQ020, doctor diagnosed)
    pub hypertension: f64,

    /// Systolic blood pressure in mmHg (BPXSY1, 66-228 typical)
    pub sys_bp: f64,

    /// Smoking: 0 = no, 1 = yes (SMQ020, smoked 100+ cigarettes ever)
    pub smoking: f64,

    /// HDL cholesterol in mg/dL (LBDHDD, 10-173 typical range)
    pub hdl_chol: f64,

    /// Serum creatinine in mg/dL (LBXSCR, 0.3-17.4 typical range)
    pub creatinine: f64,

    /// Waist circumference in cm (BMXWAIST, 40-178 typical range)
    pub waist_circ: f64,

    /// Diabetes: 0 = no, 1 = yes (DIQ010, doctor diagnosed)
    pub diabetes: f64,

    /// Glycohemoglobin HbA1c in % (LBXGH, 3.5-17.5 typical range)
    pub hba1c: f64,
}

impl PatientFeatures {
    /// Convert features to a vector for ML inference.
    /// Order matches calibrated model: RIDAGEYR, BPQ020, BPXSY1, SMQ020, LBDHDD, LBXSCR, BMXWAIST, DIQ010, LBXGH
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.age,
            self.hypertension,
            self.sys_bp,
            self.smoking,
            self.hdl_chol,
            self.creatinine,
            self.waist_circ,
            self.diabetes,
            self.hba1c,
        ]
    }

    /// Create features from a vector.
    ///
    /// # Errors
    /// Returns error if vector length is not 9.
    pub fn from_vec(v: &[f64]) -> Result<Self, String> {
        if v.len() != 9 {
            return Err(format!("Expected 9 features, got {}", v.len()));
        }

        Ok(Self {
            age: v[0],
            hypertension: v[1],
            sys_bp: v[2],
            smoking: v[3],
            hdl_chol: v[4],
            creatinine: v[5],
            waist_circ: v[6],
            diabetes: v[7],
            hba1c: v[8],
        })
    }

    /// Validate that all features are within expected ranges.
    ///
    /// # Errors
    /// Returns validation errors as a vector of strings.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if !(18.0..=120.0).contains(&self.age) {
            errors.push(format!("Age {} out of range [18, 120]", self.age));
        }
        if self.hypertension != 0.0 && self.hypertension != 1.0 {
            errors.push(format!("Hypertension {} must be 0 or 1", self.hypertension));
        }
        if !(50.0..=250.0).contains(&self.sys_bp) {
            errors.push(format!(
                "Systolic BP {} out of range [50, 250]",
                self.sys_bp
            ));
        }
        if self.smoking != 0.0 && self.smoking != 1.0 {
            errors.push(format!("Smoking {} must be 0 or 1", self.smoking));
        }
        if !(5.0..=200.0).contains(&self.hdl_chol) {
            errors.push(format!(
                "HDL cholesterol {} out of range [5, 200]",
                self.hdl_chol
            ));
        }
        if !(0.1..=20.0).contains(&self.creatinine) {
            errors.push(format!(
                "Creatinine {} out of range [0.1, 20]",
                self.creatinine
            ));
        }
        if !(30.0..=200.0).contains(&self.waist_circ) {
            errors.push(format!(
                "Waist circumference {} out of range [30, 200]",
                self.waist_circ
            ));
        }
        if self.diabetes != 0.0 && self.diabetes != 1.0 {
            errors.push(format!("Diabetes {} must be 0 or 1", self.diabetes));
        }
        if !(3.0..=20.0).contains(&self.hba1c) {
            errors.push(format!("HbA1c {} out of range [3, 20]", self.hba1c));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl PatientData {
    /// Create new patient data with the given features.
    #[must_use]
    pub fn new(features: PatientFeatures) -> Self {
        Self {
            id: None,
            features,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create new patient data with an ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>, features: PatientFeatures) -> Self {
        Self {
            id: Some(id.into()),
            features,
            created_at: chrono::Utc::now(),
        }
    }
}

/// Feature names matching the calibrated model (NHANES codes).
/// Order: RIDAGEYR, BPQ020, BPXSY1, SMQ020, LBDHDD, LBXSCR, BMXWAIST, DIQ010, LBXGH
#[allow(dead_code)]
pub const FEATURE_NAMES: [&str; 9] = [
    "age",
    "hypertension",
    "sys_bp",
    "smoking",
    "hdl_chol",
    "creatinine",
    "waist_circ",
    "diabetes",
    "hba1c",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_to_vec() {
        let features = PatientFeatures {
            age: 55.0,
            hypertension: 1.0,
            sys_bp: 138.0,
            smoking: 0.0,
            hdl_chol: 50.0,
            creatinine: 1.0,
            waist_circ: 98.0,
            diabetes: 0.0,
            hba1c: 5.7,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), 9);
        assert!((vec[0] - 55.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_features_from_vec() {
        let v = vec![55.0, 1.0, 138.0, 0.0, 50.0, 1.0, 98.0, 0.0, 5.7];
        let features = PatientFeatures::from_vec(&v).expect("Should parse");
        assert!((features.age - 55.0).abs() < f64::EPSILON);
        assert!((features.hdl_chol - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validation() {
        let valid = PatientFeatures {
            age: 55.0,
            hypertension: 1.0,
            sys_bp: 138.0,
            smoking: 0.0,
            hdl_chol: 50.0,
            creatinine: 1.0,
            waist_circ: 98.0,
            diabetes: 0.0,
            hba1c: 5.7,
        };
        assert!(valid.validate().is_ok());

        let invalid = PatientFeatures {
            age: 10.0,         // invalid (< 18)
            hypertension: 2.0, // invalid
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}
