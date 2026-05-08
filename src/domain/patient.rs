use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientData {
    pub id: Option<String>,

    pub features: PatientFeatures,

    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatientFeatures {
    pub age: f64,

    pub hypertension: f64,

    pub sys_bp: f64,

    pub smoking: f64,

    pub hdl_chol: f64,

    pub creatinine: f64,

    pub waist_circ: f64,

    pub diabetes: f64,

    pub hba1c: f64,

    #[serde(default)]
    pub sex: f64,

    #[serde(default)]
    pub race: f64,

    #[serde(default)]
    pub bmi: f64,

    #[serde(default)]
    pub total_chol: f64,

    #[serde(default)]
    pub serum_glucose: f64,

    #[serde(default)]
    pub egfr: f64,

    #[serde(default)]
    pub urine_albumin: f64,
}

impl PatientFeatures {
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.age,
            self.sex,
            self.race,
            self.bmi,
            self.waist_circ,
            self.sys_bp,
            self.hypertension,
            self.total_chol,
            self.hdl_chol,
            self.hba1c,
            self.serum_glucose,
            self.diabetes,
            self.egfr,
            self.urine_albumin,
            self.smoking,
        ]
    }

    pub fn to_model_vec(&self, feature_names: &[String]) -> Result<Vec<f64>, String> {
        feature_names
            .iter()
            .map(|name| self.value_for_model_feature(name))
            .collect()
    }

    fn value_for_model_feature(&self, name: &str) -> Result<f64, String> {
        match name {
            "RIDAGEYR" | "age" => Ok(self.age),
            "RIAGENDR" | "sex" => Ok(self.sex),
            "RIDRETH1" | "race" => Ok(self.race),
            "BMXBMI" | "bmi" => Ok(self.bmi),
            "BMXWAIST" | "waist_circ" | "waist" => Ok(self.waist_circ),
            "BPXSY1" | "sys_bp" | "sbp" => Ok(self.sys_bp),
            "BPQ020" | "hypertension" => Ok(self.hypertension),
            "LBXTC" | "total_chol" => Ok(self.total_chol),
            "LBDHDD" | "hdl_chol" | "hdl" => Ok(self.hdl_chol),
            "LBXGH" | "hba1c" => Ok(self.hba1c),
            "LBXSGL" | "serum_glucose" => Ok(self.serum_glucose),
            "DIQ010" | "diabetes" => Ok(self.diabetes),
            "LBXSCR" | "creatinine" => Ok(self.creatinine),
            "egfr" => Ok(self.egfr),
            "URXUMA" | "urine_albumin" => Ok(self.urine_albumin),
            "SMQ020" | "smoking" | "ever_smoker" => Ok(self.smoking),
            other => Err(format!("Unsupported model feature: {other}")),
        }
    }

    pub fn from_vec(v: &[f64]) -> Result<Self, String> {
        if v.len() != 15 {
            return Err(format!("Expected 15 features, got {}", v.len()));
        }

        Ok(Self {
            age: v[0],
            sex: v[1],
            race: v[2],
            bmi: v[3],
            waist_circ: v[4],
            sys_bp: v[5],
            hypertension: v[6],
            total_chol: v[7],
            hdl_chol: v[8],
            hba1c: v[9],
            serum_glucose: v[10],
            diabetes: v[11],
            creatinine: 0.0,
            egfr: v[12],
            urine_albumin: v[13],
            smoking: v[14],
        })
    }

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
        if self.sex != 0.0 && self.sex != 1.0 {
            errors.push(format!("Sex {} must be 0 or 1", self.sex));
        }
        if self.race != 0.0 && !(1.0..=10.0).contains(&self.race) {
            errors.push(format!(
                "Race {} out of expected NHANES category range",
                self.race
            ));
        }
        if self.bmi != 0.0 && !(10.0..=80.0).contains(&self.bmi) {
            errors.push(format!("BMI {} out of range [10, 80]", self.bmi));
        }
        if self.total_chol != 0.0 && !(50.0..=500.0).contains(&self.total_chol) {
            errors.push(format!(
                "Total cholesterol {} out of range [50, 500]",
                self.total_chol
            ));
        }
        if self.serum_glucose != 0.0 && !(30.0..=700.0).contains(&self.serum_glucose) {
            errors.push(format!(
                "Serum glucose {} out of range [30, 700]",
                self.serum_glucose
            ));
        }
        if self.egfr != 0.0 && !(1.0..=200.0).contains(&self.egfr) {
            errors.push(format!("eGFR {} out of range [1, 200]", self.egfr));
        }
        if self.urine_albumin != 0.0 && !(0.0..=5000.0).contains(&self.urine_albumin) {
            errors.push(format!(
                "Urine albumin {} out of range [0, 5000]",
                self.urine_albumin
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn validate_for_model(&self, feature_names: &[String]) -> Result<(), Vec<String>> {
        let mut errors = self.validate().err().unwrap_or_default();

        let requires = |name: &str| feature_names.iter().any(|feature| feature == name);

        if requires("race") && self.race == 0.0 {
            errors.push("Race is required for the loaded model".to_string());
        }
        if requires("bmi") && self.bmi == 0.0 {
            errors.push("BMI is required for the loaded model".to_string());
        }
        if requires("total_chol") && self.total_chol == 0.0 {
            errors.push("Total cholesterol is required for the loaded model".to_string());
        }
        if requires("serum_glucose") && self.serum_glucose == 0.0 {
            errors.push("Serum glucose is required for the loaded model".to_string());
        }
        if requires("egfr") && self.egfr == 0.0 {
            errors.push("eGFR is required for the loaded model".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl PatientData {
    #[must_use]
    pub fn new(features: PatientFeatures) -> Self {
        Self {
            id: None,
            features,
            created_at: chrono::Utc::now(),
        }
    }

    #[must_use]
    pub fn with_id(id: impl Into<String>, features: PatientFeatures) -> Self {
        Self {
            id: Some(id.into()),
            features,
            created_at: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_to_vec() {
        let features = PatientFeatures {
            age: 55.0,
            sex: 1.0,
            race: 3.0,
            bmi: 30.5,
            hypertension: 1.0,
            sys_bp: 138.0,
            smoking: 0.0,
            hdl_chol: 50.0,
            waist_circ: 98.0,
            diabetes: 0.0,
            hba1c: 5.7,
            total_chol: 205.0,
            serum_glucose: 108.0,
            egfr: 78.0,
            urine_albumin: 24.0,
            creatinine: 1.0,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), 15);
        assert!((vec[0] - 55.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_features_from_vec() {
        let v = vec![
            55.0, 1.0, 3.0, 30.5, 98.0, 138.0, 1.0, 205.0, 50.0, 5.7, 108.0, 0.0, 78.0, 24.0, 0.0,
        ];
        let features = PatientFeatures::from_vec(&v).expect("Should parse");
        assert!((features.age - 55.0).abs() < f64::EPSILON);
        assert!((features.hdl_chol - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validation() {
        let valid = PatientFeatures {
            age: 55.0,
            sex: 1.0,
            race: 3.0,
            bmi: 30.5,
            hypertension: 1.0,
            sys_bp: 138.0,
            smoking: 0.0,
            hdl_chol: 50.0,
            creatinine: 1.0,
            waist_circ: 98.0,
            diabetes: 0.0,
            hba1c: 5.7,
            total_chol: 205.0,
            serum_glucose: 108.0,
            egfr: 78.0,
            urine_albumin: 24.0,
        };
        assert!(valid.validate().is_ok());

        let invalid = PatientFeatures {
            age: 10.0,
            hypertension: 2.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}
