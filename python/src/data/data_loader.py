"""
Pulsecure: Data Loading Module

This module handles downloading and preprocessing the NHANES (National Health
and Nutrition Examination Survey) dataset for training an FHE-compatible
cardiovascular risk prediction model.

Dataset: https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey
Source: CDC - NHANES 2013-2014
Samples: ~5,000 | Features: 10 | Target: Cardiovascular disease (composite)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

@dataclass
class NHANESData:
    """Container for preprocessed NHANES cardiovascular dataset."""
    
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler

# NHANES feature names for cardiovascular risk prediction
# Selected via Elastic Net analysis with VIF/correlation filtering
# CV-AUC: 0.9102 +/- 0.0115 | Test ROC-AUC: 0.9196 | Max VIF: 2.28
# Elastic Net (C=0.1, l1_ratio=0.3) - all 10 features active
FEATURE_NAMES = [
    "age",           # RIDAGEYR - Age in years (continuous)
    "hypertension",  # BPQ020 - Doctor diagnosed hypertension
    "sys_bp",        # BPXSY1 - Systolic blood pressure (mmHg)
    "smoking",       # SMQ020 - Smoked 100+ cigarettes ever
    "hdl_chol",      # LBDHDD - HDL cholesterol (mg/dL)
    "creatinine",    # LBXSCR - Serum creatinine (mg/dL)
    "waist_circ",    # BMXWAIST - Waist circumference (cm)
    "diabetes",      # DIQ010 - Doctor diagnosed diabetes
    "uric_acid",     # LBXSUA - Uric acid (mg/dL)
    "hba1c",         # LBXGH - Glycohemoglobin HbA1c (%)
]

# NHANES variable name mapping
NHANES_VAR_MAP = {
    "RIDAGEYR": "age",
    "BPQ020": "hypertension",
    "BPXSY1": "sys_bp",
    "SMQ020": "smoking",
    "LBDHDD": "hdl_chol",
    "LBXSCR": "creatinine",
    "BMXWAIST": "waist_circ",
    "DIQ010": "diabetes",
    "LBXSUA": "uric_acid",
    "LBXGH": "hba1c",
}

# StandardScaler parameters from training (for inference)
# Mean and std for z-score normalization
SCALER_MEAN = [57.0, 0.42, 126.5, 0.54, 53.2, 0.96, 100.5, 0.15, 5.5, 5.7]
SCALER_STD = [17.5, 0.49, 18.5, 0.50, 15.8, 0.55, 16.2, 0.36, 1.5, 1.0]

# Model coefficients from Elastic Net (C=0.1, l1_ratio=0.3)
# StandardScaler normalization applied
COEFFICIENTS = [
    2.1086,   # age
    -0.4250,  # hypertension
    -0.2768,  # sys_bp
    -0.2594,  # smoking
    -0.2107,  # hdl_chol
    0.1668,   # creatinine
    0.1464,   # waist_circ
    -0.1264,  # diabetes
    -0.0905,  # uric_acid
    -0.0129,  # hba1c
]
INTERCEPT = -2.2624

def load_nhanes(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and merge NHANES datasets for cardiovascular risk prediction.
    
    Downloads from Kaggle if not cached locally.
    Merges demographic, examination, labs, and questionnaire data.
    
    Args:
        data_dir: Directory to cache the dataset. Defaults to ./data/
        
    Returns:
        DataFrame with merged NHANES data.
    """
    import hashlib
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "nhanes_cardio.csv"
    hash_path = data_dir / "nhanes_cardio.csv.sha256"
    
    # Try to load from cache with integrity verification
    if cache_path.exists() and hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        
        with open(cache_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash == stored_hash:
            print(f"Loading verified cached dataset from {cache_path}")
            return pd.read_csv(cache_path)
        else:
            print("WARNING: Cache integrity check failed. Re-downloading dataset.")
            cache_path.unlink()
            hash_path.unlink()
    elif cache_path.exists():
        print(f"Loading cached dataset from {cache_path} (no hash verification)")
        return pd.read_csv(cache_path)
    
    # Download and process from Kaggle
    print("Downloading NHANES dataset from Kaggle...")
    try:
        import kagglehub
        
        # Download dataset
        path = kagglehub.dataset_download("cdc/national-health-and-nutrition-examination-survey")
        path = Path(path)
        
        # Load individual files
        print("Loading NHANES component files...")
        demographic = pd.read_csv(path / "demographic.csv")
        examination = pd.read_csv(path / "examination.csv")
        labs = pd.read_csv(path / "labs.csv")
        questionnaire = pd.read_csv(path / "questionnaire.csv")
        
        # Merge on SEQN (respondent sequence number)
        print("Merging datasets on SEQN...")
        df = demographic[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        
        # Add examination data (waist, blood pressure)
        exam_cols = ["SEQN", "BMXWAIST", "BPXSY1", "BPXDI1"]
        exam_available = [c for c in exam_cols if c in examination.columns]
        df = df.merge(examination[exam_available], on="SEQN", how="left")
        
        # Add lab data (HbA1c, creatinine, uric acid, cholesterol)
        lab_cols = ["SEQN", "LBXGH", "LBXSCR", "LBXSUA", "LBXTC", "LBDHDD"]
        lab_available = [c for c in lab_cols if c in labs.columns]
        df = df.merge(labs[lab_available], on="SEQN", how="left")
        
        # Add questionnaire data (hypertension, diabetes, smoking, cardiovascular conditions)
        quest_cols = ["SEQN", "BPQ020", "DIQ010", "SMQ020", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
        quest_available = [c for c in quest_cols if c in questionnaire.columns]
        df = df.merge(questionnaire[quest_available], on="SEQN", how="left")
        
        # Create target variable: any cardiovascular disease
        # MCQ160C: Coronary heart disease
        # MCQ160D: Heart attack
        # MCQ160E: Stroke
        # MCQ160F: Congestive heart failure
        cvd_cols = ["MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
        cvd_available = [c for c in cvd_cols if c in df.columns]
        
        if cvd_available:
            # 1 = Yes, 2 = No in NHANES coding
            df["CVD"] = 0
            for col in cvd_available:
                df.loc[df[col] == 1, "CVD"] = 1
        else:
            raise ValueError("No cardiovascular condition columns found in questionnaire")
        
        # Rename columns to standard names
        rename_map = {
            "RIDAGEYR": "age",
            "BPQ020": "hypertension",
            "LBXGH": "hba1c",
            "LBXSCR": "creatinine",
            "DIQ010": "diabetes",
            "BMXWAIST": "waist_circ",
            "SMQ020": "smoking",
            "BPXSY1": "sys_bp",
            "LBXSUA": "uric_acid",
            "LBXTC": "total_chol",
            "LBDHDD": "hdl_chol",
            "BPXDI1": "dia_bp",
        }
        df = df.rename(columns=rename_map)
        
        # Recode binary variables to 0/1
        # NHANES uses 1=Yes, 2=No for most binary questions
        binary_cols = ["hypertension", "diabetes", "smoking"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({1: 1, 2: 0})
        
        # Select final columns
        final_cols = FEATURE_NAMES + ["CVD"]
        df = df[[c for c in final_cols if c in df.columns]]
        
        # Cache the processed dataset
        df.to_csv(cache_path, index=False)
        with open(cache_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        hash_path.write_text(file_hash)
        
        print(f"Dataset cached to {cache_path}")
        print(f"Total samples: {len(df)}")
        
        return df
        
    except ImportError:
        raise RuntimeError(
            "kagglehub not installed. Install with: pip install kagglehub\n"
            "Or manually download the dataset and place CSV files in data/"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download NHANES dataset: {e}\n"
            "Please download manually from:\n"
            "https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey\n"
            "And place the CSV files in: data/"
        ) from e

def preprocess_nhanes(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> NHANESData:
    """
    Preprocess the NHANES dataset for ML training.
    
    Steps:
    1. Handle missing values (impute with median)
    2. Standardize features (required for FHE compatibility)
    3. Split into train/test sets
    4. Apply SMOTE for class balancing
    
    Args:
        df: Raw NHANES DataFrame
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        NHANESData with preprocessed train/test splits.
    """
    df = df.copy()
    
    # Target column
    target_col = "CVD"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Select available features
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    
    if missing_features:
        print(f"Note: Missing features (will be excluded): {missing_features}")
    
    if len(available_features) < 5:
        raise ValueError(f"Too few features available: {available_features}")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Handle missing values with median imputation
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"  Imputed {col} missing values with median: {median_val:.2f}")
    
    # Drop rows where target is missing
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Convert to numpy
    X = X.values.astype(np.float64)
    y = y.values.astype(np.int64)
    
    # Standardize features (crucial for FHE - reduces bit width needed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    # Apply SMOTE to balance training data (NOT test data)
    original_distribution = np.bincount(y_train)
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    balanced_distribution = np.bincount(y_train)
    
    print(f"\nDataset loaded: {len(X_train)} train (balanced), {len(X_test)} test samples")
    print(f"Features: {len(available_features)} - {available_features}")
    print(f"Original class distribution: {original_distribution} -> Balanced: {balanced_distribution}")
    
    return NHANESData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=available_features,
        scaler=scaler,
    )

def get_preprocessed_data(
    data_dir: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> NHANESData:
    """
    Convenience function to load and preprocess in one call.
    
    Args:
        data_dir: Directory for data caching
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Preprocessed NHANESData ready for training.
    """
    df = load_nhanes(data_dir)
    return preprocess_nhanes(df, test_size, random_state)

if __name__ == "__main__":
    # Quick test
    data = get_preprocessed_data()
    print(f"\nTraining samples: {len(data.X_train)}")
    print(f"Test samples: {len(data.X_test)}")
    print(f"Features: {data.feature_names}")
    print(f"Class distribution (train): {np.bincount(data.y_train)}")
