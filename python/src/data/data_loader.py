from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class NHANESData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


FEATURE_NAMES = [
    "age",
    "sex",
    "race",
    "bmi",
    "waist",
    "sbp",
    "hypertension",
    "total_chol",
    "hdl",
    "hba1c",
    "serum_glucose",
    "diabetes",
    "egfr",
    "urine_albumin",
    "ever_smoker",
]

FEATURE_SCHEMA = [
    {"name": "age", "source": "RIDAGEYR", "unit": "years", "role": "Age in years"},
    {"name": "sex", "source": "RIAGENDR", "unit": "binary", "role": "Sex, 1=male and 0=female"},
    {"name": "race", "source": "RIDRETH1", "unit": "category", "role": "Race/ethnicity recode"},
    {"name": "bmi", "source": "BMXBMI", "unit": "kg/m2", "role": "Body mass index"},
    {"name": "waist", "source": "BMXWAIST", "unit": "cm", "role": "Waist circumference"},
    {"name": "sbp", "source": "mean(BPXSY1,BPXSY2,BPXSY3)", "unit": "mmHg", "role": "Mean systolic blood pressure"},
    {"name": "hypertension", "source": "BPQ020", "unit": "binary", "role": "Doctor diagnosed hypertension"},
    {"name": "total_chol", "source": "LBXTC", "unit": "mg/dL", "role": "Total cholesterol"},
    {"name": "hdl", "source": "LBDHDD", "unit": "mg/dL", "role": "HDL cholesterol"},
    {"name": "hba1c", "source": "LBXGH", "unit": "percent", "role": "Glycohemoglobin HbA1c"},
    {"name": "serum_glucose", "source": "LBXSGL", "unit": "mg/dL", "role": "Serum glucose"},
    {"name": "diabetes", "source": "DIQ010", "unit": "binary", "role": "Doctor diagnosed diabetes"},
    {"name": "egfr", "source": "LBXSCR+RIDAGEYR+RIAGENDR", "unit": "mL/min/1.73m2", "role": "CKD-EPI 2021 estimated GFR"},
    {"name": "urine_albumin", "source": "URXUMA", "unit": "mg/L", "role": "Urine albumin"},
    {"name": "ever_smoker", "source": "SMQ020", "unit": "binary", "role": "Smoked 100+ cigarettes ever"},
]


def recode_yes_no(series: pd.Series) -> pd.Series:
    return series.map({1: 1.0, 2: 0.0, 3: np.nan, 7: np.nan, 9: np.nan})


def compute_cvd_target(df: pd.DataFrame) -> pd.Series:
    cvd_cols = ["MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
    available = [col for col in cvd_cols if col in df.columns]
    if not available:
        raise ValueError("No cardiovascular condition columns found")

    known = df[available].notna().any(axis=1)
    target = pd.Series(np.nan, index=df.index, dtype=float)
    target.loc[known] = 0.0
    for col in available:
        target.loc[df[col] == 1] = 1.0
    return target


def compute_egfr_2021(age: pd.Series, sex_male: pd.Series, creatinine: pd.Series) -> pd.Series:
    female = sex_male == 0
    kappa = pd.Series(np.where(female, 0.7, 0.9), index=age.index)
    alpha = pd.Series(np.where(female, -0.241, -0.302), index=age.index)
    ratio = creatinine / kappa
    egfr = (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** -1.2
        * (0.9938 ** age)
        * np.where(female, 1.012, 1.0)
    )
    return pd.Series(egfr, index=age.index)


def build_model_frame(
    raw: pd.DataFrame,
    adult_only: bool = True,
    impute: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df["CVD"] = compute_cvd_target(raw)
    df["age"] = raw["RIDAGEYR"]
    sex_male = raw["RIAGENDR"].map({1: 1.0, 2: 0.0})
    df["sex"] = sex_male
    df["race"] = raw["RIDRETH1"]
    df["bmi"] = raw["BMXBMI"]
    df["waist"] = raw["BMXWAIST"]
    bp_cols = [col for col in ["BPXSY1", "BPXSY2", "BPXSY3"] if col in raw.columns]
    df["sbp"] = raw[bp_cols].mean(axis=1)
    df["hypertension"] = recode_yes_no(raw["BPQ020"])
    df["total_chol"] = raw["LBXTC"]
    df["hdl"] = raw["LBDHDD"]
    df["hba1c"] = raw["LBXGH"]
    df["serum_glucose"] = raw["LBXSGL"]
    df["diabetes"] = recode_yes_no(raw["DIQ010"])
    df["egfr"] = compute_egfr_2021(raw["RIDAGEYR"], sex_male, raw["LBXSCR"])
    df["urine_albumin"] = raw["URXUMA"]
    df["ever_smoker"] = recode_yes_no(raw["SMQ020"])

    if adult_only:
        df = df[df["age"] >= 20]
    df = df.dropna(subset=["CVD"])

    if impute:
        for col in FEATURE_NAMES:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    return df[FEATURE_NAMES + ["CVD"]]


def load_nhanes(data_dir: Optional[Path] = None) -> pd.DataFrame:
    import hashlib

    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "nhanes_cardio.csv"
    hash_path = data_dir / "nhanes_cardio.csv.sha256"

    if cache_path.exists() and hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        file_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        if file_hash == stored_hash:
            return pd.read_csv(cache_path)
        cache_path.unlink()
        hash_path.unlink()
    if cache_path.exists():
        return pd.read_csv(cache_path)

    try:
        import kagglehub

        path = Path(kagglehub.dataset_download("cdc/national-health-and-nutrition-examination-survey"))
        demographic = pd.read_csv(path / "demographic.csv")
        examination = pd.read_csv(path / "examination.csv")
        labs = pd.read_csv(path / "labs.csv")
        questionnaire = pd.read_csv(path / "questionnaire.csv")

        df = demographic[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1"]].copy()
        exam_cols = ["SEQN", "BMXBMI", "BMXWAIST", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1"]
        lab_cols = ["SEQN", "LBXGH", "LBXSCR", "LBXSUA", "LBXTC", "LBDHDD", "LBXSGL", "URXUMA"]
        quest_cols = ["SEQN", "BPQ020", "DIQ010", "SMQ020", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
        df = df.merge(examination[[c for c in exam_cols if c in examination.columns]], on="SEQN", how="left")
        df = df.merge(labs[[c for c in lab_cols if c in labs.columns]], on="SEQN", how="left")
        df = df.merge(questionnaire[[c for c in quest_cols if c in questionnaire.columns]], on="SEQN", how="left")
        df.to_csv(cache_path, index=False)
        hash_path.write_text(hashlib.sha256(cache_path.read_bytes()).hexdigest())
        return df
    except ImportError as exc:
        raise RuntimeError("kagglehub not installed") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load NHANES dataset: {exc}") from exc


def preprocess_nhanes(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> NHANESData:
    df_model = build_model_frame(df, adult_only=True, impute=True)
    X = df_model[FEATURE_NAMES].values.astype(np.float64)
    y = df_model["CVD"].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    return NHANESData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )


def get_preprocessed_data(
    data_dir: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> NHANESData:
    return preprocess_nhanes(load_nhanes(data_dir), test_size, random_state)
