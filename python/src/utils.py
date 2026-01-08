"""Shared utilities for Pulsecure ML pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd

def load_nhanes_data(cache_path: Path = Path("/data/nhanes_full.csv")) -> pd.DataFrame:
    """Load NHANES dataset with CVD target."""
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        import kagglehub
        path = Path(kagglehub.dataset_download(
            "cdc/national-health-and-nutrition-examination-survey"
        ))
        demographic = pd.read_csv(path / "demographic.csv")
        examination = pd.read_csv(path / "examination.csv")
        labs = pd.read_csv(path / "labs.csv")
        questionnaire = pd.read_csv(path / "questionnaire.csv")
        df = demographic.merge(examination, on="SEQN", how="left")
        df = df.merge(labs, on="SEQN", how="left")
        df = df.merge(questionnaire, on="SEQN", how="left")
        if cache_path.parent.exists():
            df.to_csv(cache_path, index=False)
    
    # CVD target
    cvd_cols = ["MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
    df["CVD"] = 0
    for col in cvd_cols:
        if col in df.columns:
            df.loc[df[col] == 1, "CVD"] = 1
    
    return df

def quantize_value(value: float, scale_factor: int) -> int:
    """Quantize float to fixed-point integer."""
    return int(round(value * scale_factor))

def generate_sigmoid_lut(
    input_bits: int = 8,
    output_bits: int = 12,
    input_range: float = 8.0
) -> list[int]:
    """Generate LUT for sigmoid function (FHE programmable bootstrapping)."""
    lut_size = 2 ** input_bits
    output_scale = 2 ** output_bits
    
    lut = []
    for i in range(lut_size):
        logit = (i / (lut_size - 1)) * 2 * input_range - input_range
        prob = 1.0 / (1.0 + np.exp(-logit))
        lut.append(int(round(prob * output_scale)))
    
    return lut
