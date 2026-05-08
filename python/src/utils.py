from pathlib import Path

import numpy as np
import pandas as pd


def load_nhanes_data(cache_path: Path | None = None) -> pd.DataFrame:
    if cache_path is None:
        repo_cache = Path(__file__).resolve().parents[2] / "data" / "nhanes_full.csv"
        cache_path = repo_cache if repo_cache.exists() else Path("/data/nhanes_full.csv")
    if cache_path.exists():
        return pd.read_csv(cache_path)

    import kagglehub

    path = Path(kagglehub.dataset_download("cdc/national-health-and-nutrition-examination-survey"))
    demographic = pd.read_csv(path / "demographic.csv")
    examination = pd.read_csv(path / "examination.csv")
    labs = pd.read_csv(path / "labs.csv")
    questionnaire = pd.read_csv(path / "questionnaire.csv")
    df = demographic.merge(examination, on="SEQN", how="left")
    df = df.merge(labs, on="SEQN", how="left")
    df = df.merge(questionnaire, on="SEQN", how="left")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def quantize_value(value: float, scale_factor: int) -> int:
    return int(round(float(value) * scale_factor))


def generate_sigmoid_lut(input_bits: int = 8, output_bits: int = 12, input_range: float = 8.0) -> list[int]:
    size = 2**input_bits
    output_scale = 2**output_bits
    xs = np.linspace(-input_range, input_range, size)
    ys = 1.0 / (1.0 + np.exp(-xs))
    return [int(round(y * output_scale)) for y in ys]
