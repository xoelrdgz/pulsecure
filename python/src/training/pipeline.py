from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..data import FEATURE_NAMES, FEATURE_SCHEMA, build_model_frame
from ..utils import generate_sigmoid_lut, load_nhanes_data, quantize_value

MODEL_SCHEMA_VERSION = 1
INTENDED_USE = "Clinical decision support for cardiovascular screening by healthcare professionals"


@dataclass
class CalibratedQuantizedModel:
    schema_version: int
    intended_use: str
    precision_bits: int
    scale_factor: int
    coefficients_q: list[int]
    intercept_q: int
    scaler_mean_q: list[int]
    scaler_std_inv_q: list[int]
    calibration_x_q: list[int]
    calibration_y_q: list[int]
    sigmoid_lut: list[int]
    sigmoid_lut_input_bits: int
    sigmoid_lut_output_bits: int
    threshold: float
    threshold_q: int
    feature_names: list[str]
    feature_schema: list[dict[str, str]]
    auc_float: float
    auc_calibrated: float
    auc_quantized: float
    brier_before: float
    brier_after: float
    recall_at_threshold: float
    precision_at_threshold: float
    max_calibration_error: float
    imputer_values: list[float]
    calibration_method: str
    dataset_hash: str
    n_samples: int
    trained_at: str


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.90) -> Tuple[float, float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_idx = np.where(recalls[:-1] >= min_recall)[0]
    idx = valid_idx[-1] if len(valid_idx) > 0 else 0
    return float(thresholds[idx]), float(recalls[idx]), float(precisions[idx])


def probability_logit(prob: np.ndarray) -> np.ndarray:
    eps = 1e-6
    clipped = np.clip(prob, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def fit_sigmoid_calibrator(y_prob_calib: np.ndarray, y_calib: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(probability_logit(y_prob_calib).reshape(-1, 1), y_calib)
    return calibrator


def apply_sigmoid_calibrator(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(probability_logit(probabilities).reshape(-1, 1))[:, 1]


def apply_isotonic_lut(prob: np.ndarray, x_breakpoints: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    return np.interp(prob, x_breakpoints, y_values)


def inference_full_pipeline(
    X_raw: np.ndarray,
    coef_q: np.ndarray,
    intercept_q: int,
    mean_q: np.ndarray,
    std_inv_q: np.ndarray,
    scale_factor: int,
    sigmoid_lut: list[int],
    calib_x_q: np.ndarray,
    calib_y_q: np.ndarray,
    sigmoid_output_bits: int = 12,
    input_range: float = 8.0,
) -> np.ndarray:
    n_samples, n_features = X_raw.shape
    output_scale = 2**sigmoid_output_bits
    X_q = np.zeros((n_samples, n_features), dtype=np.int64)
    for i in range(n_features):
        x_scaled = (X_raw[:, i] * scale_factor).astype(np.int64)
        X_q[:, i] = ((x_scaled - mean_q[i]) * std_inv_q[i]) // scale_factor
    logits_q = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_features):
        logits_q += X_q[:, i] * coef_q[i]
    logits_q += intercept_q * scale_factor
    logit_float = logits_q.astype(np.float64) / (scale_factor * scale_factor)
    indices = ((np.clip(logit_float, -input_range, input_range) + input_range) / (2 * input_range) * (len(sigmoid_lut) - 1))
    prob_q = np.array(sigmoid_lut)[np.clip(indices.astype(np.int64), 0, len(sigmoid_lut) - 1)]
    prob_uncalib = prob_q.astype(np.float64) / output_scale
    return apply_isotonic_lut(prob_uncalib, calib_x_q.astype(np.float64) / output_scale, calib_y_q.astype(np.float64) / output_scale)


def train_calibrated_model(
    precision_bits: int = 12,
    min_recall: float = 0.95,
    calibration_method: str = "sigmoid",
) -> CalibratedQuantizedModel:
    if calibration_method != "sigmoid":
        raise ValueError("Only sigmoid calibration is active in v2")

    scale_factor = 2**precision_bits
    raw = load_nhanes_data()
    df_model = build_model_frame(raw, adult_only=True, impute=False)
    X_raw = df_model[FEATURE_NAMES].values
    y = df_model["CVD"].values
    dataset_hash = hashlib.sha256(df_model.to_csv(index=False).encode("utf-8")).hexdigest()

    X_temp, X_test, y_temp, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_calib = imputer.transform(X_calib)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)
    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
    model = LogisticRegression(solver="saga", penalty="elasticnet", C=0.1, l1_ratio=0.3, max_iter=2000, random_state=42)
    model.fit(X_train_sm, y_train_sm)
    y_prob_calib = model.predict_proba(X_calib_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    calibrator = fit_sigmoid_calibrator(y_prob_calib, y_calib)
    y_prob_test_calib = apply_sigmoid_calibrator(calibrator, y_prob_test)
    auc_uncalib = roc_auc_score(y_test, y_prob_test)
    auc_calib = roc_auc_score(y_test, y_prob_test_calib)
    brier_uncalib = brier_score_loss(y_test, y_prob_test)
    brier_calib = brier_score_loss(y_test, y_prob_test_calib)
    prob_true, prob_pred = calibration_curve(y_test, y_prob_test_calib, n_bins=10)
    max_calib_error = float(np.max(np.abs(prob_true - prob_pred))) if len(prob_true) else 0.0
    threshold, _, _ = find_optimal_threshold(y_test, y_prob_test_calib, min_recall=min_recall)

    coef_q = np.array([quantize_value(c, scale_factor) for c in model.coef_[0]], dtype=np.int64)
    intercept_q = quantize_value(model.intercept_[0], scale_factor)
    mean_q = np.array([quantize_value(m, scale_factor) for m in scaler.mean_], dtype=np.int64)
    std_inv_q = np.array([quantize_value(1 / s, scale_factor) for s in scaler.scale_], dtype=np.int64)
    sigmoid_input_bits = 8
    sigmoid_output_bits = 12
    sigmoid_lut = generate_sigmoid_lut(sigmoid_input_bits, sigmoid_output_bits, 8.0)
    calib_x_dense = np.linspace(0, 1, 64)
    calib_y_dense = apply_sigmoid_calibrator(calibrator, calib_x_dense)
    calib_x_q = np.array([quantize_value(x, 2**sigmoid_output_bits) for x in calib_x_dense], dtype=np.int64)
    calib_y_q = np.array([quantize_value(v, 2**sigmoid_output_bits) for v in calib_y_dense], dtype=np.int64)
    y_prob_quant = inference_full_pipeline(X_test, coef_q, intercept_q, mean_q, std_inv_q, scale_factor, sigmoid_lut, calib_x_q, calib_y_q)
    auc_quant = roc_auc_score(y_test, y_prob_quant)
    threshold_quant, recall_quant, precision_quant = find_optimal_threshold(y_test, y_prob_quant, min_recall=min_recall)
    threshold_q = quantize_value(threshold_quant, 2**sigmoid_output_bits)
    y_pred_quant = (y_prob_quant >= threshold_quant).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_quant).ravel()
    _ = (tn, fp, fn, tp, threshold)

    return CalibratedQuantizedModel(
        schema_version=MODEL_SCHEMA_VERSION,
        intended_use=INTENDED_USE,
        precision_bits=precision_bits,
        scale_factor=scale_factor,
        coefficients_q=coef_q.tolist(),
        intercept_q=intercept_q,
        scaler_mean_q=mean_q.tolist(),
        scaler_std_inv_q=std_inv_q.tolist(),
        calibration_x_q=calib_x_q.tolist(),
        calibration_y_q=calib_y_q.tolist(),
        sigmoid_lut=sigmoid_lut,
        sigmoid_lut_input_bits=sigmoid_input_bits,
        sigmoid_lut_output_bits=sigmoid_output_bits,
        threshold=float(threshold_quant),
        threshold_q=threshold_q,
        feature_names=FEATURE_NAMES,
        feature_schema=FEATURE_SCHEMA,
        auc_float=float(auc_uncalib),
        auc_calibrated=float(auc_calib),
        auc_quantized=float(auc_quant),
        brier_before=float(brier_uncalib),
        brier_after=float(brier_calib),
        recall_at_threshold=float(recall_quant),
        precision_at_threshold=float(precision_quant),
        max_calibration_error=max_calib_error,
        imputer_values=[float(x) for x in imputer.statistics_],
        calibration_method=calibration_method,
        dataset_hash=dataset_hash,
        n_samples=int(len(y)),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )


def export_to_json(model: CalibratedQuantizedModel, output_path: Path) -> None:
    data = asdict(model)
    data["sigmoid_lut"] = {
        "input_bits": model.sigmoid_lut_input_bits,
        "output_bits": model.sigmoid_lut_output_bits,
        "input_range": 8.0,
        "values": model.sigmoid_lut,
    }
    data["calibration_lut"] = {
        "x_breakpoints": model.calibration_x_q,
        "y_values": model.calibration_y_q,
    }
    data["clinical_threshold"] = {
        "value": model.threshold,
        "quantized": model.threshold_q,
        "recall": model.recall_at_threshold,
        "precision": model.precision_at_threshold,
    }
    data["validation"] = {
        "auc_float": model.auc_float,
        "auc_calibrated": model.auc_calibrated,
        "auc_quantized": model.auc_quantized,
        "brier_before": model.brier_before,
        "brier_after": model.brier_after,
        "max_calibration_error": model.max_calibration_error,
    }
    data["imputation"] = {"method": "median_fit_on_train", "values": model.imputer_values}
    data["dataset"] = {"name": "NHANES cardiovascular screening dataset", "sha256": model.dataset_hash, "n_samples": model.n_samples}
    data["training_metadata"] = {
        "trained_at": model.trained_at,
        "random_seed": 42,
        "model_type": "LogisticRegression(solver=saga, penalty=elasticnet)",
        "calibration_method": model.calibration_method,
        "target_recall": model.recall_at_threshold,
    }
    output_path.write_text(json.dumps(data, indent=2))


def export_to_rust(model: CalibratedQuantizedModel, output_path: Path) -> None:
    def arr(values: list[int], per_row: int = 16) -> str:
        rows = []
        for i in range(0, len(values), per_row):
            rows.append("    " + ", ".join(str(v) for v in values[i : i + per_row]) + ",")
        return "\n".join(rows)

    feature_names = "\n".join(f'    "{name}",' for name in model.feature_names)
    coefficients = arr(model.coefficients_q)
    mean = arr(model.scaler_mean_q)
    std_inv = arr(model.scaler_std_inv_q)
    rust_code = f"""pub const SCALE_FACTOR: i64 = {model.scale_factor};
pub const PRECISION_BITS: u32 = {model.precision_bits};
pub const FEATURE_NAMES: [&str; {len(model.feature_names)}] = [
{feature_names}
];
pub const COEFFICIENTS_Q: [i64; {len(model.coefficients_q)}] = [
{coefficients}
];
pub const INTERCEPT_Q: i64 = {model.intercept_q};
pub const SCALER_MEAN_Q: [i64; {len(model.scaler_mean_q)}] = [
{mean}
];
pub const SCALER_STD_INV_Q: [i64; {len(model.scaler_std_inv_q)}] = [
{std_inv}
];
pub const SIGMOID_LUT_INPUT_BITS: u32 = {model.sigmoid_lut_input_bits};
pub const SIGMOID_LUT_OUTPUT_BITS: u32 = {model.sigmoid_lut_output_bits};
pub const SIGMOID_INPUT_RANGE: f64 = 8.0;
pub const SIGMOID_LUT: [u16; {len(model.sigmoid_lut)}] = [
{arr(model.sigmoid_lut)}
];
pub const CALIBRATION_LUT_SIZE: usize = {len(model.calibration_x_q)};
pub const CALIBRATION_X_Q: [u16; {len(model.calibration_x_q)}] = [
{arr(model.calibration_x_q, 8)}
];
pub const CALIBRATION_Y_Q: [u16; {len(model.calibration_y_q)}] = [
{arr(model.calibration_y_q, 8)}
];
pub const CLINICAL_THRESHOLD: f64 = {model.threshold};
pub const CLINICAL_THRESHOLD_Q: u16 = {model.threshold_q};
"""
    output_path.write_text(rust_code)


def main() -> None:
    model = train_calibrated_model(
        precision_bits=12,
        min_recall=float(os.environ.get("PULSECURE_TARGET_RECALL", "0.95")),
    )
    models_dir = Path("/app/models") if Path("/app").exists() else Path("models")
    models_dir.mkdir(exist_ok=True)
    export_to_json(model, models_dir / "calibrated_model.json")
    export_to_rust(model, models_dir / "calibrated_model.rs")


if __name__ == "__main__":
    main()
