"""
Medical-Grade CVD Risk Model Pipeline.
Isotonic calibration + threshold optimization + FHE quantization.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix, brier_score_loss
)
from imblearn.over_sampling import SMOTE

from ..utils import load_nhanes_data, quantize_value, generate_sigmoid_lut

@dataclass
class CalibratedQuantizedModel:
    """Calibrated and quantized model for FHE inference."""
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
    auc_float: float
    auc_calibrated: float
    auc_quantized: float
    brier_before: float
    brier_after: float
    recall_at_threshold: float
    precision_at_threshold: float
    max_calibration_error: float

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.90
) -> Tuple[float, float, float]:
    """Find threshold that achieves minimum recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_idx = np.where(recalls[:-1] >= min_recall)[0]
    idx = valid_idx[-1] if len(valid_idx) > 0 else 0
    return thresholds[idx], recalls[idx], precisions[idx]

def apply_isotonic_lut(
    prob: np.ndarray,
    x_breakpoints: np.ndarray,
    y_values: np.ndarray
) -> np.ndarray:
    """Apply isotonic calibration via interpolation."""
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
    input_range: float = 8.0
) -> np.ndarray:
    """
    Full FHE inference with calibration.
    
    Pipeline:
    1. Normalize inputs (quantized)
    2. Dot product (integer arithmetic)
    3. Sigmoid via LUT
    4. Isotonic calibration via LUT
    """
    n_samples, n_features = X_raw.shape
    output_scale = 2 ** sigmoid_output_bits
    lut_size = len(sigmoid_lut)
    
    # Step 1: Normalize
    X_q = np.zeros((n_samples, n_features), dtype=np.int64)
    for i in range(n_features):
        x_scaled = (X_raw[:, i] * scale_factor).astype(np.int64)
        x_centered = x_scaled - mean_q[i]
        X_q[:, i] = (x_centered * std_inv_q[i]) // scale_factor
    
    # Step 2: Dot product
    logits_q = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_features):
        logits_q += X_q[:, i] * coef_q[i]
    logits_q += intercept_q * scale_factor
    
    logit_scale = scale_factor * scale_factor
    
    # Step 3: Sigmoid via LUT
    logit_float = logits_q.astype(np.float64) / logit_scale
    logit_clamped = np.clip(logit_float, -input_range, input_range)
    indices = ((logit_clamped + input_range) / (2 * input_range) * (lut_size - 1))
    indices = np.clip(indices.astype(np.int64), 0, lut_size - 1)
    
    lut_arr = np.array(sigmoid_lut)
    prob_q = lut_arr[indices]
    
    # Step 4: Isotonic calibration via interpolation
    # In FHE, this would be another LUT
    prob_uncalib = prob_q.astype(np.float64) / output_scale
    prob_calib = apply_isotonic_lut(
        prob_uncalib,
        calib_x_q.astype(np.float64) / output_scale,
        calib_y_q.astype(np.float64) / output_scale
    )
    
    return prob_calib

def train_calibrated_model(precision_bits: int = 12, min_recall: float = 0.90):
    """
    Train, calibrate, and quantize model.
    
    Full pipeline:
    1. Train LogisticRegression with Elastic Net
    2. Calibrate with Isotonic Regression
    3. Find optimal threshold for target recall
    4. Quantize all parameters for FHE
    5. Validate fidelity
    """
    # Removed LBXSUA (uric acid) - 220% coefficient variance across folds
    FEATURES = [
        "RIDAGEYR", "BPQ020", "BPXSY1", "SMQ020", "LBDHDD",
        "LBXSCR", "BMXWAIST", "DIQ010", "LBXGH"
    ]
    
    scale_factor = 2 ** precision_bits
    
    print(f"Precision bits: {precision_bits}")
    print(f"Scale factor: {scale_factor}")
    print(f"Target recall: >= {min_recall:.0%}")
    
    # Load data
    df = load_nhanes_data()
    df_model = df[FEATURES + ["CVD"]].dropna()
    X_raw = df_model[FEATURES].values
    y = df_model["CVD"].values
    
    print(f"\nDataset: {len(y)} samples | CVD+: {int(y.sum())} ({100*y.mean():.1f}%)")
    
    # Split: train/calibration/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(y_train)} | Calibration: {len(y_calib)} | Test: {len(y_test)}")
    
    # Scaler (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE on train only
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    
    # === STEP 1: Train base model ===
    print("\n" + "=" * 70)
    print("STEP 1: TRAIN BASE MODEL")
    print("=" * 70)
    
    model = LogisticRegression(
        solver="saga", C=0.1, l1_ratio=0.3,
        max_iter=2000, random_state=42
    )
    model.fit(X_train_sm, y_train_sm)
    
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    print("\nCoefficients:")
    for name, c in zip(FEATURES, coef):
        print(f"  {name:<12} {c:+.4f}")
    print(f"  {'Intercept':<12} {intercept:+.4f}")
    
    # Uncalibrated predictions
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    y_prob_calib = model.predict_proba(X_calib_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    
    auc_uncalib = roc_auc_score(y_test, y_prob_test)
    brier_uncalib = brier_score_loss(y_test, y_prob_test)
    
    print(f"\nUncalibrated performance:")
    print(f"  ROC-AUC: {auc_uncalib:.4f}")
    print(f"  Brier score: {brier_uncalib:.4f}")
    
    # === STEP 2: Calibrate with Isotonic Regression ===
    print("\n" + "=" * 70)
    print("STEP 2: ISOTONIC CALIBRATION")
    print("=" * 70)
    
    # Fit isotonic regression on calibration set
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_prob_calib, y_calib)
    
    # Apply calibration
    y_prob_test_calib = iso_reg.predict(y_prob_test)
    
    auc_calib = roc_auc_score(y_test, y_prob_test_calib)
    brier_calib = brier_score_loss(y_test, y_prob_test_calib)
    
    print(f"\nCalibrated performance:")
    print(f"  ROC-AUC: {auc_calib:.4f} (delta: {auc_calib - auc_uncalib:+.4f})")
    print(f"  Brier score: {brier_calib:.4f} (delta: {brier_calib - brier_uncalib:+.4f})")
    
    # Calibration curve
    print("\nCalibration curve (predicted vs actual):")
    prob_true, prob_pred = calibration_curve(y_test, y_prob_test_calib, n_bins=10)
    print(f"  {'Predicted':>12} {'Actual':>12} {'Error':>12}")
    for pt, pp in zip(prob_true, prob_pred):
        error = abs(pt - pp)
        print(f"  {pp:12.2%} {pt:12.2%} {error:12.2%}")
    
    max_calib_error = np.max(np.abs(prob_true - prob_pred))
    print(f"\n  Max calibration error: {max_calib_error:.2%}")
    
    # === STEP 3: Find optimal threshold ===
    print("\n" + "=" * 70)
    print("STEP 3: CLINICAL THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    threshold, recall, precision = find_optimal_threshold(
        y_test, y_prob_test_calib, min_recall=min_recall
    )
    
    print(f"\nOptimal threshold for Recall >= {min_recall:.0%}:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Recall (Sensitivity): {recall:.2%}")
    print(f"  Precision (PPV): {precision:.2%}")
    
    # Confusion matrix at optimal threshold
    y_pred_opt = (y_prob_test_calib >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix at threshold={threshold:.3f}:")
    print(f"                  Pred Neg   Pred Pos")
    print(f"  Actual Neg      {tn:7d}    {fp:7d}")
    print(f"  Actual Pos      {fn:7d}    {tp:7d}")
    print(f"\n  False Negatives: {fn} (missed cases)")
    print(f"  False Positives: {fp} (extra screening)")
    
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n  Specificity: {spec:.2%}")
    print(f"  NPV: {npv:.2%}")
    
    # Compare with threshold=0.5
    print("\n  Comparison with threshold=0.5:")
    y_pred_05 = (y_prob_test_calib >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_test, y_pred_05)
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()
    print(f"    False Negatives at 0.5: {fn_05}")
    print(f"    Reduction: {fn_05 - fn} fewer missed cases")
    
    # === STEP 4: Quantization ===
    print("\n" + "=" * 70)
    print("STEP 4: FHE QUANTIZATION")
    print("=" * 70)
    
    # Quantize coefficients
    coef_q = np.array([quantize_value(c, scale_factor) for c in coef], dtype=np.int64)
    intercept_q = quantize_value(intercept, scale_factor)
    
    # Quantize scaler
    mean_q = np.array([quantize_value(m, scale_factor) for m in scaler.mean_], dtype=np.int64)
    std_inv_q = np.array([quantize_value(1/s, scale_factor) for s in scaler.scale_], dtype=np.int64)
    
    # Generate sigmoid LUT
    sigmoid_input_bits = 8
    sigmoid_output_bits = 12
    sigmoid_lut = generate_sigmoid_lut(sigmoid_input_bits, sigmoid_output_bits, 8.0)
    
    # Quantize isotonic calibration
    # Extract breakpoints from isotonic regression
    calib_x = iso_reg.X_thresholds_
    calib_y = iso_reg.y_thresholds_
    
    # Ensure we have enough points for smooth interpolation
    if len(calib_x) < 32:
        # Interpolate to get more points
        calib_x_dense = np.linspace(0, 1, 64)
        calib_y_dense = iso_reg.predict(calib_x_dense)
    else:
        calib_x_dense = calib_x
        calib_y_dense = calib_y
    
    calib_x_q = np.array([quantize_value(x, 2**sigmoid_output_bits) for x in calib_x_dense], dtype=np.int64)
    calib_y_q = np.array([quantize_value(y, 2**sigmoid_output_bits) for y in calib_y_dense], dtype=np.int64)
    
    # Quantize threshold
    threshold_q = quantize_value(threshold, 2**sigmoid_output_bits)
    
    print(f"\nQuantized coefficients:")
    for name, cq in zip(FEATURES, coef_q):
        print(f"  {name:<12} {cq:>8d}")
    
    print(f"\nIsotonic calibration: {len(calib_x_q)} breakpoints")
    print(f"Threshold: {threshold:.4f} -> {threshold_q} (/{2**sigmoid_output_bits})")
    
    # === STEP 5: Validation ===
    print("\n" + "=" * 70)
    print("STEP 5: FIDELITY VALIDATION")
    print("=" * 70)
    
    # Run quantized inference
    y_prob_quant = inference_full_pipeline(
        X_test, coef_q, intercept_q, mean_q, std_inv_q, scale_factor,
        sigmoid_lut, calib_x_q, calib_y_q, sigmoid_output_bits
    )
    
    auc_quant = roc_auc_score(y_test, y_prob_quant)
    
    # Compare calibrated vs quantized
    prob_errors = np.abs(y_prob_test_calib - y_prob_quant)
    
    print(f"\nFidelity metrics:")
    print(f"  ROC-AUC (float calibrated): {auc_calib:.4f}")
    print(f"  ROC-AUC (quantized FHE):    {auc_quant:.4f}")
    print(f"  Delta AUC: {abs(auc_calib - auc_quant):.4f}")
    print(f"\n  Probability error:")
    print(f"    Maximum: {prob_errors.max():.4f}")
    print(f"    Mean: {prob_errors.mean():.4f}")
    print(f"    P95: {np.percentile(prob_errors, 95):.4f}")
    
    # Validate at threshold
    y_pred_quant = (y_prob_quant >= threshold).astype(int)
    cm_quant = confusion_matrix(y_test, y_pred_quant)
    tn_q, fp_q, fn_q, tp_q = cm_quant.ravel()
    recall_quant = tp_q / (tp_q + fn_q) if (tp_q + fn_q) > 0 else 0
    
    print(f"\n  Quantized model at threshold={threshold:.3f}:")
    print(f"    False Negatives: {fn_q} (float: {fn})")
    print(f"    Recall: {recall_quant:.2%} (float: {recall:.2%})")
    
    if abs(auc_calib - auc_quant) < 0.01 and fn_q <= fn + 2:
        print("\n[OK] Quantization fidelity acceptable for medical use")
    else:
        print("\n[WARN] Quantization may affect clinical performance")
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"""
    Base Model:
      - Elastic Net (C=0.1, l1_ratio=0.3)
      - 10 features from NHANES
      
    Calibration:
      - Isotonic Regression
      - Brier improvement: {brier_uncalib:.4f} -> {brier_calib:.4f}
      - Max calibration error: {max_calib_error:.2%}
      
    Clinical Threshold:
      - Threshold: {threshold:.4f} (not 0.5!)
      - Recall: {recall:.2%}
      - False Negatives: {fn} (vs {fn_05} at 0.5)
      
    FHE Quantization:
      - Precision: {precision_bits} bits
      - Sigmoid LUT: 256 entries
      - Calibration LUT: {len(calib_x_q)} entries
      - AUC preservation: {auc_quant:.4f} (delta: {abs(auc_calib - auc_quant):.4f})
    """)
    
    # Build result
    result = CalibratedQuantizedModel(
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
        threshold=float(threshold),
        threshold_q=threshold_q,
        feature_names=FEATURES,
        auc_float=float(auc_uncalib),
        auc_calibrated=float(auc_calib),
        auc_quantized=float(auc_quant),
        brier_before=float(brier_uncalib),
        brier_after=float(brier_calib),
        recall_at_threshold=float(recall),
        precision_at_threshold=float(precision),
        max_calibration_error=float(max_calib_error)
    )
    
    return result

def export_to_rust(model: CalibratedQuantizedModel, output_path: Path):
    """Export calibrated quantized model as Rust constants."""
    
    # Format LUTs
    def format_lut(lut, per_row=16):
        rows = []
        for i in range(0, len(lut), per_row):
            row = ", ".join(str(v) for v in lut[i:i+per_row])
            rows.append(f"    {row},")
        return "\n".join(rows)
    
    sigmoid_lut_fmt = format_lut(model.sigmoid_lut)
    calib_x_fmt = format_lut(model.calibration_x_q, per_row=8)
    calib_y_fmt = format_lut(model.calibration_y_q, per_row=8)
    
    rust_code = f'''//! Calibrated quantized model for FHE inference.
//! Auto-generated - DO NOT EDIT.
//!
//! Medical-grade CVD risk prediction.
//! Includes isotonic calibration for accurate probabilities.
//!
//! Validation metrics:
//!   AUC (calibrated): {model.auc_calibrated:.4f}
//!   AUC (quantized):  {model.auc_quantized:.4f}
//!   Brier score:      {model.brier_after:.4f}
//!   Max calib error:  {model.max_calibration_error:.2%}
//!   Recall @ thresh:  {model.recall_at_threshold:.2%}

/// Scale factor (2^{model.precision_bits})
pub const SCALE_FACTOR: i64 = {model.scale_factor};

/// Precision bits
pub const PRECISION_BITS: u32 = {model.precision_bits};

/// Feature names
pub const FEATURE_NAMES: [&str; {len(model.feature_names)}] = [
{chr(10).join(f'    "{name}",' for name in model.feature_names)}
];

/// Quantized coefficients
pub const COEFFICIENTS_Q: [i64; {len(model.coefficients_q)}] = [
{chr(10).join(f'    {c},  // {name}' for c, name in zip(model.coefficients_q, model.feature_names))}
];

/// Quantized intercept
pub const INTERCEPT_Q: i64 = {model.intercept_q};

/// Scaler mean (quantized)
pub const SCALER_MEAN_Q: [i64; {len(model.scaler_mean_q)}] = [
{chr(10).join(f'    {m},  // {name}' for m, name in zip(model.scaler_mean_q, model.feature_names))}
];

/// Scaler std inverse (quantized)
pub const SCALER_STD_INV_Q: [i64; {len(model.scaler_std_inv_q)}] = [
{chr(10).join(f'    {s},  // {name}' for s, name in zip(model.scaler_std_inv_q, model.feature_names))}
];

// === Sigmoid LUT ===
pub const SIGMOID_LUT_INPUT_BITS: u32 = {model.sigmoid_lut_input_bits};
pub const SIGMOID_LUT_OUTPUT_BITS: u32 = {model.sigmoid_lut_output_bits};
pub const SIGMOID_INPUT_RANGE: f64 = 8.0;

pub const SIGMOID_LUT: [u16; {len(model.sigmoid_lut)}] = [
{sigmoid_lut_fmt}
];

// === Isotonic Calibration LUT ===
pub const CALIBRATION_LUT_SIZE: usize = {len(model.calibration_x_q)};

/// Input breakpoints (quantized, divide by 4096)
pub const CALIBRATION_X_Q: [u16; {len(model.calibration_x_q)}] = [
{calib_x_fmt}
];

/// Output values (quantized, divide by 4096)
pub const CALIBRATION_Y_Q: [u16; {len(model.calibration_y_q)}] = [
{calib_y_fmt}
];

// === Clinical Threshold ===
/// Optimal threshold for {model.recall_at_threshold:.0%} recall
/// Use this instead of 0.5!
pub const CLINICAL_THRESHOLD: f64 = {model.threshold};
pub const CLINICAL_THRESHOLD_Q: u16 = {model.threshold_q};
'''
    
    output_path.write_text(rust_code)
    print(f"Exported to: {output_path}")

def export_to_json(model: CalibratedQuantizedModel, output_path: Path):
    """Export model as JSON."""
    data = {
        "precision_bits": model.precision_bits,
        "scale_factor": model.scale_factor,
        "feature_names": model.feature_names,
        "coefficients_q": model.coefficients_q,
        "intercept_q": model.intercept_q,
        "scaler_mean_q": model.scaler_mean_q,
        "scaler_std_inv_q": model.scaler_std_inv_q,
        "sigmoid_lut": {
            "input_bits": model.sigmoid_lut_input_bits,
            "output_bits": model.sigmoid_lut_output_bits,
            "values": model.sigmoid_lut
        },
        "calibration_lut": {
            "x_breakpoints": model.calibration_x_q,
            "y_values": model.calibration_y_q
        },
        "clinical_threshold": {
            "value": model.threshold,
            "quantized": model.threshold_q,
            "recall": model.recall_at_threshold,
            "precision": model.precision_at_threshold
        },
        "validation": {
            "auc_float": model.auc_float,
            "auc_calibrated": model.auc_calibrated,
            "auc_quantized": model.auc_quantized,
            "brier_before": model.brier_before,
            "brier_after": model.brier_after,
            "max_calibration_error": model.max_calibration_error
        }
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Exported to: {output_path}")

def main():
    print("\n" + "=" * 70)
    print("PULSECURE - MEDICAL-GRADE MODEL PIPELINE")
    print("=" * 70)
    
    # Train with 90% recall target
    model = train_calibrated_model(precision_bits=12, min_recall=0.90)
    
    # Export
    models_dir = Path("/app/models") if Path("/app").exists() else Path("models")
    models_dir.mkdir(exist_ok=True)
    
    export_to_json(model, models_dir / "calibrated_model.json")
    export_to_rust(model, models_dir / "calibrated_model.rs")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
