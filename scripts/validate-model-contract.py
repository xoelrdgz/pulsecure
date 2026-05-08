from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_FEATURES = [
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
def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(path: Path, strict_metadata: bool) -> int:
    data = json.loads(path.read_text())
    errors: list[str] = []
    warnings: list[str] = []
    require(data.get("schema_version") == 1, "schema_version must be 1", errors)
    feature_names = data.get("feature_names")
    require(feature_names == EXPECTED_FEATURES, "feature_names must match the deployed feature contract", errors)
    n_features = len(feature_names) if isinstance(feature_names, list) else 0
    for key in ("coefficients_q", "scaler_mean_q", "scaler_std_inv_q"):
        value = data.get(key)
        require(isinstance(value, list) and len(value) == n_features, f"{key} must have {n_features} values", errors)
    require(is_number(data.get("intercept_q")), "intercept_q must be numeric", errors)
    require(is_number(data.get("scale_factor")) and data["scale_factor"] > 0, "scale_factor must be positive", errors)
    require(is_number(data.get("precision_bits")) and data["precision_bits"] > 0, "precision_bits must be positive", errors)
    sigmoid = data.get("sigmoid_lut") or {}
    require(isinstance(sigmoid.get("values"), list) and len(sigmoid["values"]) >= 2, "sigmoid_lut.values must contain at least 2 entries", errors)
    require(is_number(sigmoid.get("output_bits")) and sigmoid["output_bits"] > 0, "sigmoid_lut.output_bits must be positive", errors)
    calibration = data.get("calibration_lut") or {}
    x_breakpoints = calibration.get("x_breakpoints")
    y_values = calibration.get("y_values")
    require(isinstance(x_breakpoints, list) and isinstance(y_values, list), "calibration_lut must contain x_breakpoints and y_values", errors)
    if isinstance(x_breakpoints, list) and isinstance(y_values, list):
        require(len(x_breakpoints) == len(y_values) and len(x_breakpoints) >= 2, "calibration_lut arrays must have equal length >= 2", errors)
    threshold = data.get("clinical_threshold") or {}
    require(is_number(threshold.get("value")) and 0.0 <= threshold["value"] <= 1.0, "clinical_threshold.value must be in [0, 1]", errors)
    require(is_number(threshold.get("recall")), "clinical_threshold.recall must be numeric", errors)
    require(is_number(threshold.get("precision")), "clinical_threshold.precision must be numeric", errors)
    validation = data.get("validation") or {}
    for key in ("auc_float", "auc_calibrated", "auc_quantized", "brier_before", "brier_after", "max_calibration_error"):
        require(is_number(validation.get(key)), f"validation.{key} must be numeric", errors)
    dataset = data.get("dataset") or {}
    metadata = data.get("training_metadata") or {}
    lineage_missing = [
        "dataset.sha256" if not dataset.get("sha256") else "",
        "dataset.n_samples" if not dataset.get("n_samples") else "",
        "training_metadata.trained_at" if not metadata.get("trained_at") else "",
    ]
    lineage_missing = [item for item in lineage_missing if item]
    if lineage_missing:
        warnings.append("incomplete traceability metadata: " + ", ".join(lineage_missing))
        if strict_metadata:
            errors.extend(lineage_missing)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Model contract OK: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--strict-metadata", action="store_true")
    args = parser.parse_args()
    return validate(args.model, args.strict_metadata)


if __name__ == "__main__":
    raise SystemExit(main())
