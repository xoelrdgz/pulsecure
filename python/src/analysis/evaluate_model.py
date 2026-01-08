"""Evaluation of the calibrated cardiovascular risk model."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, brier_score_loss, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

from ..utils import load_nhanes_data

FEATURES = [
    "RIDAGEYR", "BPQ020", "BPXSY1", "SMQ020", "LBDHDD",
    "LBXSCR", "BMXWAIST", "DIQ010", "LBXSUA", "LBXGH"
]

def find_optimal_threshold(y_true, y_prob, min_recall=0.90):
    """Find threshold for target recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_idx = np.where(recalls[:-1] >= min_recall)[0]
    idx = valid_idx[-1] if len(valid_idx) > 0 else 0
    return thresholds[idx], recalls[idx], precisions[idx]

def main():
    df = load_nhanes_data()
    df_model = df[FEATURES + ["CVD"]].dropna()
    X = df_model[FEATURES].values
    y = df_model["CVD"].values

    print("=" * 60)
    print("MODEL EVALUATION - NHANES CVD (WITH CALIBRATION)")
    print("=" * 60)
    print(f"\nSamples: {len(y)} | CVD+: {int(y.sum())} ({100*y.mean():.1f}%)")

    # Split: train/calibration/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(y_train)} | Calibration: {len(y_calib)} | Test: {len(y_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE on train
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    # Train model
    model = LogisticRegression(
        solver="saga", C=0.1, l1_ratio=0.3, max_iter=2000, random_state=42
    )
    model.fit(X_train_sm, y_train_sm)

    # Uncalibrated predictions
    y_prob_calib = model.predict_proba(X_calib_scaled)[:, 1]
    y_prob_test_uncalib = model.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "=" * 60)
    print("UNCALIBRATED PERFORMANCE")
    print("=" * 60)
    auc_uncalib = roc_auc_score(y_test, y_prob_test_uncalib)
    brier_uncalib = brier_score_loss(y_test, y_prob_test_uncalib)
    print(f"  ROC-AUC:     {auc_uncalib:.4f}")
    print(f"  Brier Score: {brier_uncalib:.4f}")

    # Isotonic calibration
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_prob_calib, y_calib)
    y_prob_test = iso_reg.predict(y_prob_test_uncalib)

    print("\n" + "=" * 60)
    print("CALIBRATED PERFORMANCE (Isotonic Regression)")
    print("=" * 60)
    auc_calib = roc_auc_score(y_test, y_prob_test)
    brier_calib = brier_score_loss(y_test, y_prob_test)
    print(f"  ROC-AUC:     {auc_calib:.4f} (delta: {auc_calib - auc_uncalib:+.4f})")
    print(f"  Brier Score: {brier_calib:.4f} (delta: {brier_calib - brier_uncalib:+.4f})")

    # Calibration curve
    print("\nCalibration Curve:")
    prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)
    print(f"  {'Predicted':>12} {'Actual':>12} {'Error':>12}")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"  {pp:12.2%} {pt:12.2%} {abs(pt-pp):12.2%}")
    print(f"\n  Max Calibration Error: {np.max(np.abs(prob_true - prob_pred)):.2%}")

    # Optimal threshold
    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLD (Recall >= 90%)")
    print("=" * 60)
    threshold, recall, precision = find_optimal_threshold(y_test, y_prob_test, 0.90)
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  Precision: {precision:.2%}")

    # Confusion at optimal
    y_pred_opt = (y_prob_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix at threshold={threshold:.3f}:")
    print(f"                 Pred Neg   Pred Pos")
    print(f"  Actual Neg      {tn:6d}     {fp:6d}")
    print(f"  Actual Pos      {fn:6d}     {tp:6d}")
    print(f"\n  False Negatives: {fn}")
    print(f"  NPV: {tn/(tn+fn):.2%}" if (tn+fn) > 0 else "")

    # Compare with 0.5
    y_pred_05 = (y_prob_test >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_test, y_pred_05)
    fn_05 = cm_05[1, 0]
    print(f"\n  Comparison vs threshold=0.5:")
    print(f"    FN at 0.5: {fn_05} vs FN at optimal: {fn}")
    print(f"    Reduction: {fn_05 - fn} fewer missed cases")

    # Coefficients
    print("\n" + "=" * 60)
    print("MODEL COEFFICIENTS")
    print("=" * 60)
    coefs = sorted(zip(FEATURES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefs:
        print(f"  {name:<12} {'+' if coef >= 0 else '-'} {abs(coef):6.4f}")
    print(f"  {'Intercept':<12}   {model.intercept_[0]:6.4f}")

if __name__ == "__main__":
    main()
