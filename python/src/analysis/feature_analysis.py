"""
Pulsecure: Feature Selection Analysis for NHANES
Multicollinearity-Aware with Elastic Net Regularization

Objectives:
- CV-AUC > 0.85
- 5-8 active features (clinically relevant)
- VIF < 5 for all features
- Coefficient magnitudes suitable for FHE (with bootstrapping)
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

def load_nhanes_all_features(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load NHANES dataset with all features."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "nhanes_full.csv"
    
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return pd.read_csv(cache_path)
    
    print("Downloading NHANES dataset from Kaggle...")
    import kagglehub
    
    path = kagglehub.dataset_download("cdc/national-health-and-nutrition-examination-survey")
    path = Path(path)
    
    demographic = pd.read_csv(path / "demographic.csv")
    examination = pd.read_csv(path / "examination.csv")
    labs = pd.read_csv(path / "labs.csv")
    questionnaire = pd.read_csv(path / "questionnaire.csv")
    
    df = demographic.copy()
    df = df.merge(examination, on="SEQN", how="left", suffixes=('', '_exam'))
    df = df.merge(labs, on="SEQN", how="left", suffixes=('', '_lab'))
    df = df.merge(questionnaire, on="SEQN", how="left", suffixes=('', '_quest'))
    
    # Create CVD target
    cvd_cols = ["MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
    df["CVD"] = 0
    for col in cvd_cols:
        if col in df.columns:
            df.loc[df[col] == 1, "CVD"] = 1
    
    df.to_csv(cache_path, index=False)
    print(f"Cached to {cache_path}")
    return df

def get_candidate_features() -> dict:
    """Clinical features for CVD prediction."""
    return {
        "RIDAGEYR": ("Age (years)", "demographic"),
        "RIAGENDR": ("Sex (1=M, 2=F)", "demographic"),
        "BMXBMI": ("Body Mass Index", "body"),
        "BMXWAIST": ("Waist circumference (cm)", "body"),
        "BPXSY1": ("Systolic BP (mmHg)", "vitals"),
        "BPXDI1": ("Diastolic BP (mmHg)", "vitals"),
        "LBXTC": ("Total cholesterol (mg/dL)", "lipids"),
        "LBDHDD": ("HDL cholesterol (mg/dL)", "lipids"),
        "LBXGH": ("HbA1c (%)", "glucose"),
        "DIQ010": ("Diabetes diagnosed", "condition"),
        "LBXSCR": ("Creatinine (mg/dL)", "kidney"),
        "LBXSUA": ("Uric acid (mg/dL)", "kidney"),
        "SMQ020": ("Smoked 100+ cigarettes", "lifestyle"),
        "BPQ020": ("Hypertension diagnosed", "condition"),
    }

def calculate_vif(X: pd.DataFrame) -> pd.Series:
    """Calculate Variance Inflation Factor."""
    from sklearn.linear_model import LinearRegression
    
    vif_data = {}
    X_arr = X.values
    
    for i, col in enumerate(X.columns):
        y = X_arr[:, i]
        X_others = np.delete(X_arr, i, axis=1)
        
        if X_others.shape[1] == 0:
            vif_data[col] = 1.0
            continue
            
        model = LinearRegression()
        model.fit(X_others, y)
        r_squared = model.score(X_others, y)
        
        vif_data[col] = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float('inf')
    
    return pd.Series(vif_data).sort_values(ascending=False)

def select_uncorrelated_features(
    df: pd.DataFrame,
    max_features: int = 10,
    max_missing_pct: float = 40.0,
    max_correlation: float = 0.7,
) -> list[str]:
    """Select features avoiding multicollinearity."""
    candidates = get_candidate_features()
    target = "CVD"
    
    # Filter available features
    available = []
    for var_name in candidates.keys():
        if var_name in df.columns:
            missing_pct = df[var_name].isna().sum() / len(df) * 100
            if missing_pct <= max_missing_pct:
                available.append(var_name)
    
    print(f"\n{'='*70}")
    print("FEATURE SELECTION")
    print(f"{'='*70}")
    print(f"Available features (<{max_missing_pct}% missing): {len(available)}")
    
    # Prepare data
    X = df[available].copy()
    y = df[target].copy()
    
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    print(f"Samples: {len(y)} | CVD+: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Rank by predictive power
    correlations = X.corrwith(y).abs()
    mi_scores = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=5)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    
    rank_corr = correlations.rank(ascending=False)
    rank_mi = mi_scores.rank(ascending=False)
    rank_rf = rf_importance.rank(ascending=False)
    combined_score = (rank_corr + rank_mi + rank_rf) / 3
    feature_ranking = combined_score.sort_values()
    
    # Correlation matrix
    corr_matrix = X.corr()
    
    # Greedy selection
    selected = []
    print(f"\nGreedy selection (max |r| = {max_correlation}):")
    
    for feat in feature_ranking.index:
        if len(selected) >= max_features:
            break
        
        is_correlated = False
        for sel_feat in selected:
            if abs(corr_matrix.loc[feat, sel_feat]) > max_correlation:
                is_correlated = True
                print(f"  SKIP {feat} (corr with {sel_feat})")
                break
        
        if not is_correlated:
            selected.append(feat)
            print(f"  ADD  {feat} - {candidates[feat][0]}")
    
    return selected

def train_elastic_net(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[LogisticRegression, dict, StandardScaler]:
    """
    Train with Elastic Net regularization.
    
    Elastic Net = L1 + L2, allows correlated features to share weights
    instead of being eliminated like pure Lasso.
    """
    target = "CVD"
    
    X = df[features].copy()
    y = df[target].copy()
    
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    # StandardScaler (mean=0, std=1) - required for FHE coefficient comparability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print(f"\n{'='*70}")
    print("ELASTIC NET MODEL SELECTION")
    print(f"{'='*70}")
    
    # Grid search for best C and l1_ratio
    best_auc = 0
    best_params = {'C': 1.0, 'l1_ratio': 0.5}
    
    for c in [0.1, 0.5, 1.0, 2.0]:
        for l1_ratio in [0.3, 0.5, 0.7]:
            model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                C=c,
                l1_ratio=l1_ratio,
                max_iter=3000,
                random_state=42
            )
            scores = cross_val_score(model, X_train_bal, y_train_bal, cv=5, scoring='roc_auc')
            if scores.mean() > best_auc:
                best_auc = scores.mean()
                best_params = {'C': c, 'l1_ratio': l1_ratio}
    
    print(f"Best params: C={best_params['C']}, l1_ratio={best_params['l1_ratio']}")
    print(f"Training CV-AUC: {best_auc:.4f}")
    
    # Train final model
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=best_params['C'],
        l1_ratio=best_params['l1_ratio'],
        max_iter=3000,
        random_state=42
    )
    model.fit(X_train_bal, y_train_bal)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Full dataset CV
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    
    metrics = {
        'test_roc_auc': roc_auc_score(y_test, y_proba),
        'test_recall': recall_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'C': best_params['C'],
        'l1_ratio': best_params['l1_ratio'],
    }
    
    return model, metrics, scaler

def analyze_coefficients_for_fhe(
    model: LogisticRegression,
    features: list[str],
    scaler: StandardScaler,
):
    """Analyze coefficient magnitudes for FHE compatibility."""
    candidates = get_candidate_features()
    
    print(f"\n{'='*70}")
    print("COEFFICIENT ANALYSIS FOR FHE")
    print(f"{'='*70}")
    
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_[0],
        'abs_coef': np.abs(model.coef_[0]),
    }).sort_values('abs_coef', ascending=False)
    
    active_features = []
    
    print(f"\n{'Feature':<15} {'Coef':>10} {'|Coef|':>10} {'Bits':>8} {'Status':<12}")
    print("-" * 60)
    
    for _, row in coef_df.iterrows():
        coef = row['coefficient']
        abs_coef = row['abs_coef']
        feat = row['feature']
        
        # Estimate bits needed: log2(|coef| * scale_factor)
        # Assuming 16-bit fixed point for inputs
        bits_needed = int(np.ceil(np.log2(max(abs_coef * 65536, 1))))
        
        if abs_coef < 0.01:
            status = "INACTIVE"
        else:
            status = "ACTIVE"
            active_features.append(feat)
        
        sign = "+" if coef > 0 else "-"
        print(f"{feat:<15} {sign}{abs_coef:>9.4f} {abs_coef:>10.4f} {bits_needed:>8} {status:<12}")
    
    print(f"\nActive features: {len(active_features)}/{len(features)}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    
    # FHE noise budget analysis
    max_coef = coef_df['abs_coef'].max()
    total_weight = coef_df['abs_coef'].sum()
    
    print(f"\nFHE Noise Analysis:")
    print(f"  Max coefficient magnitude: {max_coef:.4f}")
    print(f"  Sum of |coefficients|: {total_weight:.4f}")
    print(f"  Estimated bits for max coef: {int(np.ceil(np.log2(max_coef * 65536)))}")
    
    if max_coef > 3.0:
        print(f"  WARNING: Max coef > 3.0, consider coefficient clipping or more regularization")
    elif max_coef > 2.0:
        print(f"  CAUTION: Max coef > 2.0, ensure adequate bootstrapping in tfhe-rs")
    else:
        print(f"  OK: Coefficients within reasonable range for FHE")
    
    return active_features

def generate_report():
    """Main analysis pipeline."""
    print("\n" + "="*70)
    print("PULSECURE - FEATURE ANALYSIS REPORT")
    print("="*70)
    
    # Load data
    df = load_nhanes_all_features()
    
    # Select uncorrelated features
    selected = select_uncorrelated_features(
        df, 
        max_features=10,
        max_missing_pct=40.0,
        max_correlation=0.7,
    )
    
    # VIF analysis
    print(f"\n{'='*70}")
    print("MULTICOLLINEARITY CHECK (VIF)")
    print(f"{'='*70}")
    
    X_check = df[selected].copy()
    for col in X_check.columns:
        X_check[col] = X_check[col].fillna(X_check[col].median())
    
    vif = calculate_vif(X_check)
    for feat, vif_val in vif.items():
        status = "SEVERE" if vif_val > 10 else ("MODERATE" if vif_val > 5 else "OK")
        print(f"  {feat:<15} VIF={vif_val:>6.2f} [{status}]")
    
    # Train model
    model, metrics, scaler = train_elastic_net(df, selected)
    
    # Report metrics
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")
    print(f"Test Recall:    {metrics['test_recall']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"CV-AUC (5-fold): {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")
    
    # Coefficient analysis
    active = analyze_coefficients_for_fhe(model, selected, scaler)
    
    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    checks = [
        ("CV-AUC > 0.85", metrics['cv_auc_mean'] > 0.85),
        ("Active features >= 5", len(active) >= 5),
        ("All VIF < 10", all(v < 10 for v in vif.values)),
        ("Max |coef| < 3.0", max(abs(model.coef_[0])) < 3.0),
    ]
    
    all_pass = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            all_pass = False
    
    # Generate code
    candidates = get_candidate_features()
    
    print(f"\n{'='*70}")
    print("RECOMMENDED FEATURE_NAMES")
    print(f"{'='*70}")
    print("\nFEATURE_NAMES = [")
    for feat in active:
        desc = candidates.get(feat, ("Unknown",))[0]
        print(f'    "{feat}",  # {desc}')
    print("]")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return {
        'features': active,
        'metrics': metrics,
        'model': model,
        'scaler': scaler,
        'validation_passed': all_pass,
    }

if __name__ == "__main__":
    generate_report()
