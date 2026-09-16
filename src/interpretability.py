"""
src/interpretability.py
TreeSHAP attribution and latent behavioral profiling for Causal Meta-Learners.
"""
import joblib
import shap
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.config import (
    MODELS_DIR,
    TABLES_DIR,
    TREATMENT_COL,
    TARGET_COL,
    RANDOM_SEED
)


def run_shap_diagnostics(
    x_learner,
    X_test: np.ndarray,
    feature_names: List[str],
    sample_size: int = 500
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes TreeSHAP attributions on the X-Learner treatment response model
    and profiles Persuadables vs. Sleeping Dogs.
    """
    print(f"\n--- Computing TreeSHAP Values (Sample Size = {sample_size}) ---")
    
    # Explain stage-2 treatment effect model (tau_1_model)
    target_model = x_learner.tau_1_model
    explainer = shap.TreeExplainer(target_model)
    
    X_sample = X_test[:sample_size]
    shap_vals = explainer.shap_values(X_sample)

    # 1. Global Feature Importance
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]

    top_features = [feature_names[i] for i in top_indices]
    top_importance = mean_abs_shap[top_indices]

    correlations = [
        np.corrcoef(X_sample[:, i], shap_vals[:, i])[0, 1] 
        for i in top_indices
    ]

    shap_summary = pd.DataFrame({
        "Rank": list(range(1, 11)),
        "Feature": top_features,
        "Mean |SHAP| (%)": (top_importance * 100).round(4),
        "Causal Dynamics": [
            "Higher Value -> Increases Churn (Sleeping Dog Risk)" if c > 0.1
            else "Higher Value -> Decreases Churn (Persuadable Signal)" if c < -0.1
            else "Non-Linear / Mixed Dynamics"
            for c in correlations
        ]
    })
    shap_summary.to_csv(TABLES_DIR / "shap_top_drivers.csv", index=False)

    # 2. Behavioral Segmentation Signature
    tau_preds = target_model.predict(X_sample)
    top_persuadables = np.argsort(tau_preds)[:50]
    top_sleeping_dogs = np.argsort(tau_preds)[::-1][:50]

    profile_df = pd.DataFrame({
        "Top Feature": [feature_names[i] for i in top_indices[:5]],
        "Persuadable Mean (Top 50)": [round(float(X_sample[top_persuadables, i].mean()), 3) for i in top_indices[:5]],
        "Sleeping Dog Mean (Top 50)": [round(float(X_sample[top_sleeping_dogs, i].mean()), 3) for i in top_indices[:5]],
        "Population Baseline": [round(float(X_sample[:, i].mean()), 3) for i in top_indices[:5]]
    })
    profile_df.to_csv(TABLES_DIR / "behavioral_profile_comparison.csv", index=False)

    return shap_summary, profile_df


if __name__ == "__main__":
    from src.data_loader import fetch_openml_data, prepare_train_test_split
    # Explicitly import all learner classes so joblib can unpickle them
    from src.meta_learners import SLearner, TLearner, XLearner, benchmark_meta_learners

    df_clean = fetch_openml_data()
    train_df, test_df = prepare_train_test_split(df_clean)

    feature_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in [TREATMENT_COL, TARGET_COL]]

    X_tr = train_df[feature_cols].values
    W_tr = train_df[TREATMENT_COL].values
    y_tr = train_df[TARGET_COL].values

    X_te = test_df[feature_cols].values

    model_path = MODELS_DIR / "meta_learners.joblib"
    if model_path.exists():
        fitted_models = joblib.load(model_path)
    else:
        _, _, fitted_models = benchmark_meta_learners(X_tr, W_tr, y_tr, X_te)

    summary_tbl, profiles_tbl = run_shap_diagnostics(fitted_models["x_learner"], X_te, feature_cols)

    print("\n================ TOP 10 CAUSAL BEHAVIORAL DRIVERS (SHAP) ================")
    print(summary_tbl.to_string(index=False))

    print("\n================ LATENT PROFILES: PERSUADABLES VS SLEEPING DOGS ================")
    print(profiles_tbl.to_string(index=False))