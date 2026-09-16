"""
src/evaluation.py
Qini metrics, Area Under Uplift Curve (AUUC), and Horvitz-Thompson P&L Policy Simulations.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict

# Fallback for numerical trapezoidal integration across NumPy versions
try:
    from numpy import trapezoid as np_trapezoid
except ImportError:
    from numpy import trapz as np_trapezoid

from src.config import (
    TABLES_DIR,
    DEFAULT_LTV,
    DEFAULT_BONUS_COST
)


def compute_qini_curve(y_true: np.ndarray, w_true: np.ndarray, tau_preds: np.ndarray, ascending: bool = True) -> Tuple[pd.DataFrame, float]:
    """
    Computes cumulative retained players across ranked treatment priority.
    ascending=True for churn (more negative tau = greater churn reduction / higher priority).
    """
    df_eval = pd.DataFrame({"y": y_true, "w": w_true, "tau": tau_preds})
    df_eval = df_eval.sort_values(by="tau", ascending=ascending).reset_index(drop=True)

    df_eval["n_t"] = (df_eval["w"] == 1).cumsum()
    df_eval["n_c"] = (df_eval["w"] == 0).cumsum()

    # Retained players: (churn == 0)
    df_eval["retained_t"] = ((df_eval["w"] == 1) & (df_eval["y"] == 0)).cumsum()
    df_eval["retained_c"] = ((df_eval["w"] == 0) & (df_eval["y"] == 0)).cumsum()

    scale_factor = np.where(df_eval["n_c"] > 0, df_eval["n_t"] / df_eval["n_c"], 1.0)
    df_eval["qini_gain"] = df_eval["retained_t"] - df_eval["retained_c"] * scale_factor

    x_axis = np.linspace(0, 1, len(df_eval))
    auuc = float(np_trapezoid(df_eval["qini_gain"], x_axis))
    return df_eval, auuc


def evaluate_policy_pnl(
    treatment_flags: np.ndarray,
    y_true: np.ndarray,
    w_true: np.ndarray,
    ltv: float = DEFAULT_LTV,
    unit_cost: float = DEFAULT_BONUS_COST
) -> Tuple[int, float, float, float, float]:
    """
    Computes unbiased expected net incremental profit via Horvitz-Thompson inverse probability weighting.
    """
    n_targeted = int(np.sum(treatment_flags))
    total_cost = n_targeted * unit_cost

    mask = (treatment_flags == 1)
    if n_targeted == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    p_w = float(np.mean(w_true))
    y_retained = 1.0 - y_true

    ret_t = np.sum(y_retained[mask & (w_true == 1)]) / p_w
    ret_c = np.sum(y_retained[mask & (w_true == 0)]) / (1.0 - p_w)
    net_retained = float(ret_t - ret_c)

    gross_revenue = net_retained * ltv
    net_profit = gross_revenue - total_cost
    roi_pct = (net_profit / total_cost) * 100.0 if total_cost > 0 else 0.0

    return n_targeted, net_retained, total_cost, net_profit, roi_pct


def run_comprehensive_evaluation(
    y_test: np.ndarray,
    w_test: np.ndarray,
    tau_x: np.ndarray,
    tau_t: np.ndarray,
    risk_baseline: np.ndarray,
    ltv: float = DEFAULT_LTV,
    unit_cost: float = DEFAULT_BONUS_COST
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes AUUC benchmarking, decile lift breakdowns, and full economic P&L simulations.
    """
    # 1. Qini & AUUC Curves
    qini_x, auuc_x = compute_qini_curve(y_test, w_test, tau_x, ascending=True)
    qini_t, auuc_t = compute_qini_curve(y_test, w_test, tau_t, ascending=True)
    qini_base, auuc_base = compute_qini_curve(y_test, w_test, risk_baseline, ascending=False)

    total_qini_final = qini_t["qini_gain"].iloc[-1]
    qini_random = np.linspace(0, total_qini_final, len(y_test))
    auuc_random = float(np_trapezoid(qini_random, np.linspace(0, 1, len(y_test))))

    auuc_df = pd.DataFrame({
        "Targeting Policy": ["Causal X-Learner", "Causal T-Learner", "Baseline Risk Model", "Random Targeting"],
        "AUUC Score": [round(auuc_x, 2), round(auuc_t, 2), round(auuc_base, 2), round(auuc_random, 2)],
        "Qini vs Random": [round(auuc_x - auuc_random, 2), round(auuc_t - auuc_random, 2), round(auuc_base - auuc_random, 2), 0.0]
    })
    auuc_df.to_csv(TABLES_DIR / "auuc_benchmark_summary.csv", index=False)

    # 2. Decile Lift Breakdown
    decile_indices = [int(len(y_test) * p) - 1 for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    deciles_df = pd.DataFrame({
        "Targeted Population": ["20%", "40%", "60%", "80%", "100%"],
        "X-Learner Lift": qini_x.loc[decile_indices, "qini_gain"].round(1).values,
        "T-Learner Lift": qini_t.loc[decile_indices, "qini_gain"].round(1).values,
        "Baseline Risk Lift": qini_base.loc[decile_indices, "qini_gain"].round(1).values,
        "Random Lift": np.round(qini_random[decile_indices], 1)
    })
    deciles_df.to_csv(TABLES_DIR / "decile_uplift_breakdown.csv", index=False)

    # 3. Policy Simulation P&L
    flag_none = np.zeros(len(y_test), dtype=int)
    flag_mass = np.ones(len(y_test), dtype=int)
    flag_risk = (risk_baseline >= np.percentile(risk_baseline, 70)).astype(int)
    flag_causal = (-tau_x > (unit_cost / ltv)).astype(int)

    pnl_none = evaluate_policy_pnl(flag_none, y_test, w_test, ltv, unit_cost)
    pnl_mass = evaluate_policy_pnl(flag_mass, y_test, w_test, ltv, unit_cost)
    pnl_risk = evaluate_policy_pnl(flag_risk, y_test, w_test, ltv, unit_cost)
    pnl_causal = evaluate_policy_pnl(flag_causal, y_test, w_test, ltv, unit_cost)

    pnl_df = pd.DataFrame({
        "Policy": [
            "1. Do Nothing",
            "2. Mass Outreach (100%)",
            "3. High-Risk Targeting (Top 30%)",
            "4. Causal Uplift (X-Learner Optimal)"
        ],
        "Contacted": [
            "0 (0.0%)",
            f"{pnl_mass[0]} (100.0%)",
            f"{pnl_risk[0]} ({pnl_risk[0]/len(y_test)*100:.1f}%)",
            f"{pnl_causal[0]} ({pnl_causal[0]/len(y_test)*100:.1f}%)"
        ],
        "Net Retained": [f"{pnl_none[1]:.1f}", f"{pnl_mass[1]:.1f}", f"{pnl_risk[1]:.1f}", f"{pnl_causal[1]:.1f}"],
        "Cost": [f"€{pnl_none[2]:,.2f}", f"€{pnl_mass[2]:,.2f}", f"€{pnl_risk[2]:,.2f}", f"€{pnl_causal[2]:,.2f}"],
        "Net Profit": [f"€{pnl_none[3]:,.2f}", f"€{pnl_mass[3]:,.2f}", f"€{pnl_risk[3]:,.2f}", f"€{pnl_causal[3]:,.2f}"],
        "ROI (%)": [f"{pnl_none[4]:.1f}%", f"{pnl_mass[4]:.1f}%", f"{pnl_risk[4]:.1f}%", f"{pnl_causal[4]:.1f}%"]
    })
    pnl_df.to_csv(TABLES_DIR / "executive_pnl_summary.csv", index=False)

    return auuc_df, deciles_df, pnl_df


if __name__ == "__main__":
    import sys
    import joblib
    import src.meta_learners
    from src.meta_learners import SLearner, TLearner, XLearner, benchmark_meta_learners
    from src.data_loader import fetch_openml_data, prepare_train_test_split
    from src.config import TREATMENT_COL, TARGET_COL, MODELS_DIR
    from sklearn.linear_model import LogisticRegression

    # Bind classes directly to __main__ so pickle resolves legacy namespace references
    sys.modules["__main__"].SLearner = SLearner
    sys.modules["__main__"].TLearner = TLearner
    sys.modules["__main__"].XLearner = XLearner

    df_clean = fetch_openml_data()
    train_df, test_df = prepare_train_test_split(df_clean)

    feature_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in [TREATMENT_COL, TARGET_COL]]

    X_tr = train_df[feature_cols].values
    W_tr = train_df[TREATMENT_COL].values
    y_tr = train_df[TARGET_COL].values

    X_te = test_df[feature_cols].values
    W_te = test_df[TREATMENT_COL].values
    y_te = test_df[TARGET_COL].values

    model_path = MODELS_DIR / "meta_learners.joblib"
    if model_path.exists():
        print(f"[INFO] Loading pre-trained meta-learners from: {model_path}")
        fitted_models = joblib.load(model_path)
        tau_x = fitted_models["x_learner"].predict_cate(X_te)
        tau_t = fitted_models["t_learner"].predict_cate(X_te)
    else:
        print("[INFO] No cached models found. Training meta-learners...")
        _, cates, fitted_models = benchmark_meta_learners(X_tr, W_tr, y_tr, X_te)
        tau_x, tau_t = cates["X-Learner"], cates["T-Learner"]

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_tr, y_tr)
    risk_preds = baseline.predict_proba(X_te)[:, 1]

    auuc_res, deciles_res, pnl_res = run_comprehensive_evaluation(
        y_te, W_te, tau_x, tau_t, risk_preds
    )

    print("\n================ OUT-OF-SAMPLE UPLIFT EVALUATION (AUUC) ================")
    print(auuc_res.to_string(index=False))

    print("\n================ CUMULATIVE INCREMENTAL GAIN BY POPULATION DECILE ================")
    print(deciles_res.to_string(index=False))

    print("\n================ EXECUTIVE P&L POLICY SIMULATION ================")
    print(pnl_res.to_string(index=False))