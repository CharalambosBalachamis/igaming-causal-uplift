"""
src/meta_learners.py
Modular implementation of Causal Meta-Learners (S, T, X) and 4-Quadrant Counterfactual Profiling.
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    MODELS_DIR,
    TABLES_DIR,
    RANDOM_SEED
)


class SLearner:
    """
    S-Learner (Single Model): Fits Y ~ f(X, W) and estimates CATE as f(X, 1) - f(X, 0).
    """
    def __init__(self, base_estimator=None):
        self.model = base_estimator or RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1
        )

    def fit(self, X: np.ndarray, W: np.ndarray, y: np.ndarray):
        X_augmented = np.column_stack([X, W])
        self.model.fit(X_augmented, y)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        X_w1 = np.column_stack([X, np.ones(len(X))])
        X_w0 = np.column_stack([X, np.zeros(len(X))])
        return self.model.predict(X_w1) - self.model.predict(X_w0)


class TLearner:
    """
    T-Learner (Two Models): Fits separate models mu_0(X) on W=0 and mu_1(X) on W=1.
    """
    def __init__(self, base_estimator=None):
        self.estimator_class = base_estimator or RandomForestRegressor
        self.model_0 = None
        self.model_1 = None

    def fit(self, X: np.ndarray, W: np.ndarray, y: np.ndarray):
        self.model_0 = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1
        )
        self.model_1 = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1
        )
        
        self.model_0.fit(X[W == 0], y[W == 0])
        self.model_1.fit(X[W == 1], y[W == 1])
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu_0 = self.model_0.predict(X)
        mu_1 = self.model_1.predict(X)
        return mu_0, mu_1

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        mu_0, mu_1 = self.predict_potential_outcomes(X)
        return mu_1 - mu_0


class XLearner:
    """
    X-Learner (Künzel et al. 2019): Cross-imputed effect modeling with propensity score weighting.
    """
    def __init__(self):
        self.mu_0 = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
        self.mu_1 = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
        self.tau_0_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
        self.tau_1_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    def fit(self, X: np.ndarray, W: np.ndarray, y: np.ndarray):
        # Stage 1: Fit base response functions
        self.mu_0.fit(X[W == 0], y[W == 0])
        self.mu_1.fit(X[W == 1], y[W == 1])

        # Stage 2: Calculate imputed counterfactual treatment effects
        d_1 = y[W == 1] - self.mu_0.predict(X[W == 1])
        d_0 = self.mu_1.predict(X[W == 0]) - y[W == 0]

        self.tau_1_model.fit(X[W == 1], d_1)
        self.tau_0_model.fit(X[W == 0], d_0)

        # Stage 3: Fit propensity estimator e(X)
        self.propensity_model.fit(X, W)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        e_hat = self.propensity_model.predict_proba(X)[:, 1]
        tau_0_pred = self.tau_0_model.predict(X)
        tau_1_pred = self.tau_1_model.predict(X)
        return e_hat * tau_0_pred + (1.0 - e_hat) * tau_1_pred


def compute_counterfactual_quadrants(
    model_0: RandomForestClassifier, 
    model_1: RandomForestClassifier, 
    X_test: np.ndarray
) -> pd.DataFrame:
    """
    Computes user decomposition into 4 counterfactual response quadrants (Verhelst et al. 2023).
    """
    p0 = model_0.predict_proba(X_test)[:, 1]
    p1 = model_1.predict_proba(X_test)[:, 1]

    prob_sure_thing = (1.0 - p0) * (1.0 - p1)
    prob_persuadable = p0 * (1.0 - p1)
    prob_sleeping_dog = (1.0 - p0) * p1
    prob_lost_cause = p0 * p1

    quadrants_df = pd.DataFrame({
        "Counterfactual Segment": [
            "Sure Things (No churn either way)",
            "Persuadables (Retained IF treated)",
            "Sleeping Dogs (Churns IF treated)",
            "Lost Causes (Churns either way)"
        ],
        "Condition": [
            "Y(0)=0, Y(1)=0",
            "Y(0)=1, Y(1)=0",
            "Y(0)=0, Y(1)=1",
            "Y(0)=1, Y(1)=1"
        ],
        "Estimated Mean (%)": [
            round(prob_sure_thing.mean() * 100, 2),
            round(prob_persuadable.mean() * 100, 2),
            round(prob_sleeping_dog.mean() * 100, 2),
            round(prob_lost_cause.mean() * 100, 2)
        ],
        "Benchmark Paper (%)": [93.1, 3.5, 3.2, 0.1]
    })
    
    quadrants_df.to_csv(TABLES_DIR / "counterfactual_quadrants_summary.csv", index=False)
    return quadrants_df


def benchmark_meta_learners(
    X_train: np.ndarray, W_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict]:
    """
    Trains S, T, and X learners, generates test set CATEs, and compiles a comparison table.
    """
    print("\n--- Training Meta-Learners ---")
    s_learner = SLearner().fit(X_train, W_train, y_train)
    t_learner = TLearner().fit(X_train, W_train, y_train)
    x_learner = XLearner().fit(X_train, W_train, y_train)

    tau_s = s_learner.predict_cate(X_test)
    tau_t = t_learner.predict_cate(X_test)
    tau_x = x_learner.predict_cate(X_test)

    cates = {"S-Learner": tau_s, "T-Learner": tau_t, "X-Learner": tau_x}
    fitted_models = {"s_learner": s_learner, "t_learner": t_learner, "x_learner": x_learner}

    # Save trained artifacts
    joblib.dump(fitted_models, MODELS_DIR / "meta_learners.joblib")

    summary_df = pd.DataFrame({
        "Estimator": ["S-Learner", "T-Learner", "X-Learner"],
        "Mean CATE (%)": [np.mean(tau_s) * 100, np.mean(tau_t) * 100, np.mean(tau_x) * 100],
        "Std CATE (%)": [np.std(tau_s) * 100, np.std(tau_t) * 100, np.std(tau_x) * 100],
        "Min CATE (Max Churn Reduction) (%)": [np.min(tau_s) * 100, np.min(tau_t) * 100, np.min(tau_x) * 100],
        "Max CATE (Worst Backfire) (%)": [np.max(tau_s) * 100, np.max(tau_t) * 100, np.max(tau_x) * 100],
        "% Persuadable (tau < -0.5%)": [np.mean(tau_s < -0.005) * 100, np.mean(tau_t < -0.005) * 100, np.mean(tau_x < -0.005) * 100],
        "% Sleeping Dog (tau > +0.5%)": [np.mean(tau_s > 0.005) * 100, np.mean(tau_t > 0.005) * 100, np.mean(tau_x > 0.005) * 100]
    }).round(4)

    summary_df.to_csv(TABLES_DIR / "meta_learners_benchmark.csv", index=False)
    return summary_df, cates, fitted_models


if __name__ == "__main__":
    from src.data_loader import fetch_openml_data, prepare_train_test_split
    from src.config import TREATMENT_COL, TARGET_COL

    df_clean = fetch_openml_data()
    train_df, test_df = prepare_train_test_split(df_clean)

    feature_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in [TREATMENT_COL, TARGET_COL]]
    
    X_tr = train_df[feature_cols].values
    W_tr = train_df[TREATMENT_COL].values
    y_tr = train_df[TARGET_COL].values

    X_te = test_df[feature_cols].values
    W_te = test_df[TREATMENT_COL].values
    y_te = test_df[TARGET_COL].values

    # 1. Run Meta-Learner tournament
    benchmark_tbl, cates, _ = benchmark_meta_learners(X_tr, W_tr, y_tr, X_te)
    print("\n================ META-LEARNER BENCHMARK SUMMARY ================")
    print(benchmark_tbl.to_string(index=False))

    # 2. Quadrant decomposition
    rf0 = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf0.fit(X_tr[W_tr == 0], y_tr[W_tr == 0])
    rf1.fit(X_tr[W_tr == 1], y_tr[W_tr == 1])

    quad_tbl = compute_counterfactual_quadrants(rf0, rf1, X_te)
    print("\n================ COUNTERFACTUAL QUADRANTS (TABLE 3 REPLICATION) ================")
    print(quad_tbl.to_string(index=False))