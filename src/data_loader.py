"""
src/data_loader.py
Ingestion, type coercion, train/test splitting, and randomized covariate balance auditing.
"""
import openml
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

from src.config import (
    OPENML_DATASET_ID,
    RAW_PARQUET_PATH,
    CLEAN_PARQUET_PATH,
    TREATMENT_COL,
    TARGET_COL,
    RANDOM_SEED,
    TEST_SIZE,
    SMD_THRESHOLD,
    TABLES_DIR
)


def fetch_openml_data(dataset_id: int = OPENML_DATASET_ID, output_path=RAW_PARQUET_PATH) -> pd.DataFrame:
    """
    Downloads or loads cached microdata from OpenML and enforces numeric typing.
    """
    if output_path.exists():
        print(f"[INFO] Loading cached raw data from: {output_path}")
        df = pd.read_parquet(output_path)
    else:
        print(f"[INFO] Downloading OpenML dataset ID {dataset_id}...")
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        df = pd.DataFrame(X, columns=attribute_names)
        if y is not None and TARGET_COL not in df.columns:
            df[TARGET_COL] = y

        df.to_parquet(output_path, index=False)
        print(f"[SUCCESS] Download complete. Cached {df.shape[0]} rows and {df.shape[1]} columns to {output_path}")

    # Standardize treatment and target columns as numeric binary integers
    if TREATMENT_COL in df.columns:
        df[TREATMENT_COL] = pd.to_numeric(df[TREATMENT_COL], errors="coerce").fillna(0).astype(int)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    return df


def audit_covariate_balance(
    df: pd.DataFrame, 
    treatment_col: str = TREATMENT_COL, 
    threshold: float = SMD_THRESHOLD
) -> pd.DataFrame:
    """
    Computes Standardized Mean Differences (SMD) and two-sample t-tests 
    across numeric covariates to verify RCT randomization balance.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    # Select only numeric feature columns (e.g. PCA regressors)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    covariates = [c for c in numeric_cols if c not in [treatment_col, TARGET_COL]]

    records: List[Dict] = []

    for col in covariates:
        s_t = treated[col].dropna()
        s_c = control[col].dropna()

        m_t, v_t = float(s_t.mean()), float(s_t.var())
        m_c, v_c = float(s_c.mean()), float(s_c.var())

        pooled_sd = np.sqrt((v_t + v_c) / 2)
        smd = (m_t - m_c) / pooled_sd if pooled_sd > 0 else 0.0
        _, p_val = stats.ttest_ind(s_t, s_c, equal_var=False)

        records.append({
            "covariate": col,
            "mean_treated": round(m_t, 4),
            "mean_control": round(m_c, 4),
            "smd": round(smd, 4),
            "abs_smd": round(abs(smd), 4),
            "p_value": round(p_val, 4)
        })

    balance_df = pd.DataFrame(records).sort_values("abs_smd", ascending=False)
    
    balance_report_path = TABLES_DIR / "covariate_balance_audit.csv"
    balance_df.to_csv(balance_report_path, index=False)

    max_smd = balance_df["abs_smd"].max()
    imbalanced = balance_df[balance_df["abs_smd"] > threshold]

    print("\n--- RCT Covariate Balance Audit ---")
    print(f"Total Numeric Covariates Checked : {len(covariates)}")
    print(f"Max Absolute SMD                 : {max_smd:.4f} (Threshold < {threshold})")
    print(f"Imbalanced Covariates            : {len(imbalanced)} ({len(imbalanced)/len(covariates)*100:.1f}%)")
    print(f"Report saved to                  : {balance_report_path}")

    return balance_df


def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = TEST_SIZE, 
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratifies by treatment and target to preserve joint distribution in train/test splits.
    """
    stratify_col = df[TREATMENT_COL].astype(str) + "_" + df[TARGET_COL].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify_col
    )
    
    df.to_parquet(CLEAN_PARQUET_PATH, index=False)
    print(f"\n[INFO] Train/Test Split Ready -> Train: {train_df.shape[0]} rows, Test: {test_df.shape[0]} rows")
    return train_df, test_df


if __name__ == "__main__":
    df_raw = fetch_openml_data()
    _ = audit_covariate_balance(df_raw)
    train_data, test_data = prepare_train_test_split(df_raw)