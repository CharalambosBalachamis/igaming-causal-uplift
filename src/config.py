"""
src/config.py
Centralized configuration, paths, and economic constants for the Causal ML pipeline.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Ensure runtime directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Data Source
OPENML_DATASET_ID = 45580
RAW_PARQUET_PATH = RAW_DATA_DIR / "orange_belgium_uplift.parquet"
CLEAN_PARQUET_PATH = PROCESSED_DATA_DIR / "uplift_clean_panel.parquet"

# Causal Variables
TREATMENT_COL = "t"
TARGET_COL = "churn"
RANDOM_SEED = 42
TEST_SIZE = 0.30

# Balance Diagnostics Thresholds
SMD_THRESHOLD = 0.10

# iGaming Economic & Retention Constants
DEFAULT_LTV = 250.0        # Value of a retained customer (€)
DEFAULT_BONUS_COST = 5.0    # Cost per targeted retention offer / bonus (€)