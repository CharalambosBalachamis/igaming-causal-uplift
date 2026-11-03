# iGaming Retention Optimization: End-to-End Causal AI Lakehouse

An end-to-end Heterogeneous Treatment Effect (HTE) engine and executive dashboard designed to maximize retention marketing ROI. Replicating the empirical methodology of Verhelst et al. (2023) on the Orange Belgium Uplift Dataset (N=11,896), this project demonstrates how shifting from traditional Churn Prediction to **Causal Machine Learning** unlocks massive profitability.



###  Executive Dashboard (Power BI)
*(The dashboard dynamically calculates Horvitz-Thompson Inverse Probability Weighting (IPW) using custom DAX to track realized incremental profit.)*

![Power BI Executive Dashboard](dashboard/Screenshot%20(23).png)

###  Executive Summary & Business Impact
Standard predictive ML models evaluate baseline churn risk. However, targeting users purely based on this risk inadvertently wastes retention bonuses on **Lost Causes** (users who churn regardless) and actively triggers churn among **Sleeping Dogs** (users annoyed by marketing intervention).

By estimating the Conditional Average Treatment Effect (CATE) via an **X-Learner** architecture, this pipeline isolates the **Persuadables** and enforces economically optimal intervention rules.

**Policy Simulation Benchmark (€250 LTV, €5 Retention Offer):**
| Policy Strategy | Population Targeted | Net Incremental Retained | Campaign Cost | Net Incremental Profit | Campaign ROI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Do Nothing** | 0% (0) | 0.0 | €0.00 | €0.00 | 0.0% |
| **2. Mass Campaign (100%)** | 100.0% (3,569) | +11.7 | €17,845.00 | -€14,913.78 | -83.6% |
| **3. Risk-Based Churn (Top 30%)** | 30.0% (1,071) | -95.8 | €5,355.00 | -€29,294.98 | -547.1% |
| **4. Causal X-Learner Policy** | **19.3% (688)** | **+108.3** | **€3,440.00** | **+€23,636.27** | **+687.1%** |

###  Tech Stack & Pipeline Architecture
* **Data Engineering (DuckDB & SQL):** In-memory columnar processing, executing raw SQL against Parquet files for type coercion and rigorous RCT covariate balance auditing.
* **Causal ML (Python, Scikit-Learn):** Implementation of S, T, and X-Learners. The X-Learner utilizes cross-imputation and propensity score weighting to correct for observational imbalances, achieving an AUUC of 9.24.
* **Interpretability (TreeSHAP):** Extraction of latent behavioral signatures identifying "Sleeping Dog" backfire risks.
* **BI & Analytics (Power BI & DAX):** Flat-file Parquet exports ingested into Power BI, utilizing variable-driven DAX measures for interactive KPI tracking.

###  Counterfactual Quadrant Decomposition
Replicating the joint potential outcome probabilities from Verhelst et al. (2023):
* **Sure Things** ($Y(0)=0, Y(1)=0$): 92.59% *(Benchmark: 93.1%)*
* **Persuadables** ($Y(0)=1, Y(1)=0$): 3.90% *(Benchmark: 3.5%)*
* **Sleeping Dogs** ($Y(0)=0, Y(1)=1$): 3.31% *(Benchmark: 3.2%)*
* **Lost Causes** ($Y(0)=1, Y(1)=1$): 0.20% *(Benchmark: 0.1%)*