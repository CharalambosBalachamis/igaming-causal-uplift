#  iGaming Causal Uplift & Retention Optimization Engine


An end-to-end Heterogeneous Treatment Effect (HTE) engine designed to maximize retention marketing ROI in the iGaming and digital subscription sectors. Replicating and extending the empirical methodology of Verhelst et al. (2023) on the **Orange Belgium Uplift Dataset (N=11,896)**, this project demonstrates why conventional churn prediction fails economically and how causal meta-learners unlock substantial net profit.

---

## Executive Summary & Business Impact

Standard predictive ML models rank users by baseline churn risk $P(Y=1 \mid X)$, inadvertently spending retention bonuses on **Lost Causes** (users who churn regardless) and triggering churn among **Sleeping Dogs** (users annoyed by marketing). 

By estimating the Conditional Average Treatment Effect (CATE) $\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$, this causal pipeline isolates **Persuadables** and enforces economically optimal intervention rules.

### Policy Simulation Benchmark (€250 LTV, €5 Retention Offer)

| Policy Strategy | Population Targeted | Net Incremental Retained | Campaign Cost | Net Incremental Profit | Campaign ROI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Do Nothing** | 0% (0) | 0.0 | €0.00 | **€0.00** | 0.0% |
| **2. Mass Campaign (100%)** | 100.0% (3,569) | +11.7 | €17,845.00 | **-€14,913.78** | -83.6% |
| **3. Risk-Based Churn Targeting (Top 30%)** | 30.0% (1,071) | -95.8 | €5,355.00 | **-€29,294.98** | -547.1% |
| **4. Causal X-Learner Policy** | **19.3% (688)** | **+108.3** | **€3,440.00** | **+€23,636.27** | **+687.1%** |

---

## Technical Methodology

### 1. Counterfactual Quadrant Decomposition
Replicating the joint potential outcome probabilities from Verhelst et al. (2023):
* **Sure Things** ($Y(0)=0, Y(1)=0$): **92.59%** *(Benchmark: 93.1%)*
* **Persuadables** ($Y(0)=1, Y(1)=0$): **3.90%** *(Benchmark: 3.5%)*
* **Sleeping Dogs** ($Y(0)=0, Y(1)=1$): **3.31%** *(Benchmark: 3.2%)*
* **Lost Causes** ($Y(0)=1, Y(1)=1$): **0.20%** *(Benchmark: 0.1%)*

### 2. Meta-Learner Tournament & AUUC Metrics
* **S-Learner Regularization Failure:** In a 160-dimensional PCA feature space, single tree ensembles regularize the sparse binary treatment indicator to zero variance (ATE = 0.00%).
* **X-Learner Superiority:** Cross-imputation and propensity score weighting achieve an **AUUC of 9.24** (vs. 4.44 Random and 4.41 Baseline Risk), more than doubling ranking efficiency.

### 3. TreeSHAP Attribution Diagnostics
* **PC1 (+4.07 vs. -0.11 pop mean):** Primary driver of persuadability and bonus responsiveness.
* **PC135 & PC5:** Latent behavioral signatures identifying sleeping-dog backfire risk.

---





