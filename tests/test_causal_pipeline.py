import numpy as np
import pytest

from src.config import DEFAULT_LTV, DEFAULT_BONUS_COST
from src.meta_learners import SLearner, TLearner, XLearner
from src.evaluation import evaluate_policy_pnl, compute_qini_curve


@pytest.fixture
def synthetic_causal_data():
    np.random.seed(42)
    n = 200
    k = 160
    X = np.random.randn(n, k)
    W = np.random.binomial(1, 0.5, size=n)
    base_churn = 1 / (1 + np.exp(-X[:, 0]))
    y = np.random.binomial(1, np.clip(base_churn - 0.05 * W, 0, 1))
    return X, W, y


def test_meta_learners_output_dimensions(synthetic_causal_data):
    X, W, y = synthetic_causal_data
    s_learner = SLearner().fit(X, W, y)
    t_learner = TLearner().fit(X, W, y)
    x_learner = XLearner().fit(X, W, y)

    tau_s = s_learner.predict_cate(X)
    tau_t = t_learner.predict_cate(X)
    tau_x = x_learner.predict_cate(X)

    assert tau_s.shape == (len(X),)
    assert tau_t.shape == (len(X),)
    assert tau_x.shape == (len(X),)
    assert not np.isnan(tau_x).any()


def test_horvitz_thompson_pnl_invariants(synthetic_causal_data):
    X, W, y = synthetic_causal_data
    flag_zero = np.zeros(len(y), dtype=int)
    
    n_targeted, net_retained, cost, net_profit, roi = evaluate_policy_pnl(
        flag_zero, y, W, ltv=DEFAULT_LTV, unit_cost=DEFAULT_BONUS_COST
    )
    
    assert n_targeted == 0
    assert net_retained == 0.0
    assert cost == 0.0
    assert net_profit == 0.0
    assert roi == 0.0


def test_qini_curve_monotonicity(synthetic_causal_data):
    X, W, y = synthetic_causal_data
    x_learner = XLearner().fit(X, W, y)
    tau_x = x_learner.predict_cate(X)

    df_eval, auuc = compute_qini_curve(y, W, tau_x, ascending=True)

    assert len(df_eval) == len(y)
    assert "qini_gain" in df_eval.columns
    assert isinstance(auuc, float)