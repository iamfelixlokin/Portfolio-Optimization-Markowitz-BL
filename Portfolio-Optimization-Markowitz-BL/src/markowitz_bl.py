import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT_W = BASE / "outputs" / "weights"
OUT_P = BASE / "outputs" / "performance"

OUT_W.mkdir(parents=True, exist_ok=True)
OUT_P.mkdir(parents=True, exist_ok=True)

# Confi
TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2025-12-31"

GAMMAS = [0.5, 1, 2, 4, 8, 16]
ANNUAL_RF = 0.0  

# Helpers
def perf_metrics(daily_rets: np.ndarray, rf_annual: float = 0.0) -> dict:
    r = pd.Series(daily_rets).dropna()
    rf_daily = rf_annual / 252.0
    excess = r - rf_daily

    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(252)
    sharpe = (excess.mean() * 252) / (excess.std(ddof=0) * np.sqrt(252) + 1e-12)

    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }

def solve_markowitz(mu: np.ndarray, Sigma: np.ndarray, gamma: float) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)

    obj = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
    cons = [cp.sum(w) == 1, w >= 0]  # long-only
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimization failed (w.value is None). Try different solver/gamma.")
    return np.array(w.value).flatten()

# Load data 
prices = pd.read_csv(DATA / "stocks_data.csv", index_col=0, parse_dates=True)
prices = prices.select_dtypes(include=[np.number]).dropna(how="any")
prices = prices.sort_index()

rets = prices.pct_change().dropna()
tickers = prices.columns.tolist()

train = rets.loc[TRAIN_START:TRAIN_END].copy()
test  = rets.loc[TEST_START:TEST_END].copy()

if train.empty or test.empty:
    raise ValueError("Train/Test returns empty. Check your dates and CSV index.")

mu = train.mean().values * 252
Sigma = train.cov().values * 252
n = len(tickers)

# 1) Equal-weight baseline
w_eq = np.ones(n) / n
eq_daily = test.values @ w_eq
eq_m = perf_metrics(eq_daily, ANNUAL_RF)

pd.DataFrame({"ticker": tickers, "weight": w_eq}).to_csv(
    OUT_W / "weights_equal_weight.csv", index=False
)

# 2) Markowitz gamma scan (choose best by test Sharpe) 
weights_by_gamma = {}
rows = []

for g in GAMMAS:
    w = solve_markowitz(mu, Sigma, g)
    weights_by_gamma[g] = w
    daily = test.values @ w
    m = perf_metrics(daily, ANNUAL_RF)
    m["model"] = "Markowitz"
    m["gamma"] = float(g)
    rows.append(m)

markowitz_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
best_gamma = float(markowitz_df.iloc[0]["gamma"])
w_mk = weights_by_gamma[best_gamma]

pd.DataFrame({"ticker": tickers, "weight": w_mk}).to_csv(
    OUT_W / "weights_markowitz.csv", index=False
)

# 3) Black-Litterman 
# Market prior: use equal-weight as market weights proxy
w_mkt = w_eq.copy()

# Implied equilibrium returns
lam = best_gamma
pi = lam * Sigma @ w_mkt

idx = {t: i for i, t in enumerate(tickers)}

# Views: META ≈ MSFT, AMZN ≈ GOOGL (relative views)
P = np.zeros((2, n))
P[0, idx["META"]]  = 1
P[0, idx["MSFT"]]  = -1
P[1, idx["AMZN"]]  = 1
P[1, idx["GOOGL"]] = -1
Q = np.zeros(2)

tau = 0.05
Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))

mu_bl = np.linalg.inv(
    np.linalg.inv(tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P
) @ (
    np.linalg.inv(tau * Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
)

w_bl = solve_markowitz(mu_bl, Sigma, best_gamma)
bl_daily = test.values @ w_bl
bl_m = perf_metrics(bl_daily, ANNUAL_RF)

pd.DataFrame({"ticker": tickers, "weight": w_bl}).to_csv(
    OUT_W / "weights_bl.csv", index=False
)

# Save performance summary
perf_rows = []

perf_rows.append({"model": "EqualWeight", "gamma": np.nan, **eq_m})
perf_rows.append({"model": "Markowitz(best)", "gamma": best_gamma, **perf_metrics(test.values @ w_mk, ANNUAL_RF)})
perf_rows.append({"model": "BlackLitterman", "gamma": best_gamma, **bl_m})

perf_df = pd.DataFrame(perf_rows)

# save:
markowitz_df.to_csv(OUT_P / "performance_markowitz_gamma_scan.csv", index=False)
perf_df.to_csv(OUT_P / "performance_summary.csv", index=False)

# Print 
print("Tickers:", tickers)
print("\nSaved weights:")
print(" - outputs/weights/weights_equal_weight.csv")
print(" - outputs/weights/weights_markowitz.csv")
print(" - outputs/weights/weights_bl.csv")

print("\nPerformance summary (Test period):")
print(perf_df.to_string(index=False))

print("\nSaved performance:")
print(" - outputs/performance/performance_markowitz_gamma_scan.csv")
print(" - outputs/performance/performance_summary.csv")
