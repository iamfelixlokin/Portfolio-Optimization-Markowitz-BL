## Project Overview

This project implements a quantitative portfolio construction pipeline
based on Mean–Variance Optimization and the Black–Litterman model.

The goal is not to maximize short-term returns, but to study how
Black–Litterman integrates investor views into portfolio allocation
while improving risk stability (volatility and drawdown) in an
out-of-sample setting.

The project is designed as a research prototype rather than a trading system.

## Methodology

The pipeline consists of the following steps:

1. Price data collection (US equities)
2. Train/Test split
   - Train: 2020–2024
   - Test: 2025
3. Baseline portfolios
   - Equal-weight (1/N)
   - Mean–Variance Optimization (Markowitz)
4. Black–Litterman model
   - Market prior: equal-weight proxy
   - Views: conservative relative views
   - Objective: stabilize allocation under uncertainty
5. Out-of-sample backtesting and performance comparison

## Why Black–Litterman?

Black–Litterman is not used to aggressively outperform Markowitz.
Instead, its purpose is to:

- Mitigate sensitivity to expected return estimates
- Integrate investor views in a controlled and interpretable manner
- Improve portfolio stability (volatility and drawdown)

As a result, Black–Litterman portfolios may exhibit similar Sharpe ratios
to Markowitz, but with more stable risk characteristics.

## Results Summary (Out-of-Sample)

| Model           | Ann Return | Ann Vol | Sharpe | Max Drawdown |
|-----------------|------------|---------|--------|--------------|
| Equal Weight    | ~27%       | ~30%    | ~0.94  | ~-28%        |
| Markowitz       | ~33%       | ~30%    | ~1.08  | ~-30%        |
| Black–Litterman | ~26%       | ~25%    | ~1.04  | ~-26%        |

Black–Litterman shows improved risk stability compared to Markowitz,
while maintaining comparable risk-adjusted performance.

## Project Structure

```text
src/
├── markowitz_bl.py      # Portfolio construction (Equal / MV / BL)
├── plot_backtest.py     # Equity & drawdown visualization

data/
├── stocks_data.csv

outputs/
├── weights/
├── performance/
```
## Limitations & Future Work

- Views are currently constructed using conservative, structured assumptions
- No unstructured data (e.g. news, earnings, macro text) is used at this stage
- The system focuses on allocation research, not execution or transaction costs

Future extensions may include:
- View generation from unstructured data
- Regime-based or factor-driven views
- Rolling rebalancing and weight stability analysis
- Integration with RWA / on-chain NAV representation
