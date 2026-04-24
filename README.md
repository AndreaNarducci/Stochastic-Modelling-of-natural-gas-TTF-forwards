# TTF Natural Gas 1-month Forwards- Stochastic Modelling Framework

A quantitative framework for modelling and forecasting TTF natural gas front-month futures prices using six stochastic models across two structurally distinct families. Built as an independent research project, following a BSc on weather derivatives pricing.

## Motivation

Natural gas prices exhibit two main dynamics: mean-reversion toward an equilibrium price driven by storage and supply fundamentals, and volatility clustering where large shocks are followed by further large shocks (memory). The 2021 to 2022 European energy crisis was responsible for an abrupt change of the price structure and dynamics.

I investigate which family of stochastic models better describes TTF price dynamics, and how does the answer change across different market regimes

## Repository Structure


ttf-stochastic-modelling/
│
├── ttf_6models.py          # Main script — six models, two families
├── ttf_statistics.py       # Exploratory data analysis and stylised facts
├── TTF_Methodology.docx    # Full methodology note (models, fitting, metrics)
└── README.md


> **Data note:** The TTF price series (`ttf_forward1.csv`) is not included in this repository. The scripts expect a CSV with a single column `Prices` containing daily front-month futures prices starting from 2018-01-02.

---
## Models

### OU Family — Mean-Reverting

All three models are based on the Ornstein–Uhlenbeck process in continuous time:

```
dY_t = κ(μ_t − Y_t) dt + σ dW_t
```

with exact discrete-time transition variance `σ²_r = σ²/(2κ) · (1 − exp(−2κΔt))`.

| Model | Mean specification | Volatility |
|---|---|---|
| **OU M1** | Constant μ | Constant σ_r |
| **OU M2** | Sinusoidal seasonal μ(t) = μ₀ + A·sin(2πt/252 + φ) | Constant σ_r |
| **OU M2 + GARCH** | Seasonal μ(t) as above | GARCH(1,1) on OU residuals |

Fitted via OLS (M1, M2) and MLE (GARCH component) on log-prices.

### GARCH Family — Random Walk

All three models assume log-prices follow a random walk with time-varying conditional variance. No mean-reversion.

| Model | Innovation distribution | Variance equation |
|---|---|---|
| **GARCH-Gaussian** | N(0,1) | Standard GARCH(1,1) |
| **GARCH-t** | Student-t(ν), estimated ν | Standard GARCH(1,1) |
| **EGARCH** | N(0,1) | log(h_t) = ω + α(\|z\| − E\|z\|) + γz + β·log(h_{t−1}) |

All fitted via MLE on log-returns using L-BFGS-B (GARCH) or Nelder-Mead (EGARCH).

## Evaluation Metrics

### CRPS Primary Metric

The Continuous Ranked Probability Score (Gneiting & Raftery 2007) is a proper scoring rule for distributional forecasts:

CRPS(F, y) = E_F|X − y| − 0.5* E_F|X − X'|

Where the original closed form expression involves the integration on R of the quadratic difference of the cumulative distribution function and the Heavside on the observed price (x); this expression is analogous to the one above as for Gneiting & Raftery 2007.

All six models are evaluated on one-step-ahead log-return CRPS, with the predictive distribution re-conditioned to the observed price each day. This guarantees strict comparability across both models families (mean reverting vs random walk)

For Gaussian predictive distributions the CRPS has a closed form (Gneiting & Raftery 2007, eq. 21):

CRPS = σ · [z(2Φ(z)−1) + 2φ(z) − 1/√π],    z = (y − μ)/σ


For GARCH-t (non-Gaussian predictive) an ensemble estimator with 10,000 draws per step is used, converging to the true CRPS at rate for large number of simulations.

### Path-Based Metrics — Secondary

RMSE, MAE and 5–95% interval coverage are computed on the median of 1,000 Monte Carlo paths. They are meaningful only for short test windows (< 60 business days), due to the unbounded nature of the variance with respect to time for random-walk based models (all GARCHs in this section), the coverage will asymptotically converge to 100% and be extremely conservative (thus another point to it being not a good indicator on the long run). Moreover over long periods the comparability across the two family of models ceases in terms of significance as mean reverting variance converges to the (diffusion parameter)/2*k.

---

## Outputs

Running `ttf_6models.py` produces four figures:

- **fig1_ou_family.png** — fan charts for OU M1, OU M2, OU M2+GARCH
- **fig2_garch_family.png** — fan charts for GARCH-Gaussian, GARCH-t, EGARCH
- **fig3_crps_time.png** — one-step CRPS over the test period, by family
- **fig4_tables.png** — unified results table + model parameters table

Running `ttf_statistics.py` produces a five-panel EDA figure:
- Historical price series
- Log-return distribution vs. Gaussian and KDE
- ACF of log-returns
- Log-price distribution with Gaussian mixture fit
- ACF of squared log-returns (volatility clustering)
I advise to first run the statistics script in order to have a visual cue
---
## Preview of the outputs - Statistics
This is an example of the outputs from the Statistics script, once the user has choosen the preferred period to investigate (this specific example shows the whole history available):


<img width="1713" height="2599" alt="image" src="https://github.com/user-attachments/assets/a04965f1-66a2-4692-bb74-8333466b474f" />

1. Absolute price history
2. T-student density vs Gaussian density on log-returns
3. Autocorrelation function on log-returns at varying time-lag; red bands are 95% Confidence Interval
4. Squared autocorrelation function on log-returns; consistent crossing of the confidence interval bands justifies volatility clustering 


## Preview of the outputs - TTF 6 models
This is an example of the outputs from the main TTF script. Once the user has choosen the calibration and out-of-sample test periods:

1. Fan charts for Ornestein-Uhlenbeck family models

<img width="1370" height="427" alt="Fancharts-Jan2025-June2025" src="https://github.com/user-attachments/assets/0994255f-727f-4510-a88a-87fc0fb1f059" />

2. Fan charts for random walk-GARCH family models

<img width="2721" height="853" alt="image" src="https://github.com/user-attachments/assets/68be3fd9-5e9f-456e-ad40-74aca0656f30" />

3. Continuous Ranked Probability Score over the test period for both family of models

<img width="1857" height="1136" alt="image" src="https://github.com/user-attachments/assets/9e4318ae-57d9-490c-ad42-ebf030fc2d25" />

4. Results and parameters table

<img width="1875" height="855" alt="image" src="https://github.com/user-attachments/assets/832a314a-84d3-443e-a60f-5f9ff72f1741" />









## Configuration

Both scripts are configured via constants at the top of the file:

```python
# ttf_6models.py
TRAIN_END = '2025-01-01'
TEST_START = '2025-03-01'
TEST_END = '2025-06-01'
CSV_PRICES = r'path/to/ttf_forward1.csv'
N_PATHS = 1000
SEED  = 42
```

```python
# ttf_statistics.py
chunk = allprices[allprices['Date'].between('2025-01-01', '2025-06-01')]
```

---

## Requirements

```
numpy
pandas
scipy
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

---

## References

- Bollerslev, T. (1986). Generalised autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347–370.
- Schwartz, E. S. (1997). The stochastic behavior of commodity prices. *Journal of Finance*, 52(3), 923–973.

---
## Future Research Direction

**1. Regime switching**

Markov Switching GARCH (MS-GARCH); explicitly models transitions between low and high volatility regimes, resolves fake persistence
Hidden Markov Model (HMM) on price levels; partially explored in this project

**2. Stochastic parameters**

Stochastic drift μ via particle filter; more robust than linear Kalman filter on regime-switching series, motivated by the comparison with Heston model estimation approaches
Two-factor Schwartz-Smith (1997) model; decomposes log-price into long-run equilibrium and short-run deviation, calibrated on the full forward curve via Kalman filter

**3. Distributional extensions**

GARCH with NIG (Normal Inverse Gaussian) innovations, more flexible than t-Student, independently parametrises skewness and tail heaviness
CEV (Constant Elasticity of Variance), price-level dependent volatility, relevant for commodity markets

**4. Exogenous regressors**

Weekly/monthly aggregation of HDD and storage deviation; daily regressors showed no significance (p > 0.48), lower frequency may recover predictive power
LNG import flows and pipeline interconnection variables

**5. Evaluation framework**

Diebold-Mariano test for formal statistical comparison between models
Probability Integral Transform (PIT) histogram for calibration diagnostics, complementary to CRPS
Walk-forward validation instead of single train/test split

**6. Term structure**

Extension to the full forward curve multi-factor model calibrated across all maturities, required for derivatives pricing beyond the front-month contract
## Author

Andrea Narducci, BSc Industrial Engineering, MSc Chemical Engineering

