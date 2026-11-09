# Bank Credit Risk & Loan Default Prediction
Python · scikit-learn · XGBoost · Monte Carlo — framed for (re)insurance underwriting: risk assessment, pricing, portfolio exposure, and excess-of-loss cover.

> **Data note:** synthetic 60,000-loan book with known ground truth and a deliberately under-differentiated tariff. Not client data.

## Pipeline (`credit_risk.py`)
1. **PD model:** logistic regression (interpretable benchmark) vs XGBoost. Holdout AUC 0.759 vs 0.754, Gini 0.52, KS 0.38; predicted default rate 6.85% vs actual 6.87%.
2. **Expected loss:** EL = PD × LGD × EAD, LGD by loan purpose.
3. **Risk-based pricing:** required rate = funding + expected loss + opex + capital charge on unexpected loss + margin; compared with the current tariff by risk band.
4. **Portfolio loss distribution:** one-factor Gaussian copula (ρ = 0.05), 4,000 simulations: EL 12.9m, 99.5% VaR 28.3m, expected shortfall 31.3m on 288m exposure.
5. **Aggregate excess-of-loss cover:** layer from 1.25 × EL up to the 99.5% VaR. Expected layer loss 0.87m, technical premium 1.45m, rate on line 11.9%, volatility cut by about 30%.
6. **Profitability by risk band and purpose:** the worst band (E, PD about 23%) loses money under the current tariff and is flagged to re-price.

## Outputs
`model_comparison.csv`, `risk_band_profitability.csv`, `purpose_profitability.csv`, `portfolio_and_reinsurance_summary.csv`, `loan_level_pricing.csv`, `logit_coefficients_standardised.csv`.

## Data analytical info 
**Bank Credit Risk & Loan Default Prediction** | Python, scikit-learn, XGBoost
- Built PD models (logistic regression, XGBoost; Gini 0.52) and expected-loss framework (PD × LGD × EAD) on a 60k-loan synthetic book.
- Derived risk-based loan pricing and flagged the highest-risk band as loss-making under the current tariff.
- Simulated portfolio credit losses (Gaussian copula) and priced an aggregate excess-of-loss cover (VaR 99.5%, rate on line, volatility reduction).
