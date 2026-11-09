"""Bank credit risk & loan default prediction, framed for (re)insurance underwriting:
PD model -> EL = PD x LGD x EAD -> risk-based pricing -> portfolio loss distribution (Gaussian copula)
-> aggregate excess-of-loss cover on credit losses -> segment profitability."""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import xgboost as xgb

SEED = 42; rng = np.random.default_rng(SEED); N = 60_000
FUNDING, OPEX, MARGIN, CAP_K = 0.030, 0.012, 0.010, 0.5   # funding cost, opex, target margin, capital-charge multiple
RHO = 0.05                                                # asset correlation (one-factor copula)
REINS_LOAD, REINS_EXP = 0.20, 0.10

# ---------- 1. Synthetic loan book (known ground truth) ----------
purposes = ["Debt consolidation", "Home improvement", "Car", "Small business", "Education", "Other"]
d = pd.DataFrame({
    "income": np.clip(rng.lognormal(10.8, 0.45, N), 15e3, 400e3).round(0),
    "credit_score": np.clip(rng.normal(690, 70, N), 450, 850).round(0),
    "dti": np.clip(rng.beta(2.2, 5, N) * 70, 2, 65).round(1),
    "emp_years": np.clip(rng.gamma(2, 3, N), 0, 40).round(1),
    "purpose": rng.choice(purposes, N, p=[.35, .15, .15, .1, .1, .15]),
    "term": rng.choice([36, 60], N, p=[.65, .35]),
})
d["loan_amount"] = np.clip(d.income * rng.uniform(.1, .6, N), 2e3, 80e3).round(-2)
d["lti"] = d.loan_amount / d.income
z = (-3.0 - 0.012 * (d.credit_score - 690) + 0.035 * (d.dti - 25) + 1.6 * (d.lti - .3)
     - 0.03 * d.emp_years.clip(upper=10) + (d.term == 60) * .35
     + d.purpose.map({"Small business": .5, "Education": .15, "Debt consolidation": .1}).fillna(0))
d["pd_true"] = 1 / (1 + np.exp(-z)); d["default"] = rng.binomial(1, d.pd_true)
lgd_mu = d.purpose.map({"Car": .45, "Home improvement": .5, "Small business": .75}).fillna(.65)
d["lgd_real"] = np.clip(rng.normal(lgd_mu, .15), .05, 1)
d["ead"] = (d.loan_amount * 0.85).round(0)
d["rate_current"] = (0.085 + 0.00012 * (700 - d.credit_score)).clip(.06, .12)  # under-differentiated tariff
d["loss_real"] = d.default * d.lgd_real * d.ead
print(f"loans={N:,} default rate={d.default.mean():.2%}")

# ---------- 2. PD models ----------
X = pd.get_dummies(d[["credit_score", "dti", "emp_years", "lti", "term", "purpose", "income"]], columns=["purpose"], dtype=int)
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, d.default, d.index, test_size=.3, random_state=SEED, stratify=d.default)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
gb = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, random_state=SEED).fit(Xtr, ytr)
res = []
for name, m in [("Logistic regression", lr), ("XGBoost", gb)]:
    p = m.predict_proba(Xte)[:, 1]; fpr, tpr, _ = roc_curve(yte, p)
    res.append(dict(model=name, auc=roc_auc_score(yte, p), gini=2 * roc_auc_score(yte, p) - 1,
                    ks=(tpr - fpr).max(), brier=brier_score_loss(yte, p), mean_pd=p.mean(), actual_dr=yte.mean()))
mc = pd.DataFrame(res); mc.to_csv("outputs/model_comparison.csv", index=False); print(mc.round(4).to_string(index=False))
coef = pd.Series(lr[-1].coef_[0], index=Xtr.columns).sort_values(); coef.to_csv("outputs/logit_coefficients_standardised.csv")

# ---------- 3. Expected loss, risk-based pricing (test set = held-out portfolio) ----------
b = d.loc[ite].copy(); b["pd"] = lr.predict_proba(Xte)[:, 1]
lgd_tab = d.loc[itr][d.loc[itr].default == 1].groupby("purpose").lgd_real.mean()
b["lgd"] = b.purpose.map(lgd_tab); b["el"] = b.pd * b.lgd * b.ead
b["el_rate"] = b.el / b.ead / (b.term / 12)                   # annualised expected loss rate (simple)
ul = np.sqrt(b.pd * (1 - b.pd)) * b.lgd                       # unexpected-loss proxy per unit EAD
b["required_rate"] = FUNDING + b.el_rate + OPEX + MARGIN + CAP_K * ul / (b.term / 12) * 0.1
b["rate_gap"] = b.rate_current - b.required_rate
b["interest_income"] = b.rate_current * b.ead * (b.term / 12) * 0.5   # avg-balance approx
b["funding_opex"] = (FUNDING + OPEX) * b.ead * (b.term / 12) * 0.5
b["net_result"] = b.interest_income - b.funding_opex - b.loss_real
b["band"] = pd.cut(b.pd, [0, .03, .06, .10, .15, 1], labels=["A", "B", "C", "D", "E"])
seg = b.groupby("band", observed=True).agg(loans=("pd", "size"), ead=("ead", "sum"), avg_pd=("pd", "mean"),
        actual_dr=("default", "mean"), expected_loss=("el", "sum"), realised_loss=("loss_real", "sum"),
        avg_rate=("rate_current", "mean"), avg_required=("required_rate", "mean"), net_result=("net_result", "sum")).reset_index()
seg["action"] = np.where((seg.net_result < 0) | (seg.avg_rate < seg.avg_required), "Re-price / tighten",
                         np.where(seg.avg_rate > seg.avg_required + .015, "Grow", "Maintain"))
seg.to_csv("outputs/risk_band_profitability.csv", index=False); print(seg.round(4).to_string(index=False))
seg2 = b.groupby("purpose").agg(loans=("pd", "size"), avg_pd=("pd", "mean"), actual_dr=("default", "mean"),
        expected_loss=("el", "sum"), realised_loss=("loss_real", "sum"), net_result=("net_result", "sum")).reset_index()
seg2.to_csv("outputs/purpose_profitability.csv", index=False)
b.drop(columns=["pd_true"]).to_csv("outputs/loan_level_pricing.csv", index=False)
print(f"Portfolio: EAD={b.ead.sum():,.0f} EL={b.el.sum():,.0f} ({b.el.sum()/b.ead.sum():.2%}) realised={b.loss_real.sum():,.0f}")

# ---------- 4. Portfolio loss distribution (one-factor Gaussian copula) ----------
pdv, lgdv, eadv = b.pd.values, b.lgd.values, b.ead.values
thr = norm.ppf(pdv); n_sims = 4000; losses = np.empty(n_sims)
for i in range(n_sims):
    zs = rng.standard_normal(); e = rng.standard_normal(len(pdv))
    dflt = (np.sqrt(RHO) * zs + np.sqrt(1 - RHO) * e) < thr
    losses[i] = (dflt * np.clip(rng.normal(lgdv, .1), .05, 1) * eadv).sum()
EL, var995, es = losses.mean(), np.percentile(losses, 99.5), losses[losses >= np.percentile(losses, 99.5)].mean()

# ---------- 5. Aggregate excess-of-loss (stop-loss) on credit losses ----------
att, lim = 1.25 * EL, np.percentile(losses, 99.5) - 1.25 * EL
layer = np.clip(losses - att, 0, lim); el_l, sd_l = layer.mean(), layer.std()
prem = (el_l + REINS_LOAD * sd_l) / (1 - REINS_EXP)
summ = {"EAD": eadv.sum(), "Expected loss": EL, "Loss std dev": losses.std(), "VaR 99.5%": var995, "Expected shortfall 99.5%": es,
        "Layer attachment (1.25 x EL)": att, "Layer limit (to VaR 99.5%)": lim, "P(layer hit)": (losses > att).mean(),
        "Expected layer loss": el_l, "Reinsurance premium": prem, "Rate on line": prem / lim, "Net std dev after cover": (losses - layer).std()}
pd.Series(summ).rename("value").to_csv("outputs/portfolio_and_reinsurance_summary.csv"); print(pd.Series(summ).round(4))
