"""
================================================================================
TEMPORAL STABILITY AUDIT  --  SINGLE-COVARIATE VERSION
================================================================================
Companion to the dual-covariate audit. Purpose: determine whether the steep
b_DMI = -31 / b_ENSO = +0.4 split seen in the 1984-2013 dual fit reflects a
genuine STRENGTHENING of the IOD effect, or a COLLINEARITY REASSIGNMENT between
two correlated indices.

Method:
  For each rolling 30-yr window, fit THREE models separately:
      M1: mu = b0 + b1*DMI            (single covariate -- the paper's headline model)
      M2: mu = b0 + b1*ENSO           (single covariate)
      M3: mu = b0 + b1*DMI + b2*ENSO  (dual -- for comparison with the first audit)
  Also report the within-window correlation r(DMI, ENSO), which drives any
  collinearity instability.

Diagnostic logic:
  * If the SINGLE-COVARIATE DMI trajectory (b1 in M1) is smooth and trends gently,
    while only the DUAL split is unstable  -> the -31 is partly collinearity.
  * If even the single-covariate DMI coefficient jumps to ~ -31 in the last window
    -> the strengthening is real, not an artifact.
  * r(DMI,ENSO) rising toward the recent windows corroborates the collinearity story.

No new data, no new formulas -- same preprocessing and GEV engine as the paper.
Outputs:
  - console table (per window: r, M1 b_DMI, M2 b_ENSO, M3 b_DMI, M3 b_ENSO,
                   100-yr RL and T_eq(-2s) from the DMI-ONLY model M1)
  - temporal_stability_singlecov.png  (two panels: coefficient paths; T_eq path)
  - temporal_stability_singlecov.csv
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------------------------------------------------ CONFIG ----
RAINFALL_FILE = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\rainfall_data_bangladesh_1948_to_2014.csv"
DMI_FILE      = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\DMI_data.txt"
NINO_FILE     = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\nino34_monthly.txt"

CENTRAL_STATIONS = ["Dhaka", "Faridpur", "Madaripur", "Tangail", "Barisal",
                    "Khulna", "Mongla", "Jessore", "Satkhira", "Bhola"]
OND_MONTHS = [10, 11, 12]

WINDOWS = [(1950, 1979), (1955, 1984), (1960, 1989), (1965, 1994),
           (1970, 1999), (1975, 2004), (1980, 2009), (1984, 2013)]

DESIGN_T  = 100
NEG_STATE = -2.0    # strong negative IOD for recurrence compression

# -------------------------------------------------------------- GEV TOOLKIT ---
def gev_logpdf(x, mu, sigma, xi):
    if sigma <= 0:
        return -np.inf
    z = (x - mu) / sigma
    if abs(xi) < 1e-6:
        return -np.log(sigma) - z - np.exp(-z)
    t = 1.0 + xi * z
    if np.any(t <= 0):
        return -np.inf
    return -np.log(sigma) - (1.0/xi + 1.0)*np.log(t) - t**(-1.0/xi)

def nll_1cov(params, y, cov):
    b0, b1, log_sigma, xi = params
    sigma = np.exp(log_sigma)
    mu = b0 + b1*cov
    s = 0.0
    for i in range(len(y)):
        v = gev_logpdf(y[i], mu[i], sigma, xi)
        if not np.isfinite(v):
            return 1e10
        s += v
    return -s

def nll_2cov(params, y, c1, c2):
    b0, b1, b2, log_sigma, xi = params
    sigma = np.exp(log_sigma)
    mu = b0 + b1*c1 + b2*c2
    s = 0.0
    for i in range(len(y)):
        v = gev_logpdf(y[i], mu[i], sigma, xi)
        if not np.isfinite(v):
            return 1e10
        s += v
    return -s

def fit_1cov(y, cov):
    mu0, s0 = np.median(y), np.std(y)
    best, bll = None, np.inf
    for g in (-40, -20, 0, 20):
        r = minimize(nll_1cov, [mu0, g, np.log(s0), 0.05], args=(y, cov),
                     method="Nelder-Mead",
                     options={"xatol":1e-6,"fatol":1e-6,"maxiter":8000})
        if r.fun < bll:
            bll, best = r.fun, r
    b0, b1, ls, xi = best.x
    return b0, b1, np.exp(ls), xi

def fit_2cov(y, c1, c2):
    mu0, s0 = np.median(y), np.std(y)
    best, bll = None, np.inf
    for g1 in (-40, -20, 0, 20):
        for g2 in (-40, -20, 0, 20):
            r = minimize(nll_2cov, [mu0, g1, g2, np.log(s0), 0.05], args=(y, c1, c2),
                         method="Nelder-Mead",
                         options={"xatol":1e-6,"fatol":1e-6,"maxiter":8000})
            if r.fun < bll:
                bll, best = r.fun, r
    b0, b1, b2, ls, xi = best.x
    return b0, b1, b2, np.exp(ls), xi

def gev_quantile(T, mu, sigma, xi):
    p = 1.0 - 1.0/T
    if abs(xi) < 1e-6:
        return mu - sigma*np.log(-np.log(p))
    return mu + (sigma/xi)*((-np.log(p))**(-xi) - 1.0)

def gev_cdf(x, mu, sigma, xi):
    z = (x - mu)/sigma
    if abs(xi) < 1e-6:
        return np.exp(-np.exp(-z))
    t = 1.0 + xi*z
    if np.any(t <= 0):
        return np.nan
    return np.exp(-t**(-1.0/xi))

# --------------------------------------------------------- DATA LOADING --------
def load_rainfall():
    df = pd.read_csv(RAINFALL_FILE)
    df = df[df["Station"].isin(CENTRAL_STATIONS)]
    df = df[df["Month"].isin(OND_MONTHS)]
    station_max = df.groupby(["Station","Year"])["Monthly_Total"].max().reset_index()
    regional = (station_max.groupby("Year")["Monthly_Total"].mean().reset_index()
                  .rename(columns={"Monthly_Total":"Rainfall"}))
    return regional

def load_dmi():
    rows = []
    with open(DMI_FILE) as f:
        for line in f:
            p = line.split()
            if len(p) >= 13:
                try:
                    yr = int(p[0])
                    ond = np.mean([float(p[10]), float(p[11]), float(p[12])])
                    rows.append([yr, ond])
                except ValueError:
                    continue
    return pd.DataFrame(rows, columns=["Year","DMI"])

def load_nino():
    raw = pd.read_csv(NINO_FILE, sep=r"\s+", header=None,
                      names=["Year","Month","Total","Clim","Anom"])
    for c in ["Year","Month","Anom"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["Year","Month","Anom"])
    raw = raw[raw["Anom"] > -90]                      # drop -99.x sentinels if any
    raw["Year"] = raw["Year"].astype(int)
    raw["Month"] = raw["Month"].astype(int)
    ond = raw[raw["Month"].isin(OND_MONTHS)]
    return ond.groupby("Year")["Anom"].mean().reset_index().rename(columns={"Anom":"NINO"})

# ---------------------------------------------------------- MAIN ---------------
def main():
    rain, dmi, nino = load_rainfall(), load_dmi(), load_nino()
    data = rain.merge(dmi, on="Year").merge(nino, on="Year").sort_values("Year").reset_index(drop=True)
    print(f"Merged record: {data.Year.min()}-{data.Year.max()} ({len(data)} years)\n")

    rows = []
    for (y0, y1) in WINDOWS:
        w = data[(data.Year>=y0)&(data.Year<=y1)].copy()
        if len(w) < 25:
            continue
        y = w["Rainfall"].values.astype(float)
        d = (w["DMI"].values  - w["DMI"].mean())  / w["DMI"].std(ddof=0)
        e = (w["NINO"].values - w["NINO"].mean()) / w["NINO"].std(ddof=0)
        r_de = np.corrcoef(d, e)[0, 1]

        # M1: DMI only (headline model)
        b0_1, b1_dmi, sig1, xi1 = fit_1cov(y, d)
        # M2: ENSO only
        b0_2, b1_enso, sig2, xi2 = fit_1cov(y, e)
        # M3: dual (for comparison)
        b0_3, b3_dmi, b3_enso, sig3, xi3 = fit_2cov(y, d, e)

        # engineering quantities from the DMI-ONLY model (the one the paper reports)
        rl100 = gev_quantile(DESIGN_T, b0_1, sig1, xi1)          # neutral state = b0_1
        mu_neg = b0_1 + b1_dmi*NEG_STATE
        p_neg = 1.0 - gev_cdf(rl100, mu_neg, sig1, xi1)
        T_eq = 1.0/p_neg if p_neg > 0 else np.inf

        rows.append({
            "Window": f"{y0}-{y1}", "Midpoint": (y0+y1)/2.0, "n": len(w),
            "r_DMI_ENSO": r_de,
            "M1_bDMI": b1_dmi, "M2_bENSO": b1_enso,
            "M3_bDMI": b3_dmi, "M3_bENSO": b3_enso,
            "RL100_M1": rl100, "T_eq_neg2s_M1": T_eq, "xi_M1": xi1
        })

    res = pd.DataFrame(rows)
    res.to_csv("temporal_stability_singlecov.csv", index=False)

    # ---------------------------------------------------- TABLE ----
    print("="*108)
    print("SINGLE- vs DUAL-COVARIATE COEFFICIENTS ACROSS ROLLING WINDOWS")
    print("(M1 = DMI-only [headline], M2 = ENSO-only, M3 = dual)")
    print("="*108)
    disp = res.copy()
    show = disp[["Window","r_DMI_ENSO","M1_bDMI","M2_bENSO","M3_bDMI","M3_bENSO",
                 "RL100_M1","T_eq_neg2s_M1"]]
    show.columns = ["Window","r(DMI,ENSO)","M1 bDMI","M2 bENSO","M3 bDMI","M3 bENSO",
                    "100yrRL(M1)","Teq(-2s,M1)"]
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(show.to_string(index=False))
    print("\nSaved: temporal_stability_singlecov.csv")

    # ---------------------------------------------- DIAGNOSTIC ----
    m1 = res["M1_bDMI"].values
    m3 = res["M3_bDMI"].values
    mids = res["Midpoint"].values
    last_jump_single = m1[-1] / np.mean(m1[:-1])
    last_jump_dual   = m3[-1] / np.mean(m3[:-1])
    T = res["T_eq_neg2s_M1"].values
    cv_T = np.std(T, ddof=1)/np.mean(T)
    slope_T = np.polyfit(mids, T, 1)[0]

    print("\n" + "="*108)
    print("DIAGNOSIS: strengthening vs collinearity")
    print("="*108)
    print(f"  DMI-only (M1) coefficient, final window     : {m1[-1]:+.2f}")
    print(f"  DMI-only (M1) coefficient, mean of earlier  : {np.mean(m1[:-1]):+.2f}  "
          f"(final/earlier = {last_jump_single:.2f}x)")
    print(f"  Dual    (M3) coefficient, final window      : {m3[-1]:+.2f}")
    print(f"  Dual    (M3) coefficient, mean of earlier   : {np.mean(m3[:-1]):+.2f}  "
          f"(final/earlier = {last_jump_dual:.2f}x)")
    print(f"  r(DMI,ENSO) early vs final                  : {res['r_DMI_ENSO'].iloc[0]:+.2f}"
          f"  ->  {res['r_DMI_ENSO'].iloc[-1]:+.2f}")
    print(f"  T_eq(-2s) from DMI-only model: min {T.min():.0f}, max {T.max():.0f}, "
          f"mean {T.mean():.0f}, CV {cv_T*100:.1f}%, slope {slope_T:+.2f} yr/yr")

    if last_jump_dual > 1.6 and last_jump_single < 1.4:
        verdict = ("The final-window spike is LARGELY COLLINEARITY: the DMI-only "
                   "coefficient is comparatively stable, so the dual model's -31 "
                   "is inflated by reassignment of shared variance away from ENSO. "
                   "Report the DMI-only trajectory as the honest stability picture.")
    elif last_jump_single >= 1.4:
        verdict = ("The strengthening is REAL: even the single-covariate DMI "
                   "coefficient rises markedly in the recent window, so the "
                   "intensification is not merely a collinearity artifact.")
    else:
        verdict = ("MIXED: modest real strengthening plus some collinearity "
                   "amplification in the dual model. Report the DMI-only range and "
                   "note the dual-model instability explicitly.")
    print("\n  VERDICT: " + verdict)
    print("="*108)

    # ------------------------------------------------- PLOT (2 panels) ----
    plt.rcParams.update({"font.family":"serif","font.size":10.5,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":0.25,"savefig.dpi":200,
                         "savefig.bbox":"tight"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    ax1.plot(mids, res["M1_bDMI"], "-o", color="#1f5fa8", lw=2, ms=6, label="DMI-only (M1)")
    ax1.plot(mids, res["M3_bDMI"], "--s", color="#1f5fa8", lw=1.6, ms=5, alpha=0.6, label="DMI in dual (M3)")
    ax1.plot(mids, res["M2_bENSO"], "-o", color="#c44e52", lw=2, ms=6, label="ENSO-only (M2)")
    ax1.plot(mids, res["M3_bENSO"], "--s", color="#c44e52", lw=1.6, ms=5, alpha=0.6, label="ENSO in dual (M3)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("Calibration-window midpoint (year)")
    ax1.set_ylabel("Location-parameter coefficient (mm per $\\sigma$)")
    ax1.set_title("(a) Coefficient paths: single vs dual")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")

    ax2.axhspan(35, 55, color="#3a8c5f", alpha=0.10, label="reference band 35-55 yr")
    ax2.plot(mids, T, "-o", color="#222222", lw=2, ms=7, mfc="#1f5fa8", zorder=3)
    for m, t in zip(mids, T):
        ax2.annotate(f"{t:.0f}", (m, t), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9)
    fit = np.polyfit(mids, T, 1)
    ax2.plot(mids, np.polyval(fit, mids), "--", color="#c44e52", lw=1.3,
             label=f"trend {slope_T:+.2f} yr/yr")
    ax2.set_xlabel("Calibration-window midpoint (year)")
    ax2.set_ylabel("Equivalent recurrence period under\n$-2\\sigma$ IOD, DMI-only model (yr)")
    ax2.set_title("(b) Recurrence compression (headline model)")
    ax2.legend(frameon=False, fontsize=8.5, loc="best")

    fig.tight_layout()
    fig.savefig("temporal_stability_singlecov.png")
    print("\nSaved: temporal_stability_singlecov.png\nDONE.")

if __name__ == "__main__":
    main()
