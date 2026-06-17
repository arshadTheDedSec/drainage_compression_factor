"""
================================================================================
TEMPORAL STABILITY AUDIT OF ENGINEERING DESIGN IMPLICATIONS
================================================================================
Reviewer concern addressed:
    "Your record ends in 2013. How do we know the recurrence-compression and
     infrastructure-risk conclusions are not artifacts of the chosen period?"

Method:
    Rolling 30-year calibration windows, 5-year increments (1950-1979 ... 1984-2013).
    For each window the SAME final model, preprocessing, and engineering formulas
    from the paper are re-fitted, and the headline engineering quantities are
    recomputed. No new data, no new model structure, no altered formulas.

Outputs:
    1. Console summary table  (Window | b_DMI | b_ENSO | 100-yr RL | T_eq(-2s) | P_fail)
    2. temporal_stability.png (equivalent recurrence period vs window midpoint)
    3. temporal_stability_table.csv
    4. Trend assessment (min, max, mean, CV, linear slope) + automatic A/B/C verdict
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------------------------------------------------ CONFIG ----
# >>> EDIT THESE PATHS to match your machine (same files as the main analysis) <<<
RAINFALL_FILE = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\rainfall_data_bangladesh_1948_to_2014.csv"
DMI_FILE      = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\DMI_data.txt"
NINO_FILE     = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\nino34_monthly.txt"

CENTRAL_STATIONS = ["Dhaka", "Faridpur", "Madaripur", "Tangail", "Barisal",
                    "Khulna", "Mongla", "Jessore", "Satkhira", "Bhola"]
OND_MONTHS = [10, 11, 12]

# Rolling windows (30-yr, 5-yr step) exactly as specified
WINDOWS = [(1950, 1979), (1955, 1984), (1960, 1989), (1965, 1994),
           (1970, 1999), (1975, 2004), (1980, 2009), (1984, 2013)]

DESIGN_T   = 100      # design return period (years)
LIFE       = 50       # service life (years) for failure probability
RL_STATE   = 0.0      # stationary design RL evaluated at neutral state
NEG_STATE  = -2.0     # strong negative IOD (-2 sigma) for recurrence compression

# ------------------------------------------------------------- GEV TOOLKIT ----
def gev_logpdf(x, mu, sigma, xi):
    if sigma <= 0:
        return -np.inf
    z = (x - mu) / sigma
    if abs(xi) < 1e-6:                       # Gumbel limit
        return -np.log(sigma) - z - np.exp(-z)
    t = 1.0 + xi * z
    if np.any(t <= 0):
        return -np.inf
    return -np.log(sigma) - (1.0/xi + 1.0)*np.log(t) - t**(-1.0/xi)

def gev_nll(params, y, cov_dmi, cov_enso):
    """Negative log-likelihood for mu = b0 + b1*DMI + b2*ENSO, constant sigma, xi."""
    b0, b1, b2, log_sigma, xi = params
    sigma = np.exp(log_sigma)
    mu = b0 + b1*cov_dmi + b2*cov_enso
    ll = 0.0
    for i in range(len(y)):
        v = gev_logpdf(y[i], mu[i], sigma, xi)
        if not np.isfinite(v):
            return 1e10
        ll += v
    return -ll

def fit_gev_dual(y, cov_dmi, cov_enso):
    """Multi-start ML fit of the dual-covariate GEV (the paper's M3 structure)."""
    mu0, s0 = np.median(y), np.std(y)
    best, best_ll = None, np.inf
    for b1g in (-40, -20, 0, 20):
        for b2g in (-40, -20, 0, 20):
            x0 = [mu0, b1g, b2g, np.log(s0), 0.05]
            r = minimize(gev_nll, x0, args=(y, cov_dmi, cov_enso),
                         method="Nelder-Mead",
                         options={"xatol":1e-6, "fatol":1e-6, "maxiter":8000})
            if r.fun < best_ll:
                best_ll, best = r.fun, r
    b0, b1, b2, log_sigma, xi = best.x
    return b0, b1, b2, np.exp(log_sigma), xi

def gev_quantile(T, mu, sigma, xi):
    """Return level for return period T (annual maxima)."""
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

# --------------------------------------------------------- DATA LOADING -------
def load_rainfall():
    """OND annual-maximum monthly rainfall, regional mean over Central stations."""
    df = pd.read_csv(RAINFALL_FILE)
    # --- adapt these column names if yours differ ---
    # expected columns: Station, Year, Month, Monthly_Total
    df = df[df["Station"].isin(CENTRAL_STATIONS)]
    df = df[df["Month"].isin(OND_MONTHS)]
    station_max = (df.groupby(["Station", "Year"])["Monthly_Total"]
                     .max().reset_index())
    regional = (station_max.groupby("Year")["Monthly_Total"]
                  .mean().reset_index()
                  .rename(columns={"Monthly_Total": "Rainfall"}))
    return regional  # columns: Year, Rainfall

def load_dmi():
    """OND-mean DMI. File assumed: year + 12 monthly columns (Jan..Dec)."""
    rows = []
    with open(DMI_FILE) as f:
        for line in f:
            p = line.split()
            if len(p) >= 13:
                try:
                    yr = int(p[0])
                    ond = np.mean([float(p[10]), float(p[11]), float(p[12])])  # Oct,Nov,Dec
                    rows.append([yr, ond])
                except ValueError:
                    continue
    return pd.DataFrame(rows, columns=["Year", "DMI"])

def load_nino():
    """OND-mean Nino3.4 anomaly. ERSSTv5 5-col monthly file: Year Month Total Clim Anom."""
    raw = pd.read_csv(NINO_FILE, delim_whitespace=True, header=None,
                      names=["Year", "Month", "Total", "Clim", "Anom"])
    ond = raw[raw["Month"].isin(OND_MONTHS)]
    nino = ond.groupby("Year")["Anom"].mean().reset_index().rename(columns={"Anom": "NINO"})
    return nino

# ---------------------------------------------------------- MAIN ANALYSIS -----
def main():
    rain = load_rainfall()
    dmi  = load_dmi()
    nino = load_nino()

    data = rain.merge(dmi, on="Year").merge(nino, on="Year").sort_values("Year")
    data = data.reset_index(drop=True)
    print(f"Merged record: {data.Year.min()}-{data.Year.max()} "
          f"({len(data)} years)\n")

    results = []
    for (y0, y1) in WINDOWS:
        w = data[(data.Year >= y0) & (data.Year <= y1)].copy()
        n = len(w)
        if n < 25:
            print(f"  [skip] {y0}-{y1}: only {n} yrs available")
            continue

        y = w["Rainfall"].values.astype(float)
        # standardize covariates WITHIN the window (same convention as the paper)
        d = (w["DMI"].values  - w["DMI"].mean())  / w["DMI"].std(ddof=0)
        e = (w["NINO"].values - w["NINO"].mean()) / w["NINO"].std(ddof=0)

        b0, b1, b2, sigma, xi = fit_gev_dual(y, d, e)

        # --- engineering quantities (identical formulas to the paper) ---
        mu_neutral = b0 + b1*RL_STATE + b2*RL_STATE
        rl100 = gev_quantile(DESIGN_T, mu_neutral, sigma, xi)         # stationary 100-yr design depth

        # equivalent recurrence of that design depth under strong negative IOD (-2 sigma DMI)
        mu_neg = b0 + b1*NEG_STATE + b2*0.0                           # ENSO held at neutral
        p_exc_neg = 1.0 - gev_cdf(rl100, mu_neg, sigma, xi)
        T_eq_neg = 1.0/p_exc_neg if p_exc_neg > 0 else np.inf

        # 50-yr lifetime failure probability under equal-time stress cycling (-2s/0/+2s on DMI),
        # ENSO neutral -- same assumption family as the manuscript's stress-cycling case
        p_list = []
        for s in (-2.0, 0.0, 2.0):
            mu_s = b0 + b1*s + b2*0.0
            p_list.append(1.0 - gev_cdf(rl100, mu_s, sigma, xi))
        # equal time in each state over the service life
        surv = 1.0
        for p in p_list:
            surv *= (1.0 - p)**(LIFE/3.0)
        p_fail = 1.0 - surv

        mid = (y0 + y1)/2.0
        results.append({
            "Window": f"{y0}-{y1}", "Midpoint": mid, "n": n,
            "b_DMI": b1, "b_ENSO": b2, "RL100": rl100,
            "T_eq_neg2sigma": T_eq_neg, "P_fail50": p_fail*100.0,
            "sigma": sigma, "xi": xi
        })

    res = pd.DataFrame(results)

    # ---------------------------------------------------- 1. SUMMARY TABLE ----
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("="*92)
    print("SUMMARY TABLE  (rolling 30-yr windows; engineering quantities)")
    print("="*92)
    show = res[["Window","b_DMI","b_ENSO","RL100","T_eq_neg2sigma","P_fail50"]].copy()
    show.columns = ["Window","b_DMI","b_ENSO","100yr_RL(mm)","T_eq(-2s)(yr)","P_fail50(%)"]
    print(show.to_string(index=False))
    print()
    res.to_csv("temporal_stability_table.csv", index=False)
    print("Saved: temporal_stability_table.csv")

    # ------------------------------------------------- 3. TREND ASSESSMENT ----
    Tser = res["T_eq_neg2sigma"].values
    mids = res["Midpoint"].values
    tmin, tmax, tmean = Tser.min(), Tser.max(), Tser.mean()
    cv = Tser.std(ddof=1)/tmean
    slope = np.polyfit(mids, Tser, 1)[0]   # years of T_eq per calendar year

    print("\n" + "="*92)
    print("TREND ASSESSMENT  (equivalent recurrence period under -2 sigma IOD)")
    print("="*92)
    print(f"  minimum                : {tmin:6.1f} yr")
    print(f"  maximum                : {tmax:6.1f} yr")
    print(f"  mean                   : {tmean:6.1f} yr")
    print(f"  coefficient of variation: {cv:6.3f}  ({cv*100:.1f}%)")
    print(f"  linear trend slope     : {slope:+.3f} yr of T_eq per calendar year")
    print(f"  total drift over record: {slope*(mids[-1]-mids[0]):+.1f} yr")

    # automatic verdict (interpretation rules A/B/C)
    spread = tmax - tmin
    rel_slope = abs(slope*(mids[-1]-mids[0]))/tmean
    if cv < 0.20 and spread < 25:
        verdict = ("CASE A -- TEMPORALLY STABLE: the engineering implication does not "
                   "depend strongly on the calibration period.")
    elif rel_slope > 0.30 and cv < 0.40:
        verdict = ("CASE B -- PERSISTENT WITH EVOLVING MAGNITUDE: the compression "
                   "implication persists across all windows but its magnitude trends "
                   "through time.")
    elif cv >= 0.40:
        verdict = ("CASE C -- CALIBRATION-PERIOD DEPENDENT: recurrence periods vary "
                   "widely; the implication is sensitive to the chosen window.")
    else:
        verdict = ("INTERMEDIATE: the implication is broadly stable with a mild "
                   "temporal component; report both the range and the slope.")
    print("\n  VERDICT: " + verdict)
    print("="*92)

    # ----------------------------------------------- 2. STABILITY PLOT --------
    plt.rcParams.update({"font.family":"serif","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":0.25,"savefig.dpi":200,
                         "savefig.bbox":"tight"})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhspan(35, 55, color="#3a8c5f", alpha=0.10,
               label="reference stability band (35-55 yr)")
    ax.plot(mids, Tser, "-o", color="#1f5fa8", lw=2, ms=7, zorder=3)
    for m, t, lab in zip(mids, Tser, res["Window"]):
        ax.annotate(f"{t:.0f}", (m, t), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    # linear trend line
    fit = np.polyfit(mids, Tser, 1)
    ax.plot(mids, np.polyval(fit, mids), "--", color="#c44e52", lw=1.3,
            label=f"linear trend ({slope:+.2f} yr/yr)")
    ax.set_xlabel("Calibration-window midpoint (year)")
    ax.set_ylabel("Equivalent recurrence period of the\nstationary 100-yr design depth under -2$\\sigma$ IOD (yr)")
    ax.set_title("Temporal stability of the recurrence-compression implication")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.savefig("temporal_stability.png")
    print("\nSaved: temporal_stability.png")
    print("\nDONE.")

if __name__ == "__main__":
    main()
