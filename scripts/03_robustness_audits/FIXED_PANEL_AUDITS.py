"""
================================================================================
FIXED-STATION-PANEL AUDITS  --  composition-confound test
================================================================================
Reviewer concern (round 3):
    The Central regional mean is built from 5 stations in 1950 rising to 10 by
    2000, and the late-joining stations are ~14% wetter. The temporal
    "strengthening" of the DMI effect (Section 5.5) could therefore be a
    compositional artifact rather than genuine climate intensification.

This script removes the confound by holding the station panel FIXED:

  AUDIT 1  (Fixed-6 temporal):
      Re-run the rolling 30-yr window audit using ONLY the six stations present
      continuously since 1948-1953:
          Dhaka, Faridpur, Barisal, Khulna, Jessore, Satkhira
      (Dhaka starts 1953, so the first window effectively uses 5-6; all six are
       present from 1953 onward, i.e. for every window midpoint >= 1968.)
      If the DMI coefficient still strengthens toward the present on this fixed
      panel, the strengthening is REAL. If it flattens, it was compositional.

  AUDIT 2  (Fixed-8 vs all-available, 1982-2013 headline):
      Refit the headline 1982-2013 model on (a) all stations available in the
      window and (b) the fixed 8 present since 1982, and compare the DMI
      coefficient and the 100-yr design depth. Negligible difference => the
      headline engineering result is not driven by composition.

Same GEV engine, same OND preprocessing, same standardization convention.
No new data sources, no changed formulas.

Outputs:
  - console tables for both audits
  - fixed6_temporal.png       (coefficient + recurrence-period paths, fixed panel)
  - fixed_panel_audits.csv
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

OND_MONTHS = [10, 11, 12]

# Stations present continuously from the 1940s-50s (the "fixed-6" panel)
FIXED6 = ["Dhaka", "Faridpur", "Barisal", "Khulna", "Jessore", "Satkhira"]
# Stations present from 1982 (the "fixed-8" panel for the headline window)
FIXED8 = ["Dhaka", "Faridpur", "Madaripur", "Barisal", "Khulna",
          "Jessore", "Satkhira", "Bhola"]
# Full Central set (for the all-available comparison)
CENTRAL10 = ["Dhaka","Faridpur","Madaripur","Tangail","Barisal",
             "Khulna","Mongla","Jessore","Satkhira","Bhola"]

WINDOWS = [(1950, 1979), (1955, 1984), (1960, 1989), (1965, 1994),
           (1970, 1999), (1975, 2004), (1980, 2009), (1984, 2013)]

DESIGN_T  = 100
NEG_STATE = -2.0

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

def nll_1cov(p, y, c):
    b0, b1, ls, xi = p
    sigma = np.exp(ls); mu = b0 + b1*c
    s = 0.0
    for i in range(len(y)):
        v = gev_logpdf(y[i], mu[i], sigma, xi)
        if not np.isfinite(v):
            return 1e10
        s += v
    return -s

def fit_1cov(y, c):
    mu0, s0 = np.median(y), np.std(y)
    best, bll = None, np.inf
    for g in (-40, -20, 0, 20):
        r = minimize(nll_1cov, [mu0, g, np.log(s0), 0.05], args=(y, c),
                     method="Nelder-Mead",
                     options={"xatol":1e-6,"fatol":1e-6,"maxiter":8000})
        if r.fun < bll:
            bll, best = r.fun, r
    b0, b1, ls, xi = best.x
    return b0, b1, np.exp(ls), xi

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
def regional_max(df, stations):
    """OND annual-max monthly rainfall, mean over the given fixed station set."""
    d = df[df["Station"].isin(stations) & df["Month"].isin(OND_MONTHS)]
    sm = d.groupby(["Station","Year"])["Monthly_Total"].max().reset_index()
    reg = sm.groupby("Year")["Monthly_Total"].mean().reset_index().rename(
            columns={"Monthly_Total":"Rainfall"})
    # also return the per-year station count for transparency
    cnt = sm.groupby("Year")["Station"].nunique().reset_index().rename(
            columns={"Station":"nstn"})
    return reg.merge(cnt, on="Year")

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
    raw = raw[raw["Anom"] > -90]
    raw["Year"] = raw["Year"].astype(int); raw["Month"] = raw["Month"].astype(int)
    ond = raw[raw["Month"].isin(OND_MONTHS)]
    return ond.groupby("Year")["Anom"].mean().reset_index().rename(columns={"Anom":"NINO"})

# ---------------------------------------------------------- MAIN ---------------
def main():
    df  = pd.read_csv(RAINFALL_FILE)
    dmi = load_dmi()

    # ============================ AUDIT 1: FIXED-6 TEMPORAL ====================
    reg6 = regional_max(df, FIXED6)
    data6 = reg6.merge(dmi, on="Year").sort_values("Year").reset_index(drop=True)

    print("="*96)
    print("AUDIT 1 -- FIXED-6 TEMPORAL  (Dhaka, Faridpur, Barisal, Khulna, Jessore, Satkhira)")
    print("DMI-only model; rolling 30-yr windows. Tests whether strengthening survives a fixed panel.")
    print("="*96)

    rows = []
    for (y0, y1) in WINDOWS:
        w = data6[(data6.Year >= y0) & (data6.Year <= y1)].copy()
        if len(w) < 25:
            print(f"  [skip] {y0}-{y1}: n={len(w)}")
            continue
        y = w["Rainfall"].values.astype(float)
        d = (w["DMI"].values - w["DMI"].mean()) / w["DMI"].std(ddof=0)
        b0, b1, sg, xi = fit_1cov(y, d)
        rl100 = gev_quantile(DESIGN_T, b0, sg, xi)
        mu_neg = b0 + b1*NEG_STATE
        p_neg = 1.0 - gev_cdf(rl100, mu_neg, sg, xi)
        T_eq = 1.0/p_neg if p_neg > 0 else np.inf
        stn_min, stn_max = int(w["nstn"].min()), int(w["nstn"].max())
        rows.append({"Window":f"{y0}-{y1}","Midpoint":(y0+y1)/2.0,
                     "nstn":f"{stn_min}-{stn_max}","b_DMI":b1,
                     "RL100":rl100,"T_eq_neg2s":T_eq})

    res6 = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(res6.to_string(index=False))

    b = res6["b_DMI"].values
    T = res6["T_eq_neg2s"].values
    mids = res6["Midpoint"].values
    early_b = np.mean(b[:max(1,len(b)//2)])
    late_b  = np.mean(b[len(b)//2:])
    print(f"\n  b_DMI early-half mean: {early_b:+.1f} | late-half mean: {late_b:+.1f} "
          f"| ratio: {late_b/early_b:.2f}x")
    print(f"  T_eq(-2s): min {T.min():.0f}, max {T.max():.0f}, mean {T.mean():.0f}, "
          f"slope {np.polyfit(mids,T,1)[0]:+.2f} yr/yr")
    if abs(late_b) > 1.3*abs(early_b):
        print("  >>> Strengthening PERSISTS on the fixed-6 panel: NOT a compositional artifact.")
    elif abs(late_b) > 1.1*abs(early_b):
        print("  >>> Mild strengthening on fixed-6: partly real, partly compositional. Soften wording.")
    else:
        print("  >>> Strengthening DISAPPEARS on fixed-6: it was largely compositional. Re-word Section 5.5.")

    # ============================ AUDIT 2: FIXED-8 vs ALL, 1982-2013 ===========
    print("\n" + "="*96)
    print("AUDIT 2 -- HEADLINE 1982-2013: fixed-8 panel vs all-available")
    print("="*96)

    def fit_window(stations, y0=1982, y1=2013):
        reg = regional_max(df, stations)
        dd = reg.merge(dmi, on="Year")
        dd = dd[(dd.Year >= y0) & (dd.Year <= y1)]
        y = dd["Rainfall"].values.astype(float)
        d = (dd["DMI"].values - dd["DMI"].mean()) / dd["DMI"].std(ddof=0)
        b0, b1, sg, xi = fit_1cov(y, d)
        rl100 = gev_quantile(DESIGN_T, b0, sg, xi)
        mu_neg = b0 + b1*NEG_STATE
        T_eq = 1.0/(1.0 - gev_cdf(rl100, mu_neg, sg, xi))
        return len(y), b1, b0, sg, xi, rl100, T_eq

    for label, stns in [("All available (8->10)", CENTRAL10),
                        ("Fixed-8 (since 1982)", FIXED8)]:
        n, b1, b0, sg, xi, rl, teq = fit_window(stns)
        print(f"  {label:24s}: n={n}, b_DMI={b1:+.2f}, 100yrRL={rl:.0f} mm, "
              f"T_eq(-2s)={teq:.0f} yr (b0={b0:.1f}, sigma={sg:.1f}, xi={xi:.3f})")

    print("\n  If b_DMI and 100yrRL match closely across the two rows, the headline")
    print("  engineering result is robust to the 8->10 composition change.")

    # save
    res6.to_csv("fixed_panel_audits.csv", index=False)
    print("\nSaved: fixed_panel_audits.csv")

    # ---------------------------------------------------- PLOT (fixed-6) ----
    plt.rcParams.update({"font.family":"serif","font.size":10.5,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":0.25,"savefig.dpi":200,
                         "savefig.bbox":"tight"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax1.plot(mids, res6["b_DMI"], "-o", color="#1f5fa8", lw=2, ms=6)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("Calibration-window midpoint (year)")
    ax1.set_ylabel("DMI coefficient (mm per $\\sigma$), fixed-6 panel")
    ax1.set_title("(a) Coefficient path on FIXED 6-station panel")
    ax2.axhspan(35, 55, color="#3a8c5f", alpha=0.10, label="reference band 35-55 yr")
    ax2.axhline(100, color="k", ls="--", lw=1.0); ax2.text(mids[0], 103, "nominal 100-yr", fontsize=8)
    ax2.plot(mids, T, "-o", color="#222222", lw=2, ms=7, mfc="#1f5fa8", zorder=3)
    for m, t in zip(mids, T):
        ax2.annotate(f"{t:.0f}", (m, t), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8.5)
    fit = np.polyfit(mids, T, 1)
    ax2.plot(mids, np.polyval(fit, mids), "--", color="#c44e52", lw=1.3,
             label=f"trend {fit[0]:+.2f} yr/yr")
    ax2.set_xlabel("Calibration-window midpoint (year)")
    ax2.set_ylabel("Equiv. recurrence period under $-2\\sigma$ IOD (yr)")
    ax2.set_title("(b) Recurrence compression, fixed-6 panel")
    ax2.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig("fixed6_temporal.png")
    print("Saved: fixed6_temporal.png\nDONE.")

if __name__ == "__main__":
    main()
