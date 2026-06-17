"""
PAPER 2: FINAL CORRECT VERSION
Variable: OND Annual Maximum Monthly Rainfall
- Only October, November, December months
- Maximum monthly value within OND each year (proper block maxima)
- Physically motivated (IOD season) + Statistically valid (block maxima)
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FILE PATHS
# ============================================================
RAINFALL_FILE = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\rainfall_data_bangladesh_1948_to_2014.csv"
DMI_FILE      = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\DMI_data.txt"
Z500_FILE     = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\Z500_Tibetan_OND.csv"

# ============================================================
# REGION DEFINITIONS
# ============================================================
REGIONS = {
    'National': None,
    'Coastal': [
        "Cox's Bazar", "Chittagong", "Sandwip", "Sitakunda",
        "Teknaf", "Kutubdia", "Hatiya", "Khepupara", "Patuakhali",
        "Ambagan(Ctg)", "Rangamati", "Feni"
    ],
    'Northwestern': [
        "Rajshahi", "Bogra", "Rangpur", "Dinajpur",
        "Ishurdi", "chuadanga", "sydpur"
    ],
    'Northeastern': [
        "Sylhet", "Srimangal", "Comilla", "Mymensingh",
        "Chandpur", "Feni", "M.court"
    ],
    'Central': [
        "Dhaka", "Faridpur", "Madaripur", "Tangail",
        "Barisal", "Khulna", "Mongla", "Jessore",
        "Satkhira", "Bhola"
    ]
}

# ============================================================
# GEV FUNCTIONS
# ============================================================

def gev_logpdf(x, mu, sigma, xi):
    if sigma <= 0:
        return -np.inf
    z = (x - mu) / sigma
    if abs(xi) < 1e-6:
        return -np.log(sigma) - z - np.exp(-z)
    t = 1 + xi * z
    if np.any(t <= 0):
        return -np.inf
    return -np.log(sigma) - (1/xi + 1)*np.log(t) - t**(-1/xi)


def fit_stationary(y):
    n = len(y)
    mu0, s0 = np.median(y), np.std(y)
    def negll(p):
        mu, ls, xi = p
        return -sum([gev_logpdf(y[i], mu, np.exp(ls), xi) for i in range(n)])
    best, bll = None, -np.inf
    for mi in [mu0, mu0*0.8, mu0*1.2]:
        r = minimize(negll, [mi, np.log(s0), 0.0], method='Nelder-Mead',
                    options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
        if -r.fun > bll:
            bll = -r.fun; best = r
    mu, ls, xi = best.x
    return mu, np.exp(ls), xi, bll


def fit_nonstationary(y, cov):
    n = len(y)
    mu0, s0 = np.median(y), np.std(y)
    def negll(p):
        b0, b1, ls, xi = p
        return -sum([gev_logpdf(y[i], b0+b1*cov[i], np.exp(ls), xi)
                    for i in range(n)])
    best, bll = None, -np.inf
    for b1i in [-60, -30, -10, 0, 10, 30, 60]:
        r = minimize(negll, [mu0, b1i, np.log(s0), 0.0], method='Nelder-Mead',
                    options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
        if -r.fun > bll:
            bll = -r.fun; best = r
    b0, b1, ls, xi = best.x
    return b0, b1, np.exp(ls), xi, bll


def fit_dual(y, cov1, cov2):
    n = len(y)
    mu0, s0 = np.median(y), np.std(y)
    def negll(p):
        b0, b1, b2, ls, xi = p
        return -sum([gev_logpdf(y[i], b0+b1*cov1[i]+b2*cov2[i],
                    np.exp(ls), xi) for i in range(n)])
    best, bll = None, -np.inf
    for b1i in [-30, 0, 30]:
        r = minimize(negll, [mu0, b1i, 0.0, np.log(s0), 0.0],
                    method='Nelder-Mead',
                    options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
        if -r.fun > bll:
            bll = -r.fun; best = r
    b0, b1, b2, ls, xi = best.x
    return b0, b1, b2, np.exp(ls), xi, bll


def return_level(T, mu, sigma, xi):
    if abs(xi) < 1e-6:
        return mu - sigma*np.log(-np.log(1-1/T))
    return mu + (sigma/xi)*((-np.log(1-1/T))**(-xi)-1)


def gev_cdf(x, mu, sigma, xi):
    z = (x - mu) / sigma
    if abs(xi) < 1e-6:
        return np.exp(-np.exp(-z))
    t = 1 + xi*z
    if t <= 0:
        return 0.0
    return np.exp(-t**(-1/xi))


def wald_ci(y, dmi_n, b0, b1, sig, xi, eps=1e-4):
    n = len(y)
    def negll(p):
        b0_, b1_, ls_, xi_ = p
        return -sum([gev_logpdf(y[i], b0_+b1_*dmi_n[i],
                     np.exp(ls_), xi_) for i in range(n)])
    params = np.array([b0, b1, np.log(sig), xi])
    f0   = negll(params)
    p_up = params.copy(); p_up[1] += eps
    p_dn = params.copy(); p_dn[1] -= eps
    se   = np.sqrt(abs(eps**2 / (negll(p_up) - 2*f0 + negll(p_dn))))
    return b1 - 1.96*se, b1 + 1.96*se

# ============================================================
# DATA LOADING
# ============================================================

def load_dmi():
    rows = []
    with open(DMI_FILE,'r') as f:
        for line in f.readlines()[1:]:
            p = line.strip().split()
            if len(p)==13:
                try:
                    rows.append([int(p[0])] + [float(x) for x in p[1:]])
                except: continue
    cols = ['Year','Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec']
    df = pd.DataFrame(rows, columns=cols)
    df['DMI_OND'] = df[['Oct','Nov','Dec']].mean(axis=1)
    return df[(df['Year']>=1948)&(df['Year']<=2013)][['Year','DMI_OND']]


def extract_ond_max(df, stations=None):
    """
    Extract OND Annual Maximum Monthly Rainfall.
    
    For each year: maximum of October, November, December monthly totals.
    This is:
    - Physically motivated: IOD affects OND season
    - Statistically valid: block maximum for GEV
    - Better than seasonal total (avoids summing non-extreme months)
    """
    if stations:
        df = df[df['Station'].isin(stations)]

    # Filter OND months only
    df_ond = df[df['Month'].isin([10, 11, 12])].copy()

    # Per station: max OND monthly value each year
    station_ond_max = df_ond.groupby(['Station', 'Year'])['Monthly_Total'].max().reset_index()

    # Regional average of OND maxima
    regional_avg = station_ond_max.groupby('Year')['Monthly_Total'].mean()

    return regional_avg

# ============================================================
# ENGINEERING TABLES
# ============================================================

def print_engineering_tables(mu_s, sig_s, xi_s,
                              mu_neg, mu_pos, sig_n, xi_n,
                              region_name):
    print(f"\n  --- Recurrence Compression ({region_name}) ---")
    print(f"\n  {'Design T':<12}{'Design RL':<14}{'Neg IOD T':<14}{'Pos IOD T':<14}{'Compression'}")
    print(f"  {'-'*62}")
    for T in [50, 100, 200]:
        rl    = return_level(T, mu_s, sig_s, xi_s)
        p_neg = gev_cdf(rl, mu_neg, sig_n, xi_n)
        p_pos = gev_cdf(rl, mu_pos, sig_n, xi_n)
        T_neg = 1/(1-p_neg) if p_neg < 1 else float('inf')
        T_pos = 1/(1-p_pos) if p_pos < 1 else float('inf')
        comp  = T/T_neg
        print(f"  {str(T)+'-yr':<12}{rl:<14.1f}{T_neg:<14.0f}{T_pos:<14.0f}{comp:.2f}x")

    print(f"\n  --- Lifetime Failure Probability ({region_name}) ---")
    print(f"\n  {'Asset':<28}{'Life':<8}{'Stat%':<12}{'Neg IOD%':<14}{'Increase'}")
    print(f"  {'-'*68}")
    assets = [
        (25,  15,  "Urban Drainage"),
        (50,  30,  "Bridges/Levees"),
        (100, 50,  "Major River Works"),
        (100, 100, "Major Dams")
    ]
    for T_d, n_l, name in assets:
        rl    = return_level(T_d, mu_s, sig_s, xi_s)
        p_neg = gev_cdf(rl, mu_neg, sig_n, xi_n)
        r_s   = (1-(1-1/T_d)**n_l)*100
        r_n   = (1-(p_neg)**n_l)*100
        inc   = r_n - r_s
        sign  = "+" if inc >= 0 else ""
        print(f"  {name:<28}{n_l:<8}{r_s:<12.1f}%{r_n:<14.1f}% {sign}{inc:.1f}%")

# ============================================================
# ANALYZE ONE REGION
# ============================================================

def analyze_region(region_name, df_rain, dmi_df, z500_df):
    print("\n" + "╔" + "═"*76 + "╗")
    print("║" + f" REGION: {region_name.upper()} ".center(76) + "║")
    print("╚" + "═"*76 + "╝")

    stations = REGIONS[region_name]

    # Extract OND annual maximum monthly rainfall
    y_series = extract_ond_max(df_rain, stations)

    # Align
    overlap = np.intersect1d(np.intersect1d(
        y_series.index.values,
        dmi_df['Year'].values),
        z500_df['Year'].values)

    y     = y_series[overlap].values
    dmi   = dmi_df[dmi_df['Year'].isin(overlap)]['DMI_OND'].values
    z500  = z500_df[z500_df['Year'].isin(overlap)]['Z500_anomaly'].values
    dmi_n  = (dmi  - dmi.mean())  / dmi.std()
    z500_n = (z500 - z500.mean()) / z500.std()

    print(f"\n  n = {len(y)} years ({overlap.min()}–{overlap.max()})")
    print(f"  OND Max Monthly mean: {np.mean(y):.1f} mm")
    print(f"  OND Max Monthly std:  {np.std(y):.1f} mm")
    print(f"  Corr(DMI,  OND max) = {np.corrcoef(dmi,  y)[0,1]:+.4f}")
    print(f"  Corr(Z500, OND max) = {np.corrcoef(z500, y)[0,1]:+.4f}")

    # FIT ALL 4 MODELS
    print(f"\n  {'Model':<20}{'k':<5}{'LL':<12}{'AIC':<12}{'ΔAIC':<12}{'p-val'}")
    print(f"  {'-'*70}")

    mu_s, sig_s, xi_s, ll0 = fit_stationary(y)
    aic0 = 2*3 - 2*ll0
    print(f"  {'M0: Stationary':<20}{'3':<5}{ll0:<12.2f}{aic0:<12.2f}{'0.00':<12}{'--'}")

    b1_0,b1_1,b1_s,b1_x,ll1 = fit_nonstationary(y, dmi_n)
    aic1 = 2*4-2*ll1
    lrt1 = 2*(ll1-ll0); p1 = 1-stats.chi2.cdf(lrt1, df=1)
    print(f"  {'M1: DMI only':<20}{'4':<5}{ll1:<12.2f}{aic1:<12.2f}{aic1-aic0:<12.2f}{p1:.4f} {'✓' if p1<0.05 else '⚠' if p1<0.10 else '✗'}")

    b2_0,b2_2,b2_s,b2_x,ll2 = fit_nonstationary(y, z500_n)
    aic2 = 2*4-2*ll2
    lrt2 = 2*(ll2-ll0); p2 = 1-stats.chi2.cdf(lrt2, df=1)
    print(f"  {'M2: Z500 only':<20}{'4':<5}{ll2:<12.2f}{aic2:<12.2f}{aic2-aic0:<12.2f}{p2:.4f} {'✓' if p2<0.05 else '⚠' if p2<0.10 else '✗'}")

    b3_0,b3_1,b3_2,b3_s,b3_x,ll3 = fit_dual(y, dmi_n, z500_n)
    aic3 = 2*5-2*ll3
    lrt3 = 2*(ll3-ll0); p3 = 1-stats.chi2.cdf(lrt3, df=2)
    print(f"  {'M3: DMI+Z500':<20}{'5':<5}{ll3:<12.2f}{aic3:<12.2f}{aic3-aic0:<12.2f}{p3:.4f} {'✓' if p3<0.05 else '⚠' if p3<0.10 else '✗'}")

    # M1 DETAILS
    print(f"\n  M1: β0={b1_0:.2f}, β1={b1_1:.4f}, σ={b1_s:.2f}, ξ={b1_x:.4f}")
    ci_lo, ci_hi = wald_ci(y, dmi_n, b1_0, b1_1, b1_s, b1_x)
    exc = "✓ excludes 0" if ci_lo*ci_hi > 0 else "⚠ includes 0"
    print(f"  β1 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]  {exc}")

    # RETURN LEVELS
    mu_neg = b1_0 + b1_1*(-1.5)
    mu_pos = b1_0 + b1_1*(+1.5)

    print(f"\n  {'T':<8}{'Stat RL':<14}{'Neg IOD':<14}{'Pos IOD':<14}{'% Change'}")
    print(f"  {'-'*56}")
    for T in [10, 25, 50, 100]:
        rl_s   = return_level(T, mu_s,   sig_s,  xi_s)
        rl_neg = return_level(T, mu_neg, b1_s,   b1_x)
        rl_pos = return_level(T, mu_pos, b1_s,   b1_x)
        pct    = (rl_neg - rl_s)/rl_s*100
        print(f"  {T:<8}{rl_s:<14.1f}{rl_neg:<14.1f}{rl_pos:<14.1f}{pct:+.1f}%")

    # ENGINEERING TABLES
    print_engineering_tables(mu_s, sig_s, xi_s,
                             mu_neg, mu_pos, b1_s, b1_x,
                             region_name)

    # SPLIT SAMPLE VALIDATION
    mid = len(y)//2
    dual_train = np.column_stack([dmi_n[:mid], z500_n[:mid]])
    dual_val   = np.column_stack([dmi_n[mid:], z500_n[mid:]])
    rt = fit_dual(y[:mid], dmi_n[:mid], z500_n[:mid])
    tb0, tb1, tb2 = rt[0], rt[1], rt[2]
    pred = np.array([tb0+tb1*dmi_n[mid+i]+tb2*z500_n[mid+i]
                    for i in range(len(y)-mid)])
    rmse = np.sqrt(np.mean((y[mid:]-pred)**2))
    corr = np.corrcoef(y[mid:], pred)[0,1]
    print(f"\n  Split-sample: RMSE={rmse:.1f} mm, r={corr:.4f}")

    return {
        'region': region_name,
        'n': len(y),
        'mean': np.mean(y),
        'beta1': b1_1,
        'corr_dmi': np.corrcoef(dmi, y)[0,1],
        'p_m1': p1,
        'delta_aic': aic1-aic0,
        'ci': (ci_lo, ci_hi),
        'p_m3': p3,
        'delta_aic_m3': aic3-aic0,
        'rmse': rmse,
        'val_r': corr
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n╔" + "═"*78 + "╗")
    print("║" + " PAPER 2: FINAL CORRECT VERSION ".center(78) + "║")
    print("║" + " Variable: OND Annual Maximum Monthly Rainfall ".center(78) + "║")
    print("║" + " (Block Maxima — Physically + Statistically Valid) ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # LOAD
    print("\nLoading data...")
    df_rain = pd.read_csv(RAINFALL_FILE)
    dmi_df  = load_dmi()
    z500_df = pd.read_csv(Z500_FILE)
    z500_df = z500_df[(z500_df['Year']>=1948)&(z500_df['Year']<=2013)]

    print(f"✓ Rainfall: {df_rain.shape}")
    print(f"✓ DMI: {len(dmi_df)} years")
    print(f"✓ Z500: {len(z500_df)} years")

    # RUN ALL REGIONS
    summary = []
    for region in ['National','Coastal','Northwestern','Northeastern','Central']:
        result = analyze_region(region, df_rain, dmi_df, z500_df)
        summary.append(result)

    # FINAL SUMMARY
    print("\n\n" + "="*90)
    print("FINAL REGIONAL SUMMARY")
    print("Variable: OND Annual Maximum Monthly Rainfall")
    print("="*90)
    print(f"\n{'Region':<15}{'Mean':<8}{'β1':<10}{'Corr':<8}{'p(M1)':<10}{'ΔAIC':<8}{'CI':<14}{'p(M3)':<10}{'Sig'}")
    print("-"*88)
    for r in summary:
        ci_exc = "excl 0 ✓" if r['ci'][0]*r['ci'][1]>0 else "incl 0 ⚠"
        sig    = "✓ STRONG" if r['p_m1']<0.05 else "⚠ MARGINAL" if r['p_m1']<0.10 else "✗ WEAK"
        print(f"{r['region']:<15}{r['mean']:<8.1f}{r['beta1']:<10.2f}"
              f"{r['corr_dmi']:<8.3f}{r['p_m1']:<10.4f}"
              f"{r['delta_aic']:<8.2f}{ci_exc:<14}{r['p_m3']:<10.4f}{sig}")

    print(f"""
{"="*90}
WHAT THIS VARIABLE MEANS
{"="*90}

OND Annual Maximum Monthly Rainfall = the single wettest month
in October, November, or December each year.

WHY THIS IS THE CORRECT VARIABLE:
1. GEV requires block maxima — this IS a proper block maximum
2. IOD affects OND season — this isolates the IOD-sensitive period
3. Maximum captures extremes — not diluted by summing all 3 months
4. Physically defensible — each year's worst OND month

This is the variable that:
- Drives flash flood events in post-monsoon Bangladesh
- Determines drainage design for OND season
- Is directly linked to IOD through Paper 1's findings
{"="*90}
""")

    print("✓ ANALYSIS COMPLETE — Results ready for manuscript")
