"""
PAPER 2: REGIONAL NON-STATIONARY GEV
Dual-Covariate Model: μ(t) = β0 + β1·DMI(t) + β2·Z500(t)
Regions: National, Coastal, Northwestern, Northeastern
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STATION REGIONAL MAPPING (Physically Motivated)
# ============================================================

REGIONS = {
    'Coastal': [
        'Cox\'s Bazar', 'Chittagong', 'Sandwip', 'Sitakunda',
        'Teknaf', 'Kutubdia', 'Hatiya', 'Khepupara', 'Patuakhali'
    ],
    'Northwestern': [
        'Rajshahi', 'Bogra', 'Rangpur', 'Dinajpur', 'Ishurdi',
        'Pabna', 'Sirajganj', 'Joypurhat', 'Chapai Nawabganj'
    ],
    'Northeastern': [
        'Sylhet', 'Sreemangal', 'Srimangal', 'Moulvibazar',
        'Comilla', 'Brahmanbaria', 'Mymensingh', 'Netrokona'
    ]
}

# ============================================================
# DATA LOADING
# ============================================================

def load_all_data(rainfall_file, dmi_file, z500_file):
    print("=" * 80)
    print("LOADING ALL DATA")
    print("=" * 80)

    # Rainfall
    df_rain = pd.read_csv(rainfall_file)
    print(f"✓ Rainfall: {df_rain.shape}, Stations: {df_rain['Station'].nunique()}")
    print(f"  Available stations: {sorted(df_rain['Station'].unique())}")

    # DMI
    dmi_rows = []
    with open(dmi_file, 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) == 13:
                try:
                    year = int(parts[0])
                    months = [float(x) for x in parts[1:]]
                    dmi_rows.append([year] + months)
                except:
                    continue
    cols = ['Year','Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec']
    df_dmi = pd.DataFrame(dmi_rows, columns=cols)
    df_dmi['DMI_OND'] = df_dmi[['Oct','Nov','Dec']].mean(axis=1)
    df_dmi = df_dmi[(df_dmi['Year'] >= 1948) & (df_dmi['Year'] <= 2013)]
    print(f"✓ DMI: {len(df_dmi)} years, OND mean={df_dmi['DMI_OND'].mean():.4f}")

    # Z500
    df_z500 = pd.read_csv(z500_file)
    df_z500 = df_z500[(df_z500['Year'] >= 1948) & (df_z500['Year'] <= 2013)]
    print(f"✓ Z500: {len(df_z500)} years, anomaly range=[{df_z500['Z500_anomaly'].min():.1f}, {df_z500['Z500_anomaly'].max():.1f}]")

    return df_rain, df_dmi, df_z500


def extract_ond_regional(df_rain):
    """Extract OND seasonal totals by region."""
    print("\n" + "=" * 80)
    print("EXTRACTING OND SEASONAL TOTALS BY REGION")
    print("=" * 80)

    df_ond = df_rain[df_rain['Month'].isin([10, 11, 12])].copy()
    ond_sum = df_ond.groupby(['Station', 'Year'])['Monthly_Total'].sum().reset_index()
    ond_pivot = ond_sum.pivot(index='Year', columns='Station', values='Monthly_Total')

    available = list(ond_pivot.columns)
    print(f"  Available stations: {available}")

    regional_series = {}

    # National average
    regional_series['National'] = ond_pivot.mean(axis=1)
    print(f"\n✓ National: {len(regional_series['National'])} years, mean={regional_series['National'].mean():.1f} mm")

    # Regional averages
    for region, stations in REGIONS.items():
        matched = [s for s in stations if s in available]
        if len(matched) >= 2:
            regional_series[region] = ond_pivot[matched].mean(axis=1)
            print(f"✓ {region}: {len(matched)} stations matched {matched}")
            print(f"   Mean OND = {regional_series[region].mean():.1f} mm")
        else:
            print(f"⚠ {region}: Only {len(matched)} stations matched — skipping")
            print(f"   Looking for: {stations}")
            print(f"   Found: {matched}")

    return regional_series


# ============================================================
# GEV CORE FUNCTIONS
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
    return -np.log(sigma) - (1/xi + 1) * np.log(t) - t**(-1/xi)


def fit_stationary(y):
    """Fit stationary GEV."""
    mu0, sigma0 = np.median(y), np.std(y)
    n = len(y)

    def negll(p):
        mu, ls, xi = p
        sig = np.exp(ls)
        return -sum([gev_logpdf(y[i], mu, sig, xi) for i in range(n)])

    res = minimize(negll, [mu0, np.log(sigma0), 0.0],
                   method='Nelder-Mead',
                   options={'xatol':1e-8,'fatol':1e-8,'maxiter':10000})
    mu, ls, xi = res.x
    return mu, np.exp(ls), xi, -res.fun


def fit_dual_covariate(y, dmi_norm, z500_norm):
    """Fit dual-covariate GEV: μ(t) = β0 + β1·DMI + β2·Z500."""
    mu0, sigma0 = np.median(y), np.std(y)
    n = len(y)

    def negll(p):
        b0, b1, b2, ls, xi = p
        sig = np.exp(ls)
        ll = 0
        for i in range(n):
            mu_t = b0 + b1*dmi_norm[i] + b2*z500_norm[i]
            ll += gev_logpdf(y[i], mu_t, sig, xi)
        return -ll

    best_res, best_ll = None, -np.inf
    for b1 in [-60, -30, -10, 0, 10, 30, 60]:
        for b2 in [-30, 0, 30]:
            try:
                r = minimize(negll, [mu0, b1, b2, np.log(sigma0), 0.0],
                             method='Nelder-Mead',
                             options={'xatol':1e-8,'fatol':1e-8,'maxiter':10000})
                if -r.fun > best_ll:
                    best_ll = -r.fun
                    best_res = r
            except:
                continue

    b0, b1, b2, ls, xi = best_res.x
    return b0, b1, b2, np.exp(ls), xi, best_ll


def bootstrap_ci(y, dmi, z500, n_boot=1000):
    """Bootstrap 95% CI for β1, β2, and 100-yr return level."""
    np.random.seed(42)
    n = len(y)
    dmi_norm = (dmi - np.mean(dmi)) / np.std(dmi)
    z500_norm = (z500 - np.mean(z500)) / np.std(z500)

    b1_boot, b2_boot, rl100_boot = [], [], []

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yb = y[idx]
        db = dmi_norm[idx]
        zb = z500_norm[idx]
        try:
            b0, b1, b2, sig, xi, _ = fit_dual_covariate(yb, db, zb)
            b1_boot.append(b1)
            b2_boot.append(b2)
            if abs(xi) < 1e-6:
                rl = b0 - sig * np.log(-np.log(0.99))
            else:
                rl = b0 + (sig/xi) * ((-np.log(0.99))**(-xi) - 1)
            rl100_boot.append(rl)
        except:
            continue

    ci = lambda arr, a=5: (np.percentile(arr, a), np.percentile(arr, 100-a))
    return ci(b1_boot), ci(b2_boot), ci(rl100_boot)


def split_sample(y, dmi, z500, years):
    """Split-sample validation."""
    mid = len(y) // 2
    y1, d1, z1 = y[:mid], dmi[:mid], z500[:mid]
    y2, d2, z2 = y[mid:], dmi[mid:], z500[mid:]

    d1n = (d1 - np.mean(d1)) / np.std(d1)
    z1n = (z1 - np.mean(z1)) / np.std(z1)
    d2n = (d2 - np.mean(d1)) / np.std(d1)  # Use train mean/std
    z2n = (z2 - np.mean(z1)) / np.std(z1)

    b0, b1, b2, sig, xi, _ = fit_dual_covariate(y1, d1n, z1n)
    pred_mu = np.array([b0 + b1*d2n[i] + b2*z2n[i] for i in range(len(y2))])

    rmse = np.sqrt(np.mean((y2 - pred_mu)**2))
    mae = np.mean(np.abs(y2 - pred_mu))
    corr = np.corrcoef(y2, pred_mu)[0, 1]

    return rmse, mae, corr, years[mid], years[-1]


def return_level(T, mu, sigma, xi):
    if abs(xi) < 1e-6:
        return mu - sigma * np.log(-np.log(1 - 1/T))
    return mu + (sigma/xi) * ((-np.log(1 - 1/T))**(-xi) - 1)


# ============================================================
# RUN ANALYSIS FOR ONE REGION
# ============================================================

def analyze_region(name, rainfall_series, dmi_df, z500_df):
    print("\n" + "=" * 80)
    print(f"ANALYZING REGION: {name.upper()}")
    print("=" * 80)

    # Align years
    years_r = rainfall_series.index.values
    years_d = dmi_df['Year'].values
    years_z = z500_df['Year'].values
    overlap = np.intersect1d(np.intersect1d(years_r, years_d), years_z)

    y = rainfall_series[overlap].values
    dmi = dmi_df[dmi_df['Year'].isin(overlap)]['DMI_OND'].values
    z500 = z500_df[z500_df['Year'].isin(overlap)]['Z500_anomaly'].values
    years = overlap

    print(f"  n = {len(y)} years ({years.min()}–{years.max()})")
    print(f"  Rainfall mean = {np.mean(y):.1f} mm, std = {np.std(y):.1f} mm")
    print(f"  Corr(DMI, Rain) = {np.corrcoef(dmi, y)[0,1]:.4f}")
    print(f"  Corr(Z500, Rain) = {np.corrcoef(z500, y)[0,1]:.4f}")

    # Normalize
    dmi_norm = (dmi - np.mean(dmi)) / np.std(dmi)
    z500_norm = (z500 - np.mean(z500)) / np.std(z500)

    # Stationary
    mu_s, sigma_s, xi_s, ll_s = fit_stationary(y)
    aic_s = 2*3 - 2*ll_s
    print(f"\n  Stationary: μ={mu_s:.2f}, σ={sigma_s:.2f}, ξ={xi_s:.4f}")
    print(f"  LL={ll_s:.2f}, AIC={aic_s:.2f}")

    # Dual-covariate
    b0, b1, b2, sigma_n, xi_n, ll_n = fit_dual_covariate(y, dmi_norm, z500_norm)
    aic_n = 2*5 - 2*ll_n
    print(f"\n  Dual-Covariate: β0={b0:.2f}, β1={b1:.4f}, β2={b2:.4f}")
    print(f"  σ={sigma_n:.2f}, ξ={xi_n:.4f}")
    print(f"  LL={ll_n:.2f}, AIC={aic_n:.2f}, ΔAIC={aic_n-aic_s:.2f}")

    # LRT
    Lambda = 2 * (ll_n - ll_s)
    pval = 1 - stats.chi2.cdf(Lambda, df=2)
    print(f"\n  LRT: Λ={Lambda:.4f}, p={pval:.4f} {'✓ SIGNIFICANT' if pval < 0.05 else '✗ not significant'}")

    # Bootstrap CI
    print(f"\n  Bootstrap CI (n=500)...")
    ci_b1, ci_b2, ci_rl100 = bootstrap_ci(y, dmi, z500, n_boot=500)
    print(f"  β1 95% CI: [{ci_b1[0]:.2f}, {ci_b1[1]:.2f}]  {'✓ excludes 0' if ci_b1[0]*ci_b1[1] > 0 else '⚠ includes 0'}")
    print(f"  β2 95% CI: [{ci_b2[0]:.2f}, {ci_b2[1]:.2f}]  {'✓ excludes 0' if ci_b2[0]*ci_b2[1] > 0 else '⚠ includes 0'}")
    print(f"  100-yr RL 95% CI: [{ci_rl100[0]:.1f}, {ci_rl100[1]:.1f}] mm")

    # Split-sample
    rmse, mae, corr, y_start, y_end = split_sample(y, dmi, z500, years)
    print(f"\n  Split-sample validation ({years[0]}–{years[len(y)//2-1]} → {y_start}–{y_end}):")
    print(f"  RMSE={rmse:.1f} mm, MAE={mae:.1f} mm, r={corr:.4f}")

    # Return levels
    T_vals = [10, 25, 50, 100]
    mu_neg = b0 + b1*(-1.5) + b2*(-1.5)  # Strong negative IOD + low Z500
    mu_pos = b0 + b1*(+1.5) + b2*(+1.5)  # Strong positive IOD + high Z500

    print(f"\n  Return Level Comparison (mm):")
    print(f"  {'T':<6} {'Stationary':<14} {'Neg IOD+Z500':<16} {'Pos IOD+Z500':<16} {'% Change'}")
    print(f"  {'-'*60}")
    for T in T_vals:
        rl_s = return_level(T, mu_s, sigma_s, xi_s)
        rl_neg = return_level(T, mu_neg, sigma_n, xi_n)
        rl_pos = return_level(T, mu_pos, sigma_n, xi_n)
        pct = (rl_neg - rl_s) / rl_s * 100
        print(f"  {T:<6} {rl_s:<14.1f} {rl_neg:<16.1f} {rl_pos:<16.1f} {pct:+.1f}%")

    return {
        'region': name,
        'n': len(y),
        'beta0': b0, 'beta1': b1, 'beta2': b2,
        'sigma': sigma_n, 'xi': xi_n,
        'll_stat': ll_s, 'll_nonstat': ll_n,
        'aic_stat': aic_s, 'aic_nonstat': aic_n,
        'delta_aic': aic_n - aic_s,
        'lrt_lambda': Lambda, 'lrt_pval': pval,
        'ci_beta1': ci_b1, 'ci_beta2': ci_b2,
        'ci_rl100': ci_rl100,
        'val_rmse': rmse, 'val_mae': mae, 'val_corr': corr
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n╔" + "═"*78 + "╗")
    print("║" + " PAPER 2: REGIONAL DUAL-COVARIATE NON-STATIONARY GEV ".center(78) + "║")
    print("║" + " μ(t) = β0 + β1·DMI(t) + β2·Z500(t) ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # FILE PATHS
    rainfall_file = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\rainfall_data_bangladesh_1948_to_2014.csv"
    dmi_file      = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\DMI_data.txt"
    z500_file     = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\Z500_Tibetan_OND.csv"

    # LOAD
    df_rain, df_dmi, df_z500 = load_all_data(rainfall_file, dmi_file, z500_file)

    # EXTRACT REGIONAL OND
    regional_series = extract_ond_regional(df_rain)

    # RUN FOR EACH REGION
    all_results = []
    for region_name, series in regional_series.items():
        try:
            result = analyze_region(region_name, series, df_dmi, df_z500)
            all_results.append(result)
        except Exception as e:
            print(f"\n⚠ {region_name} failed: {e}")
            continue

    # SUMMARY TABLE
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: ALL REGIONS")
    print("=" * 80)
    print(f"\n{'Region':<15} {'β1':<10} {'β2':<10} {'p-val':<10} {'ΔAIC':<10} {'RMSE':<10} {'r_val'}")
    print("-" * 75)
    for r in all_results:
        sig = "✓" if r['lrt_pval'] < 0.05 else "✗"
        print(f"{r['region']:<15} {r['beta1']:<10.2f} {r['beta2']:<10.2f} "
              f"{r['lrt_pval']:<10.4f} {r['delta_aic']:<10.2f} "
              f"{r['val_rmse']:<10.1f} {r['val_corr']:.4f} {sig}")

    print("\n✓ ANALYSIS COMPLETE")
