"""
REGIONAL GEV HETEROGENEITY ANALYSIS (CORRECTED)
================================================
Four correctly-defined regions, 1950–2013 (n=64).
For each region: M0 (stationary), M1 (DMI), M2 (ENSO), M3 (both).
One summary table: Region | β_DMI | p_DMI | β_ENSO | p_ENSO | Best Model
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURE PATHS
# ============================================================
RAINFALL_FILE = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\rainfall_data_bangladesh_1948_to_2014.csv"
DMI_FILE      = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\DMI_data.txt"
NINO_FILE     = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\nino34_monthly.txt"

YEAR_START = 1950
YEAR_END = 2013

# CORRECT regional definitions (5 regions)
REGIONS = {
    'Central': ["Dhaka", "Faridpur", "Madaripur", "Tangail", "Barisal", "Khulna", "Mongla", "Jessore", "Satkhira", "Bhola"],
    'Northeastern': ["Sylhet", "Srimangal", "Comilla", "Mymensingh", "Chandpur", "Feni", "Maijdee Court"],
    'Coastal': ["Chittagong", "Cox's Bazar", "Hatiya", "Sandwip", "Kutubdia", "Sitakunda", "Patenga", "Teknaf"],
    'Northwestern': ["Rajshahi", "Bogra", "Dinajpur", "Rangpur", "Thakurgaon"],
    'Southwestern': ["Patuakhali", "Khepupara"],
}

# ============================================================
# GEV FUNCTIONS
# ============================================================

def gev_logpdf(x, mu, sigma, xi):
    """Log PDF of GEV distribution"""
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
    """M0: Stationary GEV"""
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
            bll = -r.fun
            best = r
    
    mu, ls, xi = best.x
    return mu, np.exp(ls), xi, bll

def fit_covariate(y, cov):
    """M1/M2: μ = β₀ + β₁·cov"""
    n = len(y)
    mu0, s0 = np.median(y), np.std(y)
    
    def negll(p):
        b0, b1, ls, xi = p
        return -sum([gev_logpdf(y[i], b0+b1*cov[i], np.exp(ls), xi)
                    for i in range(n)])
    
    best, bll = None, -np.inf
    for b1i in [-50, -25, 0, 25, 50]:
        r = minimize(negll, [mu0, b1i, np.log(s0), 0.0], method='Nelder-Mead',
                    options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
        if -r.fun > bll:
            bll = -r.fun
            best = r
    
    b0, b1, ls, xi = best.x
    return b0, b1, np.exp(ls), xi, bll

def fit_dual_covariate(y, cov1, cov2):
    """M3: μ = β₀ + β₁·cov1 + β₂·cov2"""
    n = len(y)
    mu0, s0 = np.median(y), np.std(y)
    
    def negll(p):
        b0, b1, b2, ls, xi = p
        return -sum([gev_logpdf(y[i], b0+b1*cov1[i]+b2*cov2[i], np.exp(ls), xi)
                    for i in range(n)])
    
    best, bll = None, -np.inf
    for b1i in [-50, -25, 0, 25, 50]:
        for b2i in [-50, -25, 0, 25, 50]:
            r = minimize(negll, [mu0, b1i, b2i, np.log(s0), 0.0],
                        method='Nelder-Mead',
                        options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
            if -r.fun > bll:
                bll = -r.fun
                best = r
    
    b0, b1, b2, ls, xi = best.x
    return b0, b1, b2, np.exp(ls), xi, bll

# ============================================================
# DATA LOADING
# ============================================================

def load_rainfall_by_region(region_stations):
    """Load OND annual max rainfall for specific region"""
    df = pd.read_csv(RAINFALL_FILE)
    
    # Filter region stations
    df_region = df[df['Station'].isin(region_stations)].copy()
    
    if len(df_region) == 0:
        print(f"  ⚠ WARNING: No stations found for region. Stations requested: {region_stations}")
        return np.array([]), np.array([])
    
    # Extract OND
    df_ond = df_region[df_region['Month'].isin([10, 11, 12])].copy()
    
    # Annual max per station-year
    annual_max = df_ond.groupby(['Station', 'Year'])['Monthly_Total'].max().reset_index()
    
    # Regional average
    regional = annual_max.groupby('Year')['Monthly_Total'].mean().reset_index()
    regional = regional[(regional['Year'] >= YEAR_START) & (regional['Year'] <= YEAR_END)]
    
    return regional['Year'].values, regional['Monthly_Total'].values

def load_dmi():
    """Load DMI (OND average)"""
    rows = []
    with open(DMI_FILE, 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 13:
                try:
                    year = int(parts[0])
                    oct = float(parts[11])
                    nov = float(parts[12])
                    dec = float(parts[1])
                    rows.append([year, (oct + nov + dec) / 3])
                except:
                    continue
    
    df = pd.DataFrame(rows, columns=['Year', 'DMI_OND'])
    return df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]

def load_nino34():
    """Load Niño 3.4 (OND average) - ERSSTv5"""
    rows = []
    with open(NINO_FILE, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith('YR'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    nino34_anom = float(parts[4])
                    if month in [10, 11, 12]:
                        rows.append([year, month, nino34_anom])
                except (ValueError, IndexError):
                    continue
    
    df = pd.DataFrame(rows, columns=['Year', 'Month', 'NINO34_ANOM'])
    df_ond = df.groupby('Year')['NINO34_ANOM'].mean().reset_index()
    df_ond.columns = ['Year', 'NINO34_OND']
    return df_ond[(df_ond['Year'] >= YEAR_START) & (df_ond['Year'] <= YEAR_END)]

# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == "__main__":

    print("\n╔" + "═"*78 + "╗")
    print("║" + " REGIONAL SPATIAL HETEROGENEITY ANALYSIS ".center(78) + "║")
    print("║" + " GEV-DMI and GEV-ENSO across 5 regions (1950–2013) ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # Load climate data
    print("\nLoading climate indices...")
    dmi_df = load_dmi()
    nino_df = load_nino34()
    
    dmi = dmi_df['DMI_OND'].values
    nino = nino_df['NINO34_OND'].values
    
    # Merge to ensure same years
    data_clim = pd.DataFrame({'Year': dmi_df['Year'], 'DMI': dmi, 'NINO': nino})
    data_clim = data_clim.merge(nino_df, on='Year', how='inner')
    
    dmi = data_clim['DMI'].values
    nino = data_clim['NINO34_OND'].values
    
    # Standardize
    dmi_mean = dmi.mean()
    dmi_std = dmi.std()
    nino_mean = nino.mean()
    nino_std = nino.std()
    
    dmi_n = (dmi - dmi_mean) / dmi_std
    nino_n = (nino - nino_mean) / nino_std
    
    print(f"✓ Climate data loaded: n = {len(dmi)} years (1950–2013)")
    print(f"  DMI: mean={dmi_mean:.4f}, std={dmi_std:.4f}")
    print(f"  Niño3.4: mean={nino_mean:.4f}, std={nino_std:.4f}")

    # Analyze each region
    results = []

    for region_name, stations in REGIONS.items():
        print(f"\n" + "="*80)
        print(f"REGION: {region_name}")
        print("="*80)
        print(f"Stations ({len(stations)}): {', '.join(stations)}")
        
        # Load rainfall
        years, rainfall = load_rainfall_by_region(stations)
        
        if len(rainfall) == 0:
            print(f"  ✗ SKIPPED: No valid data")
            continue
        
        y = rainfall
        n = len(y)
        
        print(f"Data: n = {n} years, mean = {y.mean():.1f} mm, std = {y.std():.1f} mm")

        # ---- M0: Stationary ----
        print(f"\n  M0 (Stationary)...", end='', flush=True)
        mu_s, sig_s, xi_s, ll0 = fit_stationary(y)
        aic0 = 2*3 - 2*ll0
        print(f" Done")

        # ---- M1: DMI ----
        print(f"  M1 (DMI covariate)...", end='', flush=True)
        b0_dmi, b1_dmi, sig_dmi, xi_dmi, ll1_dmi = fit_covariate(y, dmi_n)
        aic1_dmi = 2*4 - 2*ll1_dmi
        lrt_dmi = 2 * (ll1_dmi - ll0)
        pval_dmi = 1 - stats.chi2.cdf(lrt_dmi, df=1)
        sig_dmi = "***" if pval_dmi < 0.001 else "**" if pval_dmi < 0.01 else "*" if pval_dmi < 0.05 else "NS"
        print(f" Done")
        print(f"    β₁ = {b1_dmi:.4f} mm/std, p = {pval_dmi:.6f} {sig_dmi}")

        # ---- M2: ENSO ----
        print(f"  M2 (ENSO covariate)...", end='', flush=True)
        b0_nino, b1_nino, sig_nino, xi_nino, ll1_nino = fit_covariate(y, nino_n)
        aic1_nino = 2*4 - 2*ll1_nino
        lrt_nino = 2 * (ll1_nino - ll0)
        pval_nino = 1 - stats.chi2.cdf(lrt_nino, df=1)
        sig_nino = "***" if pval_nino < 0.001 else "**" if pval_nino < 0.01 else "*" if pval_nino < 0.05 else "NS"
        print(f" Done")
        print(f"    β₁ = {b1_nino:.4f} mm/std, p = {pval_nino:.6f} {sig_nino}")

        # ---- M3: DMI + ENSO ----
        print(f"  M3 (DMI + ENSO)...", end='', flush=True)
        b0_dual, b1_dual_dmi, b1_dual_nino, sig_dual, xi_dual, _ = fit_dual_covariate(y, dmi_n, nino_n)
        print(f" Done")
        print(f"    β₁_DMI = {b1_dual_dmi:.4f}, β₁_ENSO = {b1_dual_nino:.4f}")

        # Determine best single model
        if aic1_dmi < aic1_nino:
            best_single = "M1 (DMI)"
            best_aic = aic1_dmi
        else:
            best_single = "M2 (ENSO)"
            best_aic = aic1_nino

        results.append({
            'Region': region_name,
            'n': n,
            'Mean_mm': y.mean(),
            'Std_mm': y.std(),
            'beta_dmi': b1_dmi,
            'pval_dmi': pval_dmi,
            'sig_dmi': sig_dmi,
            'beta_enso': b1_nino,
            'pval_enso': pval_nino,
            'sig_enso': sig_nino,
            'best_model': best_single,
            'aic_dmi': aic1_dmi,
            'aic_enso': aic1_nino,
        })

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "="*80)
    print("SUMMARY TABLE: Regional Climate Driver Comparison")
    print("="*80)

    df_results = pd.DataFrame(results)

    print(f"\n{'Region':<15} {'n':<5} {'β_DMI':<12} {'p_DMI':<10} {'β_ENSO':<12} {'p_ENSO':<10} {'Best Model':<12}")
    print("-"*80)

    for _, row in df_results.iterrows():
        print(f"{row['Region']:<15} {row['n']:<5} "
              f"{row['beta_dmi']:>10.4f}  {row['pval_dmi']:>9.6f}  "
              f"{row['beta_enso']:>10.4f}  {row['pval_enso']:>9.6f}  {row['best_model']:<12}")

    # ============================================================
    # HETEROGENEITY ASSESSMENT
    # ============================================================
    print("\n" + "="*80)
    print("REGIONAL HETEROGENEITY")
    print("="*80)

    print(f"\nDMI coefficient (β₁) across regions:")
    dmi_vals = df_results['beta_dmi'].values
    print(f"  Range: [{dmi_vals.min():.4f}, {dmi_vals.max():.4f}]")
    print(f"  Mean: {dmi_vals.mean():.4f}")
    print(f"  Std: {dmi_vals.std():.4f}")
    
    sig_dmi_regions = len(df_results[df_results['pval_dmi'] < 0.05])
    print(f"  Significant regions: {sig_dmi_regions}/{len(df_results)}")

    print(f"\nENSO coefficient (β₁) across regions:")
    enso_vals = df_results['beta_enso'].values
    print(f"  Range: [{enso_vals.min():.4f}, {enso_vals.max():.4f}]")
    print(f"  Mean: {enso_vals.mean():.4f}")
    print(f"  Std: {enso_vals.std():.4f}")
    
    sig_enso_regions = len(df_results[df_results['pval_enso'] < 0.05])
    print(f"  Significant regions: {sig_enso_regions}/{len(df_results)}")

    print(f"\nBest model by region:")
    for _, row in df_results.iterrows():
        print(f"  {row['Region']}: {row['best_model']}")

    print("\n" + "="*80)
    print("✓ REGIONAL ANALYSIS COMPLETE")
    print("="*80)
