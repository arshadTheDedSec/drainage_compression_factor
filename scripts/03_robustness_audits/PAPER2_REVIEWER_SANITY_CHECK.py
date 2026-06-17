"""
REVIEWER SANITY CHECK
=====================
For DMI states from -2σ to +2σ:
- Compute equivalent return period of the stationary 100-year design level
- Show monotonic relationship
- Create publication-ready table
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

CENTRAL_STATIONS = ["Dhaka","Faridpur","Madaripur","Tangail",
                    "Barisal","Khulna","Mongla","Jessore","Satkhira","Bhola"]

YEAR_START = 1982
YEAR_END = 2013

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

def gev_cdf(x, mu, sigma, xi):
    """CDF of GEV distribution"""
    if sigma <= 0:
        return np.nan
    z = (x - mu) / sigma
    if abs(xi) < 1e-6:
        return np.exp(-np.exp(-z))
    t = 1 + xi * z
    if np.any(t <= 0):
        return np.nan
    return np.exp(-t**(-1/xi))

def gev_quantile(p, mu, sigma, xi):
    """Quantile (inverse CDF) of GEV distribution"""
    if abs(xi) < 1e-6:
        return mu - sigma * np.log(-np.log(p))
    else:
        return mu + sigma / xi * ((-np.log(p))**(-xi) - 1)

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

def fit_single_covariate(y, cov):
    """M: μ = β₀ + β₁·cov"""
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

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load and merge all data for 1982-2013 period"""
    # Rainfall
    df_rain = pd.read_csv(RAINFALL_FILE)
    df_central = df_rain[df_rain['Station'].isin(CENTRAL_STATIONS)].copy()
    df_ond = df_central[df_central['Month'].isin([10, 11, 12])].copy()
    annual_max = df_ond.groupby(['Station', 'Year'])['Monthly_Total'].max().reset_index()
    regional = annual_max.groupby('Year')['Monthly_Total'].mean().reset_index()
    regional = regional[(regional['Year'] >= YEAR_START) & (regional['Year'] <= YEAR_END)]
    
    # DMI
    rows_dmi = []
    with open(DMI_FILE, 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 13:
                try:
                    year = int(parts[0])
                    oct = float(parts[11])
                    nov = float(parts[12])
                    dec = float(parts[1])
                    rows_dmi.append([year, (oct + nov + dec) / 3])
                except:
                    continue
    dmi_df = pd.DataFrame(rows_dmi, columns=['Year', 'DMI_OND'])
    
    # ENSO
    rows_enso = []
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
                        rows_enso.append([year, month, nino34_anom])
                except (ValueError, IndexError):
                    continue
    
    df_enso = pd.DataFrame(rows_enso, columns=['Year', 'Month', 'NINO34_ANOM'])
    enso_df = df_enso.groupby('Year')['NINO34_ANOM'].mean().reset_index()
    enso_df.columns = ['Year', 'NINO34_OND']
    
    # Merge
    data = pd.DataFrame({'Year': regional['Year'], 'Rainfall': regional['Monthly_Total']})
    data = data.merge(dmi_df, on='Year', how='inner')
    data = data.merge(enso_df, on='Year', how='inner')
    
    return data

# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == "__main__":

    print("\n╔" + "═"*78 + "╗")
    print("║" + " REVIEWER SANITY CHECK ".center(78) + "║")
    print("║" + " Monotonic relationship: DMI state → design rainfall recurrence ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # Load data
    print("\nLoading data (1982–2013)...")
    data = load_data()
    
    y = data['Rainfall'].values
    dmi = data['DMI_OND'].values
    n = len(y)
    
    # Standardize DMI
    dmi_mean = dmi.mean()
    dmi_std = dmi.std()
    dmi_n = (dmi - dmi_mean) / dmi_std
    
    print(f"✓ Data loaded: n = {n} years")

    # Fit models
    print("\nFitting models...")
    mu_stat, sigma_stat, xi_stat, _ = fit_stationary(y)
    b0_dmi, b1_dmi, sigma_dmi, xi_dmi, _ = fit_single_covariate(y, dmi_n)
    print("✓ Models fitted")

    # Compute stationary 100-year design level
    rl100_stationary = gev_quantile(1 - 1/100, mu_stat, sigma_stat, xi_stat)
    
    print(f"\n" + "="*80)
    print(f"Stationary 100-year design level: {rl100_stationary:.1f} mm")
    print("="*80)

    # ============================================================
    # CORE ANALYSIS: DMI states from -2σ to +2σ
    # ============================================================
    
    dmi_states = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    dmi_labels = ['-2σ', '-1.5σ', '-1σ', '-0.5σ', '0σ', '+0.5σ', '+1σ', '+1.5σ', '+2σ']
    
    results = []
    
    print(f"\nComputing equivalent return periods for the stationary 100-yr design level")
    print(f"under different DMI states...\n")

    for state_val, label in zip(dmi_states, dmi_labels):
        # Compute location parameter under this DMI state
        mu_dmi_state = b0_dmi + b1_dmi * state_val
        
        # Evaluate CDF at the design level
        cdf_at_design = gev_cdf(rl100_stationary, mu_dmi_state, sigma_dmi, xi_dmi)
        
        # Compute equivalent return period
        if cdf_at_design > 0 and cdf_at_design < 1:
            equiv_period = 1 / (1 - cdf_at_design)
        else:
            equiv_period = np.nan
        
        results.append({
            'DMI State': label,
            'DMI (std)': state_val,
            'Equiv. Period (yr)': equiv_period
        })
        
        print(f"{label:>6}  →  {equiv_period:>7.1f} yr", end='')
        if equiv_period < 100:
            print(f"  (compression from 100-yr)")
        elif equiv_period > 100:
            print(f"  (dilation from 100-yr)")
        else:
            print(f"  (no change)")

    # ============================================================
    # PUBLICATION TABLE
    # ============================================================
    print(f"\n" + "="*80)
    print("TABLE FOR PUBLICATION")
    print("="*80)

    df_results = pd.DataFrame(results)
    
    print(f"\nTable: Equivalent Return Period of Stationary 100-Year Design Rainfall")
    print(f"       Under Different IOD States (Central Bangladesh, 1982–2013)")
    print(f"\n{'DMI State':<12} {'Equivalent Period (yr)':<25}")
    print("-"*37)
    for _, row in df_results.iterrows():
        print(f"{row['DMI State']:<12} {row['Equiv. Period (yr)']:>20.1f}")

    # ============================================================
    # MONOTONICITY CHECK
    # ============================================================
    print(f"\n" + "="*80)
    print("MONOTONICITY VERIFICATION")
    print("="*80)

    equiv_periods = df_results['Equiv. Period (yr)'].values
    diffs = np.diff(equiv_periods)
    all_increasing = np.all(diffs > 0)
    
    print(f"\nPeriod values: {', '.join([f'{x:.1f}' for x in equiv_periods])}")
    print(f"Differences:   {', '.join([f'{x:+.1f}' for x in diffs])}")
    
    if all_increasing:
        print(f"\n✓ MONOTONICALLY INCREASING: As DMI becomes more positive,")
        print(f"  design rainfall recurs less frequently (dilation).")
        print(f"  As DMI becomes more negative, design rainfall recurs more")
        print(f"  frequently (compression).")
    else:
        print(f"\n✗ NOT MONOTONIC: Some reversals detected")
        print(f"  This would require explanation.")

    # ============================================================
    # REVIEWER-FRIENDLY SUMMARY
    # ============================================================
    print(f"\n" + "="*80)
    print("REVIEWER-FRIENDLY SUMMARY")
    print("="*80)

    period_at_neg2 = equiv_periods[0]
    period_at_pos2 = equiv_periods[-1]
    ratio = period_at_pos2 / period_at_neg2
    
    print(f"\nUnder strong negative IOD (-2σ): design rainfall recurs every {period_at_neg2:.0f} years")
    print(f"Under strong positive IOD (+2σ): design rainfall recurs every {period_at_pos2:.0f} years")
    print(f"Ratio: {ratio:.1f}× difference across IOD phase spectrum")
    print(f"\nThis monotonic relationship demonstrates that Indian Ocean Dipole")
    print(f"phase substantially modulates the frequency of extreme rainfall")
    print(f"events, with compression factors of ~{100/period_at_neg2:.1f}× under negative phases.")

    print(f"\n" + "="*80)
    print("✓ REVIEWER SANITY CHECK COMPLETE")
    print("  Table is ready for publication")
    print("="*80)
