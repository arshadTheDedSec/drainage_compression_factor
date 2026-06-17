"""
STEP 1: LEAVE-ONE-OUT SENSITIVITY ANALYSIS
===========================================
Remove extreme years one at a time (1988, 1998, 2004, 2007).
Test robustness of DMI and ENSO coefficients.

If coefficients are stable → "Results are not driven by extreme years"
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

YEARS_TO_TEST = [1988, 1998, 2004, 2007]

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
    data = data[(data['Year'] >= 1982) & (data['Year'] <= 2013)]
    
    return data

# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == "__main__":

    print("\n╔" + "═"*78 + "╗")
    print("║" + " LEAVE-ONE-OUT SENSITIVITY ANALYSIS ".center(78) + "║")
    print("║" + " Robustness: Are results driven by extreme years? ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # Load full data (1982-2013)
    print("\nLoading data (1982–2013)...")
    data_full = load_data()
    
    y_full = data_full['Rainfall'].values
    dmi_full = data_full['DMI_OND'].values
    enso_full = data_full['NINO34_OND'].values
    years_full = data_full['Year'].values
    
    print(f"✓ Full dataset: n = {len(y_full)} years")

    # Standardize full data
    dmi_mean_full = dmi_full.mean()
    dmi_std_full = dmi_full.std()
    enso_mean_full = enso_full.mean()
    enso_std_full = enso_full.std()
    
    dmi_n_full = (dmi_full - dmi_mean_full) / dmi_std_full
    enso_n_full = (enso_full - enso_mean_full) / enso_std_full
    
    # Fit full data
    print("\nFitting full dataset (1982–2013)...")
    b0_dmi_full, b1_dmi_full, _, _, _ = fit_single_covariate(y_full, dmi_n_full)
    b0_enso_full, b1_enso_full, _, _, _ = fit_single_covariate(y_full, enso_n_full)
    
    print(f"  DMI effect: β₁ = {b1_dmi_full:.4f} mm/std")
    print(f"  ENSO effect: β₁ = {b1_enso_full:.4f} mm/std")

    # ============================================================
    # LEAVE-ONE-OUT TEST
    # ============================================================
    print("\n" + "="*80)
    print("LEAVE-ONE-OUT SENSITIVITY TEST")
    print("="*80)

    results = {
        'Full (1982–2013)': {
            'n': len(y_full),
            'exclude_year': None,
            'b1_dmi': b1_dmi_full,
            'b1_enso': b1_enso_full,
            'rainfall_mean': y_full.mean(),
            'rainfall_sd': y_full.std()
        }
    }

    for exclude_year in YEARS_TO_TEST:
        print(f"\nRemoving {exclude_year}...", end='', flush=True)
        
        # Find and exclude
        mask = years_full != exclude_year
        y = y_full[mask]
        dmi = dmi_full[mask]
        enso = enso_full[mask]
        
        # Re-standardize (important: use new mean/std)
        dmi_n = (dmi - dmi.mean()) / dmi.std()
        enso_n = (enso - enso.mean()) / enso.std()
        
        # Fit
        b0_dmi, b1_dmi, _, _, _ = fit_single_covariate(y, dmi_n)
        b0_enso, b1_enso, _, _, _ = fit_single_covariate(y, enso_n)
        
        results[f'Exclude {exclude_year}'] = {
            'n': len(y),
            'exclude_year': exclude_year,
            'b1_dmi': b1_dmi,
            'b1_enso': b1_enso,
            'rainfall_mean': y.mean(),
            'rainfall_sd': y.std()
        }
        
        print(f" Done")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Dataset':<25} {'n':<5} {'Rainfall':<15} {'DMI β₁':<15} {'ENSO β₁':<15}")
    print(f"{'':25} {'':5} {'Mean±SD':<15} {'':<15} {'':<15}")
    print("-"*75)

    for label, results_dict in results.items():
        n = results_dict['n']
        rain_mean = results_dict['rainfall_mean']
        rain_sd = results_dict['rainfall_sd']
        b1_dmi = results_dict['b1_dmi']
        b1_enso = results_dict['b1_enso']
        
        print(f"{label:<25} {n:<5} {rain_mean:>6.1f}±{rain_sd:<5.1f} {b1_dmi:>14.4f} {b1_enso:>14.4f}")

    # ============================================================
    # SENSITIVITY ANALYSIS
    # ============================================================
    print("\n" + "="*80)
    print("COEFFICIENT STABILITY")
    print("="*80)

    b1_dmi_full = results['Full (1982–2013)']['b1_dmi']
    b1_enso_full = results['Full (1982–2013)']['b1_enso']

    print(f"\nDMI effect:")
    print(f"  Full model: β₁ = {b1_dmi_full:.4f}")
    print(f"  Range when excluding extreme years:")
    
    b1_dmi_vals = [results[k]['b1_dmi'] for k in results if k != 'Full (1982–2013)']
    b1_dmi_min = min(b1_dmi_vals)
    b1_dmi_max = max(b1_dmi_vals)
    b1_dmi_mean_loo = np.mean(b1_dmi_vals)
    
    print(f"    Min: {b1_dmi_min:.4f}")
    print(f"    Max: {b1_dmi_max:.4f}")
    print(f"    Range: {b1_dmi_max - b1_dmi_min:.4f}")
    print(f"    Mean (LOO): {b1_dmi_mean_loo:.4f}")
    
    pct_change_dmi = (abs(b1_dmi_full - b1_dmi_mean_loo) / abs(b1_dmi_full)) * 100
    print(f"    % change from full: {pct_change_dmi:.1f}%")

    print(f"\nENSO effect:")
    print(f"  Full model: β₁ = {b1_enso_full:.4f}")
    print(f"  Range when excluding extreme years:")
    
    b1_enso_vals = [results[k]['b1_enso'] for k in results if k != 'Full (1982–2013)']
    b1_enso_min = min(b1_enso_vals)
    b1_enso_max = max(b1_enso_vals)
    b1_enso_mean_loo = np.mean(b1_enso_vals)
    
    print(f"    Min: {b1_enso_min:.4f}")
    print(f"    Max: {b1_enso_max:.4f}")
    print(f"    Range: {b1_enso_max - b1_enso_min:.4f}")
    print(f"    Mean (LOO): {b1_enso_mean_loo:.4f}")
    
    pct_change_enso = (abs(b1_enso_full - b1_enso_mean_loo) / abs(b1_enso_full)) * 100
    print(f"    % change from full: {pct_change_enso:.1f}%")

    # ============================================================
    # ROBUSTNESS VERDICT
    # ============================================================
    print("\n" + "="*80)
    print("ROBUSTNESS VERDICT")
    print("="*80)

    threshold = 10  # 10% change threshold

    if pct_change_dmi < threshold and pct_change_enso < threshold:
        print(f"\n✓ ROBUST: Coefficients stable across leave-one-out")
        print(f"  DMI: {pct_change_dmi:.1f}% change (< {threshold}%)")
        print(f"  ENSO: {pct_change_enso:.1f}% change (< {threshold}%)")
        print(f"\n  Reviewer defense:")
        print(f"  'Results are not driven by extreme years. Leave-one-out")
        print(f"   sensitivity analysis confirms coefficient stability'")
    else:
        if pct_change_dmi > threshold:
            print(f"\n⚠ DMI COEFFICIENT SENSITIVE: {pct_change_dmi:.1f}% change")
            print(f"   Investigate which year drives the sensitivity")
        if pct_change_enso > threshold:
            print(f"\n⚠ ENSO COEFFICIENT SENSITIVE: {pct_change_enso:.1f}% change")
            print(f"   Investigate which year drives the sensitivity")

    print("\n" + "="*80)
    print("✓ SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
