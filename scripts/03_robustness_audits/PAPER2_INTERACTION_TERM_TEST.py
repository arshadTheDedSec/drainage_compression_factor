"""
INTERACTION TERM TEST
=====================
Pooled 1950–2013 analysis with regime interaction term.

Model: μ = β₀ + β₁·DMI + β₂·ENSO + β₃·Regime + β₄·(DMI×Regime)

H0: β₄ = 0 (DMI effect is constant across regimes)
H1: β₄ ≠ 0 (DMI effect changes after 1982)
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

def fit_model(y, dmi_n, enso_n, regime=None, dmi_x_regime=None):
    """
    Fit GEV with covariates.
    
    regime=None → no regime term (additive model)
    regime array → include regime + DMI×Regime interaction
    """
    n = len(y)
    mu0, s0 = np.median(y), np.std(y)
    
    if regime is None:
        # M4: μ = β₀ + β₁·DMI + β₂·ENSO
        def negll(p):
            b0, b1_dmi, b2_enso, ls, xi = p
            mu_array = b0 + b1_dmi*dmi_n + b2_enso*enso_n
            return -sum([gev_logpdf(y[i], mu_array[i], np.exp(ls), xi) 
                        for i in range(n)])
        
        best, bll = None, -np.inf
        for b1 in [-60, -30, -10, 0, 10, 30, 60]:
            for b2 in [-60, -30, -10, 0, 10, 30, 60]:
                r = minimize(negll, [mu0, b1, b2, np.log(s0), 0.0],
                            method='Nelder-Mead',
                            options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
                if -r.fun > bll:
                    bll = -r.fun
                    best = r
        
        b0, b1_dmi, b2_enso, ls, xi = best.x
        return b0, b1_dmi, b2_enso, None, None, np.exp(ls), xi, bll
    
    else:
        # M5: μ = β₀ + β₁·DMI + β₂·ENSO + β₃·Regime + β₄·(DMI×Regime)
        def negll(p):
            b0, b1_dmi, b2_enso, b3_regime, b4_interact, ls, xi = p
            mu_array = (b0 + b1_dmi*dmi_n + b2_enso*enso_n + 
                       b3_regime*regime + b4_interact*dmi_x_regime)
            return -sum([gev_logpdf(y[i], mu_array[i], np.exp(ls), xi) 
                        for i in range(n)])
        
        best, bll = None, -np.inf
        # Reduced grid: 5×5×3×3 = 225 calls instead of 1225
        for b1 in [-30, -10, 0, 10, 30]:
            for b2 in [-30, -10, 0, 10, 30]:
                for b3 in [-10, 0, 10]:
                    for b4 in [-10, 0, 10]:
                        r = minimize(negll, [mu0, b1, b2, b3, b4, np.log(s0), 0.0],
                                    method='Nelder-Mead',
                                    options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
                        if -r.fun > bll:
                            bll = -r.fun
                            best = r
        
        b0, b1_dmi, b2_enso, b3_regime, b4_interact, ls, xi = best.x
        return b0, b1_dmi, b2_enso, b3_regime, b4_interact, np.exp(ls), xi, bll

# ============================================================
# DATA LOADING
# ============================================================

def load_rainfall():
    """Load BMD rainfall data"""
    df = pd.read_csv(RAINFALL_FILE)
    df_central = df[df['Station'].isin(CENTRAL_STATIONS)].copy()
    df_ond = df_central[df_central['Month'].isin([10, 11, 12])].copy()
    annual_max = df_ond.groupby(['Station', 'Year'])['Monthly_Total'].max().reset_index()
    regional = annual_max.groupby('Year')['Monthly_Total'].mean().reset_index()
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
    return df

def load_nino34():
    """Load Niño 3.4 (OND average) - ERSSTv5 5-column format"""
    rows = []
    try:
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
    except FileNotFoundError:
        print(f"ERROR: Niño file not found")
        return pd.DataFrame(columns=['Year', 'NINO34_OND'])
    
    if not rows:
        return pd.DataFrame(columns=['Year', 'NINO34_OND'])
    
    df = pd.DataFrame(rows, columns=['Year', 'Month', 'NINO34_ANOM'])
    df_ond = df.groupby('Year')['NINO34_ANOM'].mean().reset_index()
    df_ond.columns = ['Year', 'NINO34_OND']
    
    return df_ond

# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == "__main__":

    print("\n╔" + "═"*78 + "╗")
    print("║" + " INTERACTION TERM TEST ".center(78) + "║")
    print("║" + " Does DMI effect change after 1982? ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # Load data
    print("\nLoading data...")
    years_rain, rainfall = load_rainfall()
    dmi_df = load_dmi()
    nino_df = load_nino34()
    print("✓ Data loaded")

    # Merge
    data = pd.DataFrame({'Year': years_rain, 'Rainfall': rainfall})
    data = data.merge(dmi_df, on='Year', how='inner')
    data = data.merge(nino_df, on='Year', how='inner')
    data = data[(data['Year'] >= 1950) & (data['Year'] <= 2013)]

    y = data['Rainfall'].values
    dmi = data['DMI_OND'].values
    enso = data['NINO34_OND'].values
    regime = (data['Year'].values >= 1982).astype(int)  # 0 before 1982, 1 after

    # Standardize
    dmi_n = (dmi - dmi.mean()) / dmi.std()
    enso_n = (enso - enso.mean()) / enso.std()
    
    # Interaction term
    dmi_x_regime = dmi_n * regime

    print(f"\nPooled dataset: 1950–2013, n = {len(y)} years")
    print(f"  Pre-1982: {sum(regime==0)} years")
    print(f"  Post-1982: {sum(regime==1)} years")

    # ============================================================
    # FIT MODELS
    # ============================================================
    print("\n" + "="*80)
    print("FITTING MODELS")
    print("="*80)

    # M4: No interaction (baseline additive)
    print("\nM4: μ = β₀ + β₁·DMI + β₂·ENSO (NO regime term)...")
    b0_m4, b1_dmi_m4, b2_enso_m4, _, _, sig_m4, xi_m4, ll_m4 = fit_model(y, dmi_n, enso_n)
    aic_m4 = 2*5 - 2*ll_m4
    
    print(f"  β₀={b0_m4:.2f}")
    print(f"  β₁_DMI={b1_dmi_m4:.4f}")
    print(f"  β₂_ENSO={b2_enso_m4:.4f}")
    print(f"  AIC = {aic_m4:.2f}")

    # M5: With interaction
    print("\nM5: μ = β₀ + β₁·DMI + β₂·ENSO + β₃·Regime + β₄·(DMI×Regime)...")
    b0_m5, b1_dmi_m5, b2_enso_m5, b3_regime, b4_interact, sig_m5, xi_m5, ll_m5 = fit_model(
        y, dmi_n, enso_n, regime=regime, dmi_x_regime=dmi_x_regime
    )
    aic_m5 = 2*6 - 2*ll_m5
    
    print(f"  β₀={b0_m5:.2f}")
    print(f"  β₁_DMI={b1_dmi_m5:.4f}")
    print(f"  β₂_ENSO={b2_enso_m5:.4f}")
    print(f"  β₃_Regime={b3_regime:.4f}")
    print(f"  β₄_DMI×Regime={b4_interact:.4f}  ← INTERACTION")
    print(f"  AIC = {aic_m5:.2f}")

    # ============================================================
    # HYPOTHESIS TEST
    # ============================================================
    print("\n" + "="*80)
    print("HYPOTHESIS TEST: β₄ = 0?")
    print("="*80)

    lrt_interact = 2 * (ll_m5 - ll_m4)
    pval_interact = 1 - stats.chi2.cdf(lrt_interact, df=1)
    aic_diff = aic_m4 - aic_m5

    print(f"\nNull: β₄ = 0 (DMI effect constant across regimes)")
    print(f"Alt:  β₄ ≠ 0 (DMI effect changes after 1982)")
    print(f"\nLRT test:")
    print(f"  LL(M4) = {ll_m4:.2f}")
    print(f"  LL(M5) = {ll_m5:.2f}")
    print(f"  LRT = 2×(LL_M5 - LL_M4) = {lrt_interact:.4f}")
    print(f"  χ²(df=1) critical = 3.841 (p=0.05)")
    print(f"  p-value = {pval_interact:.6f}")

    if pval_interact < 0.05:
        sig_flag = "***" if pval_interact < 0.001 else "**" if pval_interact < 0.01 else "*"
        print(f"\n✓ SIGNIFICANT {sig_flag}: β₄ ≠ 0")
        print(f"  The DMI effect DID change after 1982")
    else:
        print(f"\n✗ NOT SIGNIFICANT: β₄ = 0")
        print(f"  The DMI effect is constant across regimes")

    print(f"\nAIC comparison:")
    print(f"  AIC(M4 no interaction) = {aic_m4:.2f}")
    print(f"  AIC(M5 with interaction) = {aic_m5:.2f}")
    print(f"  ΔAIC = {aic_diff:.2f}")
    if aic_diff > 2:
        print(f"  → M5 is substantially better (supports regime change)")
    elif aic_diff > 0:
        print(f"  → M5 is slightly better (weak evidence)")
    else:
        print(f"  → M4 is better (no interaction needed)")

    # ============================================================
    # INTERPRET REGIME EFFECTS
    # ============================================================
    print("\n" + "="*80)
    print("REGIME-SPECIFIC DMI EFFECTS")
    print("="*80)

    print(f"\nM5 coefficients:")
    print(f"  β₁ (base DMI effect, pre-1982) = {b1_dmi_m5:.4f} mm/std")
    print(f"  β₄ (interaction/change term) = {b4_interact:.4f} mm/std")

    dmi_effect_pre = b1_dmi_m5
    dmi_effect_post = b1_dmi_m5 + b4_interact

    print(f"\nImplied DMI effects by regime:")
    print(f"  Pre-1982 (Regime=0): β1 = {dmi_effect_pre:.4f} mm/std")
    print(f"  Post-1982 (Regime=1): β1 = {dmi_effect_post:.4f} mm/std")
    print(f"  Change: {dmi_effect_post - dmi_effect_pre:.4f} mm/std")

    if abs(dmi_effect_pre) > 0.1 and abs(dmi_effect_post) > 0.1:
        if np.sign(dmi_effect_pre) == np.sign(dmi_effect_post):
            ratio = dmi_effect_post / dmi_effect_pre
            print(f"  Direction is consistent (both negative)")
            print(f"  Magnitude ratio (post/pre) = {ratio:.2f}×")
        else:
            print(f"  Direction FLIPPED between regimes")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("SUMMARY & CONCLUSION")
    print("="*80)

    if pval_interact < 0.05:
        print(f"\n✓ CONFIRMED: The DMI effect significantly changed after 1982")
        print(f"  Pre-1982:  DMI effect = {dmi_effect_pre:.2f} mm/std (p={pval_interact:.4f})")
        print(f"  Post-1982: DMI effect = {dmi_effect_post:.2f} mm/std")
        print(f"\n  This explains the regime flip observed in the split analysis.")
        print(f"  The climate system behavior shifted in 1982.")
    else:
        print(f"\n✗ NOT CONFIRMED: The DMI effect did NOT significantly change")
        print(f"  Both regimes have similar β₁ ≈ {b1_dmi_m5:.2f} mm/std")
        print(f"  The 1950–2013 pooled analysis is valid.")

    print(f"\nFor Paper 2:")
    print(f"  • Report both regime-specific AND pooled results")
    print(f"  • Discuss whether 1982 represents a climate regime shift")
    print(f"  • If β₄ is significant, emphasize temporal non-stationarity")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
