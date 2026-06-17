import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class NonStationaryGEV:
    """
    Non-Stationary GEV Modeling with IOD (DMI) and Z500 covariates.
    μ(t) = β0 + β1*DMI(t) + β2*Z500(t)
    σ, ξ constant (stationary)
    """
    
    def __init__(self, rainfall_data, dmi_data, z500_data):
        """
        rainfall_data: pd.Series or np.array (annual max OND rainfall)
        dmi_data: pd.Series (IOD index, same length as rainfall)
        z500_data: pd.Series (Z500 anomaly, same length as rainfall)
        """
        self.y = np.array(rainfall_data)
        self.dmi = np.array(dmi_data)
        self.z500 = np.array(z500_data)
        self.n = len(self.y)
        
        # Standardize covariates
        self.dmi_std = (self.dmi - np.mean(self.dmi)) / np.std(self.dmi)
        self.z500_std = (self.z500 - np.mean(self.z500)) / np.std(self.z500)
        
        self.results = {}
        
    def gev_logpdf(self, x, mu, sigma, xi):
        """GEV log probability density function"""
        if sigma <= 0:
            return -np.inf
        
        z = (x - mu) / sigma
        
        if xi == 0:  # Gumbel
            return -np.log(sigma) - z - np.exp(-z)
        else:
            t = 1 + xi * z
            if np.any(t <= 0):
                return -np.inf
            return -np.log(sigma) - (1/xi + 1) * np.log(t) - t**(-1/xi)
    
    def loglikelihood_stationary(self, params):
        """LL for stationary GEV (baseline)"""
        mu, sigma, xi = params
        if sigma <= 0:
            return 1e10
        
        ll = 0
        for i in range(self.n):
            ll += self.gev_logpdf(self.y[i], mu, sigma, xi)
        
        return -ll  # Negative for minimization
    
    def loglikelihood_nonstationary(self, params):
        """LL for non-stationary GEV: μ(t) = β0 + β1*DMI + β2*Z500"""
        beta0, beta1, beta2, sigma, xi = params
        
        if sigma <= 0:
            return 1e10
        
        ll = 0
        for i in range(self.n):
            mu_t = beta0 + beta1 * self.dmi_std[i] + beta2 * self.z500_std[i]
            ll += self.gev_logpdf(self.y[i], mu_t, sigma, xi)
        
        return -ll  # Negative for minimization
    
    def fit_stationary(self):
        """Fit stationary GEV model"""
        print("\n" + "=" * 70)
        print("FITTING STATIONARY GEV (BASELINE)")
        print("=" * 70)
        
        # Initial guess
        mu0 = np.median(self.y)
        sigma0 = np.std(self.y)
        xi0 = 0.1
        
        bounds = [(mu0-200, mu0+200), (sigma0/2, sigma0*3), (-0.5, 0.5)]
        
        result = minimize(
            self.loglikelihood_stationary,
            [mu0, sigma0, xi0],
            method='L-BFGS-B',
            bounds=bounds
        )
        
        mu_stat, sigma_stat, xi_stat = result.x
        ll_stat = -result.fun
        
        print(f"✓ Converged: {result.success}")
        print(f"  μ = {mu_stat:.2f}")
        print(f"  σ = {sigma_stat:.2f}")
        print(f"  ξ = {xi_stat:.4f}")
        print(f"  Log-Likelihood = {ll_stat:.2f}")
        
        self.results['stationary'] = {
            'params': (mu_stat, sigma_stat, xi_stat),
            'll': ll_stat,
            'aic': 2*3 - 2*ll_stat,
            'bic': 3*np.log(self.n) - 2*ll_stat
        }
        
        return mu_stat, sigma_stat, xi_stat
    
    def fit_nonstationary(self):
        """Fit non-stationary GEV with DMI and Z500 covariates"""
        print("\n" + "=" * 70)
        print("FITTING NON-STATIONARY GEV (μ(t) = β0 + β1*DMI + β2*Z500)")
        print("=" * 70)
        
        # Initial guess from stationary
        mu_init, sigma_init, xi_init = self.fit_stationary()
        
        beta0_init = mu_init
        beta1_init = 10
        beta2_init = 10
        
        bounds = [
            (beta0_init - 100, beta0_init + 100),  # β0
            (-100, 100),                             # β1
            (-100, 100),                             # β2
            (sigma_init/2, sigma_init*3),           # σ
            (-0.5, 0.5)                              # ξ
        ]
        
        result = minimize(
            self.loglikelihood_nonstationary,
            [beta0_init, beta1_init, beta2_init, sigma_init, xi_init],
            method='L-BFGS-B',
            bounds=bounds
        )
        
        beta0, beta1, beta2, sigma, xi = result.x
        ll_ns = -result.fun
        
        print(f"\n✓ Converged: {result.success}")
        print(f"  β0 (intercept) = {beta0:.2f}")
        print(f"  β1 (DMI coef) = {beta1:.4f}")
        print(f"  β2 (Z500 coef) = {beta2:.4f}")
        print(f"  σ = {sigma:.2f}")
        print(f"  ξ = {xi:.4f}")
        print(f"  Log-Likelihood = {ll_ns:.2f}")
        
        self.results['nonstationary'] = {
            'params': (beta0, beta1, beta2, sigma, xi),
            'll': ll_ns,
            'aic': 2*5 - 2*ll_ns,
            'bic': 5*np.log(self.n) - 2*ll_ns
        }
        
        return beta0, beta1, beta2, sigma, xi
    
    def likelihood_ratio_test(self):
        """LRT: Non-stationary vs Stationary"""
        print("\n" + "=" * 70)
        print("LIKELIHOOD RATIO TEST (Non-Stationary vs Stationary)")
        print("=" * 70)
        
        ll_ns = self.results['nonstationary']['ll']
        ll_stat = self.results['stationary']['ll']
        
        Lambda = 2 * (ll_ns - ll_stat)  # Positive when NS is better
        p_value = 1 - stats.chi2.cdf(Lambda, df=2)  # df = 2 extra parameters
        
        print(f"  Lambda = {Lambda:.4f}")
        print(f"  χ²(df=2) critical value (α=0.05) = {stats.chi2.ppf(0.95, df=2):.4f}")
        print(f"  p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT: Non-stationary model is better (p < 0.05)")
        else:
            print(f"  ✗ NOT significant: Stationary model is sufficient")
        
        self.results['lrt'] = {'Lambda': Lambda, 'p_value': p_value}
    
    def model_comparison(self):
        """Compare AIC and BIC"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON (AIC / BIC)")
        print("=" * 70)
        
        aic_stat = self.results['stationary']['aic']
        aic_ns = self.results['nonstationary']['aic']
        bic_stat = self.results['stationary']['bic']
        bic_ns = self.results['nonstationary']['bic']
        
        print(f"{'Model':<20} {'AIC':<12} {'BIC':<12}")
        print("-" * 44)
        print(f"{'Stationary':<20} {aic_stat:<12.2f} {bic_stat:<12.2f}")
        print(f"{'Non-Stationary':<20} {aic_ns:<12.2f} {bic_ns:<12.2f}")
        print(f"{'ΔAIc':<20} {aic_ns-aic_stat:<12.2f}")
        print(f"{'ΔBIC':<20} {bic_ns-bic_stat:<12.2f}")
        print("-" * 44)
        
        if aic_ns < aic_stat:
            print("✓ Non-Stationary model preferred (lower AIC)")
        else:
            print("✗ Stationary model preferred (lower AIC)")


# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    
    # LOAD DATA
    print("Loading OND rainfall + climate indices...")
    
    # UPDATE WITH YOUR PATHS
    ond_file = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\OND_Seasonal_Totals.csv"
    
    # Load OND data
    ond_data = pd.read_csv(ond_file, index_col=0)
    print(f"✓ OND data: {ond_data.shape}")
    
    # For now, use national average or first station
    rainfall_series = ond_data.mean(axis=1)  # National average
    print(f"✓ Using national average OND rainfall: {len(rainfall_series)} years")
    
    # You'll need to provide DMI and Z500
    # For now, create dummy data (REPLACE WITH ACTUAL)
    years = rainfall_series.index.values
    dmi_data = np.sin(np.arange(len(years))/10) * 0.5  # Placeholder
    z500_data = np.cos(np.arange(len(years))/10) * 50  # Placeholder
    
    print("\n⚠ WARNING: Using placeholder DMI and Z500 data")
    print("  Replace with actual DMI and Z500 indices from your files")
    
    # FIT MODELS
    gev = NonStationaryGEV(rainfall_series.values, dmi_data, z500_data)
    gev.fit_stationary()
    gev.fit_nonstationary()
    gev.likelihood_ratio_test()
    gev.model_comparison()
    
    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETE: Ready for Stage 3 (IDF Curves)")
    print("=" * 70)
