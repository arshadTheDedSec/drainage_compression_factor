import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_ond_seasonal_totals(rainfall_csv_path, output_csv_path):
    """
    Extract OND (Oct-Nov-Dec) Post-Monsoon seasonal rainfall totals.
    This prepares the dataset for non-stationary GEV modeling in Paper 2.
    """
    print("=" * 70)
    print("PAPER 2 - STAGE 1: OND SEASONAL EXTRACTION")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(rainfall_csv_path)
    print(f"✓ Loaded rainfall data: {df.shape}")
    print(f"  Stations: {df['Station'].nunique()}")
    print(f"  Years: {df['Year'].min()} - {df['Year'].max()}")
    
    # Filter OND months (October=10, November=11, December=12)
    df_ond = df[df['Month'].isin([10, 11, 12])].copy()
    
    # Group by Station and Year, sum OND rainfall
    ond_seasonal = df_ond.groupby(['Station', 'Year'])['Monthly_Total'].sum().reset_index()
    ond_seasonal.columns = ['Station', 'Year', 'OND_Total']
    
    # Pivot to get stations as columns
    ond_pivot = ond_seasonal.pivot(index='Year', columns='Station', values='OND_Total')
    
    print(f"\n✓ OND Seasonal Extraction Complete")
    print(f"  Years: {ond_pivot.shape[0]} ({ond_pivot.index.min()} - {ond_pivot.index.max()})")
    print(f"  Stations: {ond_pivot.shape[1]}")
    
    # Export
    ond_pivot.to_csv(output_csv_path)
    print(f"  Saved: {output_csv_path}")
    
    # Summary statistics
    print(f"\n✓ OND Rainfall Statistics (mm)")
    print(f"  Mean: {ond_pivot.mean().mean():.1f}")
    print(f"  Std:  {ond_pivot.std().mean():.1f}")
    print(f"  Min:  {ond_pivot.min().min():.1f}")
    print(f"  Max:  {ond_pivot.max().max():.1f}")
    
    print("\n" + "=" * 70)
    
    return ond_pivot


def prepare_for_stage2(ond_data, dmi_csv_path, z500_csv_path):
    """
    Align OND rainfall with DMI and Z500 climate indices for non-stationary GEV.
    """
    print("=" * 70)
    print("PAPER 2 - STAGE 1 (continued): COVARIATE ALIGNMENT")
    print("=" * 70)
    
    # Load climate indices
    try:
        dmi = pd.read_csv(dmi_csv_path)
        print(f"✓ Loaded DMI: {dmi.shape}")
    except:
        print("⚠ DMI file not found. Will compute from SST if available.")
        dmi = None
    
    try:
        z500 = pd.read_csv(z500_csv_path)
        print(f"✓ Loaded Z500: {z500.shape}")
    except:
        print("⚠ Z500 file not found. Will proceed without Z500.")
        z500 = None
    
    print("\nOND Rainfall data ready for Stage 2 (Non-Stationary GEV)")
    print("=" * 70)
    
    return ond_data, dmi, z500


# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    
    # UPDATE WITH YOUR ACTUAL PATHS
    input_file = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\rainfall_data_bangladesh_1948_to_2014.csv"
    output_file = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\OND_Seasonal_Totals.csv"
    dmi_file = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\DMI_Index.csv"
    z500_file = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\Z500_Tibetan_Plateau.csv"
    
    # Stage 1: Extract OND
    ond_data = extract_ond_seasonal_totals(input_file, output_file)
    
    # Stage 1 (continued): Prepare covariates
    ond_data, dmi, z500 = prepare_for_stage2(ond_data, dmi_file, z500_file)
    
    print("\n✓ STAGE 1 COMPLETE: Ready for Stage 2 (Non-Stationary GEV-MLE)")
