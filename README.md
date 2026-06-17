# Climate-State-Dependent Design Rainfall in Central Bangladesh
### Code and Data Supplement 

This repository accompanies the manuscript:

> **[Arshad, S.]** (2026). *Climate-state-dependent design rainfall
> and the reliability of stationary infrastructure design in Central Bangladesh.*
> Submitted to *Journal of Hydrology* (under review).

It contains the analysis scripts, derived datasets, and figures needed to understand
and substantially reproduce the results reported in the manuscript. It does **not**
contain the raw Bangladesh Meteorological Department (BMD) station rainfall series,
which BMD does not permit third-party redistribution of — see **Section 3** below for
exactly how to obtain it.

---

## 1. What's in this repository

```
.
├── README.md                          <- this file
├── LICENSE                            <- code license (MIT) — see Section 6
├── CITATION.cff                       <- machine-readable citation metadata
│
├── figures/                           <- all 9 manuscript figures, 300 dpi PNG
│   ├── Fig1.png   Study area map (verified BMD station locations, real Bangladesh border)
│   ├── Fig2.png   Model comparison (ΔAIC) + regional covariate coefficients
│   ├── Fig3.png   State-conditioned return levels (100-yr / 50-yr) vs IOD state
│   ├── Fig4.png   Recurrence compression curve (the 100-yr → 37-yr result)
│   ├── Fig5.png   Lifetime failure probability by climate regime
│   ├── Fig6.png   Engineering decision framework (verification + operations)
│   ├── Fig7.png   Diagnostic plots (scatter + QQ / probability plot)
│   ├── Fig8.png   Temporal stability audit (8 rolling 30-yr windows)
│   └── Fig9.png   Fixed-6-station panel audit (composition-confound test)
│
├── scripts/
│   ├── 01_data_extraction/
│   │   └── PAPER2_STAGE1_OND_Extraction.py    Builds OND annual-maximum series from
│   │                                          monthly station rainfall (input: raw BMD
│   │                                          file you must supply yourself — see §3)
│   │
│   ├── 02_model_fitting/
│   │   ├── PAPER2_STAGE2_NonStationary_GEV.py       Core nonstationary GEV (location ~ DMI)
│   │   ├── PAPER2_FINAL_CORRECT.py                   Full pipeline, OND block maxima, final form
│   │   ├── PAPER2_REGIONAL_DUAL_COVARIATE.py         DMI + Niño3.4 dual-covariate regional fits
│   │   └── PAPER2_REGIONAL_HETEROGENEITY_CORRECTED.py  Per-region coefficient comparison
│   │
│   ├── 03_robustness_audits/          <- one script per audit described in manuscript §5
│   │   ├── PAPER2_SENSITIVITY_LOO.py              Leave-one-out sensitivity
│   │   ├── PAPER2_INTERACTION_TERM_TEST.py         DMI×regime interaction test
│   │   ├── PAPER2_GEV_GOF_VALIDATION.py            KS / Anderson-Darling goodness-of-fit
│   │   ├── PAPER2_GPCC_VALIDATION.py                External validation against GPCC
│   │   ├── TEMPORAL_STABILITY_AUDIT.py             Rolling 30-yr window audit (dual-covariate)
│   │   ├── TEMPORAL_STABILITY_SINGLECOV.py         Rolling window audit (single-covariate)
│   │   ├── FIXED_PANEL_AUDITS.py                   Fixed-6-station composition-confound test
│   │   ├── PAPER2_SPATIAL_ROBUSTNESS.py            Station-subsampling robustness
│   │   ├── PAPER2_SOUTHWESTERN_AUDIT.py            Southwestern-region exclusion justification
│   │   └── PAPER2_REVIEWER_SANITY_CHECK.py         Consolidated sanity-check table
│   │
│   ├── 04_engineering_metrics/
│   │   ├── PAPER2_CLIMATE_STATE_RETURN_LEVELS_CORRECTED.py   Recurrence compression,
│   │   │                                                      equivalent return period
│   │   └── PAPER2_ENGINEERING_CI.py                Lifetime failure probability, point CIs
│   │
│   └── 05_figures/
│       ├── GENERATE_ALL_FIGURES_FIXED.py           Regenerates Figs 1–6, 8
│       └── MAKE_FIG7_DIAGNOSTICS.py                Regenerates Fig 7 (real-data diagnostics)
│
└── data/
    ├── public_climate_indices/        <- freely redistributable, sourced from public records
    │   ├── DMI_data.txt                NOAA HadISST-derived Dipole Mode Index, OND mean
    │   ├── nino34_monthly.txt          ERSSTv5 Niño 3.4 index, monthly
    │   ├── nino34_anomaly.txt          ERSSTv5 Niño 3.4 anomaly series
    │   ├── Z500_Tibetan_OND.csv        500-hPa geopotential height, Tibetan sector, OND
    │   └── bob_sst_jjas.csv            Bay of Bengal SST, JJAS mean
    │
    ├── derived_results/                <- this study's own computed outputs (shareable)
    │   ├── GEV_Parameters_Final.csv            Fitted GEV parameters by epoch
    │   ├── GEV_Return_Levels_Final.csv         Return levels by epoch
    │   ├── Station_Trend_Summary.csv           Mann-Kendall trend statistics per station
    │   ├── fixed_panel_audits.csv              Fixed-6-station audit results (Fig. 9 source)
    │   ├── temporal_stability_table.csv        Rolling-window audit results (Fig. 8 source)
    │   ├── bob_sst_vs_rainfall.csv             SST/rainfall pairing used in early exploration
    │   ├── final_bob_sst_vs_rainfall.csv       Final SST/rainfall pairing
    │   └── national_avg_monsoon_rainfall.csv   National-average monsoon rainfall series
    │
    └── restricted_NOT_INCLUDED/
        └── PLACEHOLDER_NOTE.txt        Explains what's missing and how to get it (§3)
```

---

## 2. Reproducing the analysis

The scripts are numbered in pipeline order. With the raw BMD file in hand (§3):

```bash
pip install numpy scipy pandas matplotlib statsmodels --break-system-packages

# 1. Build the OND annual-maximum series from raw monthly station data
python scripts/01_data_extraction/PAPER2_STAGE1_OND_Extraction.py

# 2. Fit the nonstationary GEV models (stationary, DMI-only, dual-covariate, regional)
python scripts/02_model_fitting/PAPER2_FINAL_CORRECT.py
python scripts/02_model_fitting/PAPER2_REGIONAL_DUAL_COVARIATE.py

# 3. Run the robustness audits described in manuscript Section 5
python scripts/03_robustness_audits/PAPER2_SENSITIVITY_LOO.py
python scripts/03_robustness_audits/TEMPORAL_STABILITY_AUDIT.py
python scripts/03_robustness_audits/FIXED_PANEL_AUDITS.py
# ...(remaining audit scripts are independent of one another and can be run in any order)

# 4. Translate audited model coefficients into engineering metrics
python scripts/04_engineering_metrics/PAPER2_CLIMATE_STATE_RETURN_LEVELS_CORRECTED.py
python scripts/04_engineering_metrics/PAPER2_ENGINEERING_CI.py

# 5. Regenerate all figures
python scripts/05_figures/GENERATE_ALL_FIGURES_FIXED.py
python scripts/05_figures/MAKE_FIG7_DIAGNOSTICS.py
```

Each script writes its numerical results to stdout and/or a CSV in the working
directory; cross-check these against the values quoted in the manuscript tables.
Random seeds are fixed where stochastic methods (e.g. multi-start optimization)
are used, so results should match to reported precision.

**Note on superseded scripts:** the original research process went through many
exploratory iterations (different block-maxima definitions, alternative covariates,
early drafts of each audit) before reaching the versions listed above. Only the
final, audited versions — the ones whose outputs are actually quoted in the
manuscript — are included here, to keep the repository reproducible and unambiguous.
If you need an intermediate version for any reason, contact the corresponding author.

---

## 3. Getting the raw rainfall data (BMD restriction)

The Bangladesh Meteorological Department (BMD) does not permit third-party
redistribution of its raw station-level rainfall records. This repository therefore
omits:

- `Station_Annual_Rainfall_Cleaned.csv` (cleaned annual series used as the modeling input)
- The original raw monthly station rainfall file

**To obtain this data:**

1. Contact BMD directly through their data request process (Bangladesh Meteorological
   Department, Agargaon, Dhaka — see bmd.gov.bd for current contact details), **or**
2. Contact the corresponding author of the manuscript, who can advise on the request
   process and confirm which 17 stations and which 1982–2013 / 1950–2013 windows are
   needed to reproduce this study exactly.

Once obtained, place the raw file at the path expected by
`scripts/01_data_extraction/PAPER2_STAGE1_OND_Extraction.py` (see the top of that
script for the exact expected filename and column format) and the pipeline in
Section 2 will reproduce the full analysis.

---

## 4. Figure-to-result mapping

| Figure | What it shows | Source script |
|---|---|---|
| Fig. 1 | Study area & station map | (static map; coordinates verified against BMD station registry) |
| Fig. 2 | Model comparison (ΔAIC) & regional coefficients | `02_model_fitting/PAPER2_REGIONAL_HETEROGENEITY_CORRECTED.py` |
| Fig. 3 | Return levels vs IOD state | `04_engineering_metrics/PAPER2_CLIMATE_STATE_RETURN_LEVELS_CORRECTED.py` |
| Fig. 4 | Recurrence compression (100-yr → 37-yr) | `04_engineering_metrics/PAPER2_CLIMATE_STATE_RETURN_LEVELS_CORRECTED.py` |
| Fig. 5 | Lifetime failure probability | `04_engineering_metrics/PAPER2_ENGINEERING_CI.py` |
| Fig. 6 | Engineering decision framework | (schematic; no script — built directly for the manuscript) |
| Fig. 7 | Diagnostics (scatter + QQ) | `05_figures/MAKE_FIG7_DIAGNOSTICS.py` |
| Fig. 8 | Temporal stability, 8 rolling windows | `03_robustness_audits/TEMPORAL_STABILITY_AUDIT.py` |
| Fig. 9 | Fixed-6-station panel audit | `03_robustness_audits/FIXED_PANEL_AUDITS.py` |

---

## 5. Citing this repository

If you use this code or the derived data, please cite both the manuscript and this
repository. A `CITATION.cff` file is included for automatic citation generation on
GitHub; once deposited on Zenodo, replace the DOI badge and citation block below with
the one Zenodo issues automatically on first release (see upload instructions).

```
[Arshad S.](2026). Code and data supplement for "Climate-state-dependent design
rainfall and the reliability of stationary infrastructure design in Central
Bangladesh." Zenodo. [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.20736485)
```

---

## 6. License

Code in this repository is released under the **MIT License** (see `LICENSE`).
Derived data files (`data/derived_results/`, `figures/`) are released under
**CC BY 4.0** — you may reuse them with attribution. Public climate-index files in
`data/public_climate_indices/` retain the license terms of their original source
(NOAA, ERSSTv5, HadISST); see each source's terms for reuse outside this repository.

---

## 7. Contact

Corresponding author: **[Sadman Arshad]**
Institution: Bangladesh University of Engineering and Technology (BUET)
Email: 2304052@ce.buet.ac.bd
For data-access requests, reproducibility questions, or to report an issue with
this repository, please open a GitHub Issue or contact the corresponding author
directly.
