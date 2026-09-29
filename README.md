# Pipeline Health Check — EDA, Corruption & Distribution Shift

**Objective:** Diagnose and repair data-quality failures in a merged country-year panel using exploratory data analysis alone, then build monitoring tools that detect distribution shift between training and inference data.

## Methodology
- Profiled a 260-row, 10-country panel (2000–2023) merged from three sources with a four-part checklist: structure, distributions, completeness, and domain ranges.
- Identified and fixed 5 planted data-quality issues: sign-flipped GDP, life expectancy recorded in months, near-duplicate country-year rows, mixed decimal/percentage units and impossible values in trade share, and a thousands-vs-millions GDP unit mismatch. The cleaned panel has 230 rows and passes all validation checks.
- Measured train-vs-inference drift with the Population Stability Index (PSI) and calibrated alarm levels against a permutation null distribution.
- Compared manual EDA with automated profiling (ydata-profiling) on the corrupted data.
- Packaged reusable functions in `eda_utils.py` (domain-constraint checks, PSI, EDA summary) and built an interactive ipywidgets/plotly pipeline-health dashboard.

## Key Findings
- GDP PSI = **2.4589** after a simulated 1.3× shift, far above the null 95th percentile (≈0.33).
- Undrifted columns still scored PSI 0.09–0.23 at n = 138 vs 92: the conventional 0.10/0.25 thresholds are unreliable at small sample sizes and should be replaced by a null-calibrated alarm.
- Automated profiling raised no alert for any of the five corruptions; near-duplicates (0 exact duplicates) and the GDP unit mismatch required domain knowledge of the panel key and units.
- Min/max range checks caught 36 corrupted rows but missed 80 values that were wrong yet in range (50 GDP rows in thousands, 30 trade shares stored as decimals).

## Files
- `lab-ch03-diagnostic.ipynb` — full analysis with outputs
- `eda_utils.py` — reusable data-quality and drift-detection module
