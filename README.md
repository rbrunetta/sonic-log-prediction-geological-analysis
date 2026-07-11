# Sonic Log Prediction using Machine Learning

> PhD thesis project — Geophysics | Universidade Federal de Sergipe  
> Benchmark study for automated well-to-seismic tie in the Sergipe-Alagoas Basin, Brazil

## Overview

This repository accompanies a doctoral thesis in Geophysics investigating the prediction of sonic logs (DT, µs/ft) from standard petrophysical curves (GR, RT90, RHOB, NPHI) using Machine Learning. The study benchmarks 9 prediction methods under Leave-One-Well-Out (LOWO) validation across 27 wells in the Sergipe-Alagoas Basin.

**Key results (mean LOWO R²):**

| Method | R² |
|---|---|
| HistGradientBoosting | 0.73 |
| LightGBM | 0.73 |
| Random Forest | 0.73 |
| XGBoost | 0.69 |
| MLP | 0.52 |
| CNN-1D | 0.36 |
| RHG (empirical) | −1.81 |
| Wyllie (empirical) | −6.33 |
| Gardner (empirical) | −285.92 |

## Repository Structure

```
sonic_log_prediction/
├── notebooks/                      # Analysis notebooks (sequential workflow)
│   ├── 00_well_metadata.ipynb      # Well inventory and metadata
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_lazypredict_screening.ipynb
│   ├── 03_lithology_integration.ipynb
│   ├── 04_xgboost_lowo.ipynb
│   ├── 05_lightgbm_lowo.ipynb
│   ├── 06_histgb_lowo.ipynb
│   ├── 07_randomforest_lowo.ipynb
│   ├── 08_algorithms_comparison.ipynb
│   ├── 09_neural_networks_lowo.ipynb   # MLP + CNN-1D
│   ├── 10_all_methods_comparison_V1.ipynb
│   ├── 11_xgboost_sensitivity_analysis.ipynb
│   ├── 12_directional_data.ipynb
│   ├── 13_geological_hypotheses.ipynb  # Article 2: H1–H8 framework
│   └── 14_well_atlas.ipynb             # Visual atlas for all 27 wells
│
├── sonic_ml_utils/                 # Custom Python library
│   ├── plotting.py                 # Well visualization functions
│   ├── statistics.py               # Metrics and statistical tests
│   └── configs/config_curves.json  # Log curve display configuration
│
├── data/
│   ├── processed/
│   │   ├── wells_iqr_with_lithology.csv   # Main dataset (27 wells, ~232k samples)
│   │   ├── wells_iqr_with_lithology.zip   # Compressed version
│   │   └── directional_wells.csv          # Well trajectory data
│   ├── formations.csv              # Stratigraphic intervals per well
│   └── annotations/               # Per-well interpretive notes (JSON)
│
├── results/
│   ├── xgboost/                   # predictions/, metrics/, models/
│   ├── lightgbm/
│   ├── histgb/
│   ├── randomforest/
│   ├── mlp/
│   ├── cnn/
│   ├── empirical/                 # Gardner, Wyllie, RHG
│   ├── comparison/                # Cross-method metrics and tables
│   └── directional/
│
├── scripts/
├── tests/
├── docs/
├── environment.yml                # Conda environment (name: geo)
├── requirements.txt
├── setup.py
└── CITATION.cff
```

## Dataset

The processed dataset (`data/processed/wells_iqr_with_lithology.csv`) was built from LAS files sourced from the [ANP public well database](https://reate.cprm.gov.br/anp/). Raw LAS files are included in `data/raw/LAS/`.

| Field | Description |
|---|---|
| `Well_Name` | Well identifier (27 wells) |
| `DEPTH` | Measured depth (m) |
| `GR` | Gamma Ray (API) |
| `RT90` | Deep resistivity (ohm·m) |
| `RHOB` | Bulk density (g/cm³) |
| `NPHI` | Neutron porosity (pu) |
| `DT` | Sonic log — target variable (µs/ft) |
| `Lithology` | Interpreted lithology |
| `DT_gardner` / `DT_wyllie` / `DT_rhg` | Empirical predictions |

Dominant lithology: 83% shale, 17% other. Depth range: ~2,000–6,000 m.

## Getting Started

### 1. Environment setup

```bash
# Clone the repository
git clone https://github.com/rbrunetta/sonic_log_prediction.git
cd sonic_log_prediction

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate geo

# Or using pip
pip install -r requirements.txt

# Install the custom library
pip install -e .
```

### 2. Run the notebooks

Execute notebooks in numerical order. Each saves its outputs to `results/`.

```
00 → Well inventory
01 → Exploratory analysis
02 → Algorithm screening (LazyPredict)
03 → Lithology integration
04–07 → XGBoost, LightGBM, HistGB, Random Forest (LOWO)
08 → Comparison of ML methods
09 → Neural networks (MLP + CNN-1D)
10 → Full comparison (all 9 methods)
11 → XGBoost sensitivity analysis
12 → Directional well analysis
13 → Geological hypotheses (H1–H8)
14 → Well atlas (visual reference)
```

### 3. Use the custom library

```python
from sonic_ml_utils import (
    plot_well_profile_and_scatter,
    plot_well_petrophysical_log,
    plot_petrophysical_diagnosis_panel,
    calculate_metrics,
    LITHO_COLORS,
    FORMATION_COLORS,
)
```

## Geological Hypotheses (Notebook 13)

Notebook 13 investigates 8 hypotheses (H1–H8) linking prediction performance to geological factors:

| # | Hypothesis | Driver |
|---|---|---|
| H1 | Depth | Compaction and burial |
| H2 | DT extrapolation | Out-of-range target values |
| H3 | Feature space extrapolation | Out-of-range input features |
| H4 | Geological formation | Stratigraphic unit |
| H5 | Lithology | Rock type |
| H6 | Layer thickness | Thin-bed effects |
| H7 | Well directionality | Deviated wells |
| H8 | Petrophysical regime | Diagenesis and mineralogy |

## Validation Strategy

All ML models use **Leave-One-Well-Out (LOWO)**: each well is held out as the test set while the remaining 26 wells form the training set. This simulates real-world deployment where the model must generalize to unseen wells. Hyperparameters are tuned with `GroupKFold(n_splits=10)` on the training split.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@software{brunetta2026sonic,
  author = {Brunetta, Rodrigo},
  title  = {Sonic Log Prediction using Machine Learning — Sergipe-Alagoas Basin},
  year   = {2026},
  url    = {https://github.com/rbrunetta/sonic_log_prediction}
}
```

See also `CITATION.cff` for the structured citation format.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Rodrigo Brunetta**  
PhD candidate in Geophysics | Universidade Federal de Sergipe  
rbrunetta.colab@gmail.com
