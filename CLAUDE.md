# Sonic Log Prediction — PhD Project

## Project Overview

PhD thesis in Geophysics: predicting sonic log (DT) using Machine Learning for
automated well-to-seismic tie in the Sergipe-Alagoas Basin, Brazil.
Benchmark study comparing 9 prediction methods with LOWO validation across 32 wells.

## Environment

- Conda environment: `geo`
- Python 3.8+, PyTorch (NNs), scikit-learn (ML), Plotly (visualizations)
- OS: Windows — use backslash-safe paths or pathlib when creating new code

## Project Structure

```
sonic_log_prediction/
├── data/processed/                    # wells_iqr_with_lithology.csv (main dataset)
├── notebooks/                         # Numbered 01-12, sequential workflow
│   └── archive/                       # Old experimental notebooks
├── results/
│   ├── <algorithm>/                   # xgboost, lightgbm, histgb, randomforest, mlp, cnn
│   │   ├── figures/
│   │   ├── metrics/
│   │   ├── models/
│   │   └── predictions/               # lowo_<algo>_predictions.csv
│   ├── empirical/                     # Gardner, Wyllie, RHG
│   ├── comparison/                    # Cross-method analysis (notebook 12)
│   ├── lithology/                     # Lithology statistics CSVs
│   └── archive/                       # Old experimental results
├── sonic_ml_utils/                    # Custom library
│   ├── configs/config_curves.json
│   ├── plotting.py                    # plot_well(), plot_well_profile_and_scatter()
│   └── statistics.py                  # calculate_metrics(), statistical_test_models()
├── docs/
├── scripts/
└── tests/
```

## Dataset

- 32 wells, ~230k samples
- Features: GR, RT90, RHOB, NPHI (CALI excluded — indicates measurement quality, not formation)
- Target: DT (sonic log, μs/ft)
- Lithology: 83% shale, 17% other
- Empirical predictions stored in main dataset as: DT_gardner, DT_wyllie, DT_rhg

## Prediction CSV Format (all methods)

```
DEPTH, Well_Name, DT_real, DT_pred, GR, RT90, RHOB, NPHI
```

## Methods and Results (R² mean LOWO)

- HistGradientBoosting: ~0.73 (best)
- LightGBM: ~0.73
- RandomForest: ~0.73
- XGBoost: ~0.69
- MLP: ~0.52
- CNN 1D: ~0.36
- RHG: -1.81 | Wyllie: -6.33 | Gardner: -285.92

## Key Rules

- NEVER modify CSVs in `data/processed/`
- NEVER delete files in `results/<algorithm>/predictions/`
- Use relative paths (`../results/`, `../data/`) in notebooks
- LOWO (Leave-One-Well-Out) is the primary validation strategy
- Empirical methods have no training phase — "LOWO" structure is for consistent per-well evaluation only
- All notebooks should use `from sonic_ml_utils.plotting import plot_well` for well visualization
- Plotting config is in `sonic_ml_utils/configs/config_curves.json`

## Code Conventions

- Functions in sonic_ml_utils follow sklearn-style API
- Docstrings in Portuguese or English (consistent within file)
- Use pandas for data manipulation, plotly for interactive plots, matplotlib for static figures
- Hyperparameter tuning uses continuous distributions with GroupKFold (cv=10)
