# Notebooks

Execute in numerical order. Each notebook saves its results to `../results/`.

## Execution Order

**`00_well_metadata.ipynb`** — Well inventory: names, depth ranges, sample counts, lithology distribution.

**`01_exploratory_analysis.ipynb`** — Descriptive statistics, log distributions, cross-plots, outlier detection (IQR), and generation of the main processed dataset.

**`02_lazypredict_screening.ipynb`** — Rapid multi-algorithm screening with LazyPredict to identify the most promising candidates.

**`03_lithology_integration.ipynb`** — Lithology classification integration and quality control; produces `wells_iqr_with_lithology.csv`.

**`04_xgboost_lowo.ipynb`** — XGBoost with LOWO validation. Hyperparameter tuning via RandomizedSearch + GroupKFold(10).

**`05_lightgbm_lowo.ipynb`** — LightGBM with LOWO validation.

**`06_histgb_lowo.ipynb`** — HistGradientBoosting (scikit-learn) with LOWO validation.

**`07_randomforest_lowo.ipynb`** — Random Forest with LOWO validation.

**`08_algorithms_comparison.ipynb`** — Statistical comparison of the four ML models: boxplots, Wilcoxon tests, ranking.

**`09_neural_networks_lowo.ipynb`** — MLP and CNN-1D (PyTorch) with LOWO validation and learning curve analysis.

**`10_all_methods_comparison_V1.ipynb`** — Full benchmark: all 9 methods (4 ML + 2 NN + 3 empirical), summary table, and statistical tests.

**`11_xgboost_sensitivity_analysis.ipynb`** — Sensitivity analysis of XGBoost hyperparameters and feature importance.

**`12_directional_data.ipynb`** — Analysis of directional surveys and well trajectory effects on prediction performance.

**`13_geological_hypotheses.ipynb`** — Article 2 core analysis. Tests hypotheses H1–H8 linking geological factors to prediction error.

**`14_well_atlas.ipynb`** — Visual reference atlas for all 27 wells: petrophysical logs, formation columns, and error profiles.

## Requirements

```bash
conda activate geo
# or
pip install -r ../requirements.txt
pip install -e ..
```

## Common Issues

**`ModuleNotFoundError: No module named 'sonic_ml_utils'`** — Run `pip install -e .` from the project root.

**`FileNotFoundError: data/processed/wells_iqr_with_lithology.csv`** — Run notebooks 01–03 first, or unzip `wells_iqr_with_lithology.zip`.
