"""
Statistical utility functions for evaluating ML models in sonic log prediction.

Author: [Rodrigo Brunetta]
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score


def calculate_metrics(y_true, y_pred):
    """
    Calculates standard regression metrics.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    dict : Dictionary with all metrics
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Residuals
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Maximum and minimum error
    max_error = np.max(np.abs(residuals))
    min_error = np.min(np.abs(residuals))
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape,
        'Mean_Residual': mean_residual,
        'Std_Residual': std_residual,
        'Max_Error': max_error,
        'Min_Error': min_error
    }


def cross_validation_analysis(model, X, y, cv=5, scoring='r2'):
    """
    Performs cross-validation and returns statistics.

    Parameters:
    -----------
    model : estimator
        ML model
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of folds
    scoring : str
        Evaluation metric

    Returns:
    --------
    dict : Dictionary with CV results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'cv_folds': cv,
        'metric': scoring
    }


def compare_models(models_dict, X_test, y_test):
    """
    Compares multiple trained models.

    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and trained models as values
    X_test : array-like
        Test features
    y_test : array-like
        Test target

    Returns:
    --------
    pd.DataFrame : DataFrame with comparative metrics
    """
    results = []
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        metrics['Model'] = model_name
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    df_results = df_results[['Model'] + [col for col in df_results.columns if col != 'Model']]
    
    return df_results


def statistical_test_models(model1_scores, model2_scores, alpha=0.05,
                            wells1=None, wells2=None):
    """
    Performs paired statistical tests to compare two models.

    Supports arrays of different sizes as long as well names are
    provided — alignment is performed automatically using the
    intersection of common wells.

    Parameters:
    -----------
    model1_scores : array-like
        Scores for model 1 (e.g., R² per well in LOWO)
    model2_scores : array-like
        Scores for model 2
    alpha : float
        Significance level (default: 0.05)
    wells1 : array-like, optional
        Well names corresponding to model1_scores
    wells2 : array-like, optional
        Well names corresponding to model2_scores

    Returns:
    --------
    dict : Results of paired t-test and Wilcoxon test
    """
    s1 = np.array(model1_scores)
    s2 = np.array(model2_scores)

    # Align by well if names provided
    if wells1 is not None and wells2 is not None:
        w1 = np.array(wells1)
        w2 = np.array(wells2)
        common = np.intersect1d(w1, w2)
        if len(common) == 0:
            raise ValueError("No common wells between the two models.")
        s1 = np.array([s1[w1 == w][0] for w in common])
        s2 = np.array([s2[w2 == w][0] for w in common])
        n_common = len(common)
    else:
        if len(s1) != len(s2):
            raise ValueError(
                f"Arrays with different sizes ({len(s1)} vs {len(s2)}). "
                "Provide wells1 and wells2 for automatic alignment by well."
            )
        n_common = len(s1)

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(s1, s2)

    # Wilcoxon test (more robust for non-normal distributions)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(s1, s2)
    except ValueError:
        # Occurs when all differences are zero
        w_stat, p_wilcoxon = 0.0, 1.0

    significant_ttest    = p_ttest    < alpha
    significant_wilcoxon = p_wilcoxon < alpha
    diff_mean = np.mean(s1) - np.mean(s2)

    return {
        # t-test
        't_statistic'         : t_stat,
        'p_value_ttest'        : p_ttest,
        'significant_ttest'    : significant_ttest,
        # Wilcoxon
        'w_statistic'          : w_stat,
        'p_value_wilcoxon'     : p_wilcoxon,
        'significant_wilcoxon' : significant_wilcoxon,
        # General
        'alpha'                : alpha,
        'n_wells'              : n_common,
        'mean_difference'      : diff_mean,
        # Consensus: significant if BOTH tests agree
        'significant'          : significant_ttest and significant_wilcoxon,
        'interpretation'       : (
            'Significant difference (t-test and Wilcoxon)'
            if significant_ttest and significant_wilcoxon
            else 'No significant difference'
        )
    }


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculates confidence interval for a metric.

    Parameters:
    -----------
    data : array-like
        Data (e.g., CV scores)
    confidence : float
        Confidence level

    Returns:
    --------
    dict : Confidence interval
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    interval = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    
    return {
        'mean': mean,
        'lower_bound': mean - interval,
        'upper_bound': mean + interval,
        'confidence_level': confidence,
        'margin_of_error': interval
    }


def residual_analysis(y_true, y_pred):
    """
    Detailed residual analysis.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    dict : Residual statistics
    """
    residuals = y_true - y_pred
    
    # Normality test (Shapiro-Wilk)
    # For large datasets, use a random sample
    if len(residuals) > 5000:
        sample_idx = np.random.choice(len(residuals), 5000, replace=False)
        sample_residuals = residuals[sample_idx]
    else:
        sample_residuals = residuals
    
    shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
    
    # Homoscedasticity test (visual analysis recommended)
    # Compute correlation between residuals and predicted values
    heteroscedasticity_corr, hetero_p = stats.pearsonr(y_pred, np.abs(residuals))
    
    # Residual percentiles
    percentiles = {
        'p5': np.percentile(residuals, 5),
        'p25': np.percentile(residuals, 25),
        'p50': np.percentile(residuals, 50),
        'p75': np.percentile(residuals, 75),
        'p95': np.percentile(residuals, 95)
    }
    
    return {
        'mean': np.mean(residuals),
        'median': np.median(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'shapiro_statistic': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'normality': 'Normal' if shapiro_p > 0.05 else 'Non-normal',
        'heteroscedasticity_corr': heteroscedasticity_corr,
        'heteroscedasticity_p': hetero_p,
        'percentiles': percentiles
    }


def calculate_adjusted_r2(r2, n_samples, n_features):
    """
    Calculates adjusted R².

    Parameters:
    -----------
    r2 : float
        Model R²
    n_samples : int
        Number of samples
    n_features : int
        Number of features

    Returns:
    --------
    float : Adjusted R²
    """
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2


def feature_importance_analysis(model, feature_names, top_n=None,
                                 X=None, y=None, n_repeats=10, random_state=42):
    """
    Extracts and analyzes feature importance.

    For models with feature_importances_ (XGBoost, LightGBM, RandomForest),
    uses the native attribute. For models without it (HistGradientBoosting),
    uses permutation importance — requires X and y.

    Parameters:
    -----------
    model : estimator
        Trained model
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to return
    X : array-like, optional
        Input data for permutation importance
    y : array-like, optional
        Target for permutation importance
    n_repeats : int
        Number of repetitions for permutation importance (default: 10)
    random_state : int
        Seed for permutation importance (default: 42)

    Returns:
    --------
    pd.DataFrame : DataFrame with features sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        if X is None or y is None:
            raise ValueError(
                "This model does not have feature_importances_. "
                "Provide X and y to compute permutation importance."
            )
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        importances = result.importances_mean

    total = importances.sum()
    df_importance = pd.DataFrame({
        'Feature'       : feature_names,
        'Importance'    : importances,
        'Importance_Pct': (importances / total * 100) if total > 0 else importances
    })

    df_importance = df_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

    if top_n:
        df_importance = df_importance.head(top_n)

    return df_importance


def bootstrap_confidence_interval(y_true, y_pred, metric_func=r2_score, 
                                  n_bootstrap=1000, confidence=0.95):
    """
    Calculates confidence interval using bootstrap.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    metric_func : function
        Metric function (e.g., r2_score, mean_squared_error)
    n_bootstrap : int
        Number of bootstrap iterations
    confidence : float
        Confidence level

    Returns:
    --------
    dict : Bootstrap confidence interval
    """
    n_samples = len(y_true)
    bootstrap_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    
    return {
        'mean': np.mean(bootstrap_scores),
        'median': np.median(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'lower_bound': lower,
        'upper_bound': upper,
        'confidence_level': confidence,
        'n_bootstrap': n_bootstrap,
        'values': bootstrap_scores
    }


def error_by_range(y_true, y_pred, n_bins=10):
    """
    Analyzes error by value range.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_bins : int
        Number of bins to split the data

    Returns:
    --------
    pd.DataFrame : Metrics by value range
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'error': np.abs(y_true - y_pred)
    })
    
    df['range'] = pd.cut(df['y_true'], bins=n_bins)
    
    results = df.groupby('range').agg({
        'y_true': ['count', 'mean'],
        'error': ['mean', 'std', 'max']
    }).round(4)
    
    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    results = results.reset_index()
    
    return results


def summary_report(y_true, y_pred, model_name="Model"):
    """
    Generates a complete model evaluation report.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Model name

    Returns:
    --------
    dict : Complete report
    """
    metrics = calculate_metrics(y_true, y_pred)
    residual_stats = residual_analysis(y_true, y_pred)
    ci = calculate_confidence_interval(y_pred, confidence=0.95)
    
    report = {
        'model_name': model_name,
        'n_samples': len(y_true),
        'metrics': metrics,
        'residual_analysis': residual_stats,
        'confidence_interval': ci
    }
    
    return report


def print_summary_report(report):
    """
    Prints the report in a formatted layout.

    Parameters:
    -----------
    report : dict
        Report generated by the summary_report function
    """
    print("=" * 60)
    print(f"EVALUATION REPORT - {report['model_name']}")
    print("=" * 60)
    print(f"\nNumber of samples: {report['n_samples']}")

    print("\n--- PERFORMANCE METRICS ---")
    for metric, value in report['metrics'].items():
        print(f"{metric:20s}: {value:.4f}")

    print("\n--- RESIDUAL ANALYSIS ---")
    for key, value in report['residual_analysis'].items():
        if key != 'percentiles':
            if isinstance(value, (int, float)):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")

    print("\n--- CONFIDENCE INTERVAL (95%) ---")
    ci = report['confidence_interval']
    print(f"Mean: {ci['mean']:.4f}")
    print(f"CI: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}]")
    print("=" * 60)


def analyze_lowo_results(results_df, predictions_df, algorithm_name):
    """
    Full analysis of LOWO results

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with columns ['Well_Name', 'R2', 'RMSE', 'MAE', ...]
    predicoes_df : pd.DataFrame
        DataFrame with columns ['Well_Name', 'DEPTH', 'DT_real', 'DT_pred', ...]
    algorithm_name : str
        Algorithm name (e.g., 'XGBoost', 'LightGBM')

    Returns:
    --------
    dict : Dictionary with all statistics
    """
    
    # Extract arrays
    r2_scores = results_df['R2'].values
    rmse_scores = results_df['RMSE'].values
    mae_scores = results_df['MAE'].values

    # Calculate residuals
    predictions_df['Residuals'] = predictions_df['DT_real'] - predictions_df['DT_pred']
    predictions_df['Abs_Residuals'] = np.abs(predictions_df['Residuals'])
    residuals = predictions_df['Residuals'].values
    abs_residuals = predictions_df['Abs_Residuals'].values
    
    # ========================================================
    # STATISTICS
    # ========================================================
    print("="*80)
    print(f"GENERAL STATISTICS - {algorithm_name}")
    print("="*80)

    print(f"\nR² Score:")
    print(f"   Mean:     {r2_scores.mean():.4f}")
    print(f"   Median:   {np.median(r2_scores):.4f}")
    print(f"   Std:      {r2_scores.std():.4f}")
    print(f"   Min:      {r2_scores.min():.4f} ({results_df.loc[r2_scores.argmin(), 'Well_Name']})")
    print(f"   Max:      {r2_scores.max():.4f} ({results_df.loc[r2_scores.argmax(), 'Well_Name']})")
    print(f"   Range:    {r2_scores.max() - r2_scores.min():.4f}")

    print(f"\nRMSE:")
    print(f"   Mean:     {rmse_scores.mean():.4f} μs/ft")
    print(f"   Median:   {np.median(rmse_scores):.4f} μs/ft")
    print(f"   Std:      {rmse_scores.std():.4f} μs/ft")
    print(f"   Min:      {rmse_scores.min():.4f} μs/ft")
    print(f"   Max:      {rmse_scores.max():.4f} μs/ft")

    print(f"\nMAE:")
    print(f"   Mean:     {mae_scores.mean():.4f} μs/ft")
    print(f"   Median:   {np.median(mae_scores):.4f} μs/ft")
    print(f"   Std:      {mae_scores.std():.4f} μs/ft")

    print(f"\nPerformance Distribution:")
    print(f"   Wells R² > 0.90: {(r2_scores > 0.90).sum():2d} ({(r2_scores > 0.90).sum()/len(r2_scores)*100:5.1f}%)")
    print(f"   Wells R² > 0.80: {(r2_scores > 0.80).sum():2d} ({(r2_scores > 0.80).sum()/len(r2_scores)*100:5.1f}%)")
    print(f"   Wells R² > 0.70: {(r2_scores > 0.70).sum():2d} ({(r2_scores > 0.70).sum()/len(r2_scores)*100:5.1f}%)")
    print(f"   Wells R² > 0.60: {(r2_scores > 0.60).sum():2d} ({(r2_scores > 0.60).sum()/len(r2_scores)*100:5.1f}%)")
    print(f"   Wells R² < 0.50: {(r2_scores < 0.50).sum():2d} ({(r2_scores < 0.50).sum()/len(r2_scores)*100:5.1f}%)")

    print(f"\nRESIDUAL ANALYSIS:")
    print(f"   Mean:     {residuals.mean():.4f} μs/ft {'(no bias)' if abs(residuals.mean()) < 0.5 else '(bias detected)'}")
    print(f"   Std:      {residuals.std():.4f} μs/ft")
    print(f"   Skewness: {pd.Series(residuals).skew():.4f} {'(symmetric)' if abs(pd.Series(residuals).skew()) < 0.5 else '(asymmetric)'}")
    print(f"   Kurtosis: {pd.Series(residuals).kurtosis():.4f}")

    print(f"\nAbsolute Error Percentiles:")
    print(f"   P50:  {np.percentile(abs_residuals, 50):.2f} μs/ft")
    print(f"   P75:  {np.percentile(abs_residuals, 75):.2f} μs/ft")
    print(f"   P90:  {np.percentile(abs_residuals, 90):.2f} μs/ft")
    print(f"   P95:  {np.percentile(abs_residuals, 95):.2f} μs/ft")
    print(f"   P99:  {np.percentile(abs_residuals, 99):.2f} μs/ft")

    # Normality test
    from scipy import stats
    sample_size = min(5000, len(residuals))
    _, p_value = stats.shapiro(np.random.choice(residuals, sample_size, replace=False))
    print(f"\nNormality Test (Shapiro-Wilk, n={sample_size}):")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Result: {'Residuals are normal' if p_value > 0.05 else 'Residuals are not normal'}")

    print(f"\nTop 5 Hardest Wells (Lowest R²):")
    worst_5 = results_df.nsmallest(5, 'R2')
    for i, (_, row) in enumerate(worst_5.iterrows(), 1):
        print(f"   {i}. {row['Well_Name']:25s} - R²={row['R2']:.3f}, RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

    print(f"\nTop 5 Easiest Wells (Highest R²):")
    best_5 = results_df.nlargest(5, 'R2')
    for i, (_, row) in enumerate(best_5.iterrows(), 1):
        print(f"   {i}. {row['Well_Name']:25s} - R²={row['R2']:.3f}, RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

    print("="*80)

    # ========================================================
    # RETURN STATISTICS DICTIONARY
    # ========================================================
    stats_dict = {
        'algorithm': algorithm_name,
        'r2_mean': r2_scores.mean(),
        'r2_median': np.median(r2_scores),
        'r2_std': r2_scores.std(),
        'r2_min': r2_scores.min(),
        'r2_max': r2_scores.max(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_median': np.median(rmse_scores),
        'rmse_std': rmse_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_median': np.median(mae_scores),
        'mae_std': mae_scores.std(),
        'residuals_mean': residuals.mean(),
        'residuals_std': residuals.std(),
        'residuals_skew': pd.Series(residuals).skew(),
        'p90_error': np.percentile(abs_residuals, 90),
        'p95_error': np.percentile(abs_residuals, 95),
        'n_wells_r2_gt_90': (r2_scores > 0.90).sum(),
        'n_wells_r2_gt_80': (r2_scores > 0.80).sum(),
        'n_wells_r2_gt_70': (r2_scores > 0.70).sum(),
        'n_wells_r2_lt_50': (r2_scores < 0.50).sum(),
        'shapiro_pvalue': p_value
    }
    
    return stats_dict


def print_summary_table(results_df):
    """
    Prints a summary table for all wells

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with LOWO results
    """
    summary_table = results_df[['Well_Name', 'R2', 'RMSE', 'MAE', 'N_samples_test']].copy()
    summary_table = summary_table.sort_values('R2', ascending=False)
    summary_table = summary_table.reset_index(drop=True)
    summary_table.index = summary_table.index + 1

    print("\nCOMPLETE TABLE - ALL WELLS (Sorted by R²)")
    print("="*80)
    print(summary_table.to_string())
    
    
def plot_cv_vs_test(results_df, algorithm_name):
    """
    Plots comparison between CV Score and Test Score

    Parameters:
    -----------
    results_df : pd.DataFrame
    algorithm_name : str
    """
    import matplotlib.pyplot as plt
    
    wells_ordered = results_df.sort_values('R2', ascending=False)['Well_Name'].values
    cv_scores = results_df.set_index('Well_Name').loc[wells_ordered, 'Best_CV_Score'].values
    test_scores = results_df.set_index('Well_Name').loc[wells_ordered, 'R2'].values
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    x = np.arange(len(wells_ordered))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cv_scores, width, label='CV Score (Train)',
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test Score (LOWO)', 
                   alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Wells (Sorted by R² Test)', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title(f'CV Score vs Test Score Comparison - {algorithm_name}',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(wells_ordered, rotation=90, ha='right', fontsize=7)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Gap statistics
    gap = cv_scores - test_scores
    print(f"\nGap CV → Test ({algorithm_name}):")
    print(f"   Mean:    {gap.mean():.4f}")
    print(f"   Median:  {np.median(gap):.4f}")
    print(f"   Std:     {gap.std():.4f}")
    print(f"\n   Interpretation:")
    if gap.mean() > 0.15:
        print("    Overfitting detected (CV >> Test)")
    elif gap.mean() < -0.05:
        print("    Underfitting or problematic training data (Test >> CV)")
    else:
        print("   Model generalizing adequately")
