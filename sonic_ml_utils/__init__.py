"""
sonic_ml_utils - Utility library for sonic log prediction with Machine Learning

Author: Rodrigo Brunetta
Version: 1.0.0
Date: December 2025

Modules:
--------
- plotting: Functions for result visualization and well log display
- statistics: Functions for model statistical analysis
- configs: Well log curve configurations

Usage Example:
---------------
>>> from sonic_ml_utils import plot_well, calculate_metrics
>>>
>>> # Plot well logs
>>> fig = plot_well(df_well, columns=['GR', 'RHOB', 'DT'])
>>> fig.show()
>>>
>>> # Calculate model metrics
>>> metrics = calculate_metrics(y_test, y_pred)
>>> print(f"R² = {metrics['R2']:.4f}")
"""

__version__ = '1.0.0'
__author__ = 'Rodrigo Brunetta'

# ==================== PLOTTING ====================
from .plotting import (
    plot_well,
    plot_prediction_vs_actual,
    plot_residuals,
    plot_cv_results,
    plot_feature_importance,
    plot_comparison_boxplot,
    plot_learning_curve,
    plot_well_profile_and_scatter,
    plot_petrophysical_diagnosis_panel,
    plot_well_profile_and_hexabin,
    plot_well_petrophysical_log,
    LITHO_COLORS,
    FALLBACK_COLOR,
    FORMATION_COLORS,
)

# ==================== STATISTICS ====================
from .statistics import (
    calculate_metrics,
    cross_validation_analysis,
    compare_models,
    statistical_test_models,
    calculate_confidence_interval,
    residual_analysis,
    calculate_adjusted_r2,
    feature_importance_analysis,
    bootstrap_confidence_interval,
    error_by_range,
    summary_report,
    print_summary_report
)

# ==================== CONFIGS ====================
from .configs import (
    CURVES_CONFIG_PATH,
    load_curves_config,
    get_curve_config,
    add_custom_curve
)

# ==================== __all__ ====================
# Defines what is exported when someone does: from sonic_ml_utils import *
__all__ = [
    # Plotting
    'plot_well',
    'plot_prediction_vs_actual',
    'plot_residuals',
    'plot_cv_results',
    'plot_feature_importance',
    'plot_comparison_boxplot',
    'plot_learning_curve',
    'plot_well_profile_and_scatter',
    'plot_petrophysical_diagnosis_panel',
    'plot_well_profile_and_hexabin',
    'plot_well_petrophysical_log',
    'LITHO_COLORS',
    'FALLBACK_COLOR',
    'FORMATION_COLORS',
    # Statistics
    'calculate_metrics',
    'cross_validation_analysis',
    'compare_models',
    'statistical_test_models',
    'calculate_confidence_interval',
    'residual_analysis',
    'calculate_adjusted_r2',
    'feature_importance_analysis',
    'bootstrap_confidence_interval',
    'error_by_range',
    'summary_report',
    'print_summary_report',
    
    # Configs
    'CURVES_CONFIG_PATH',
    'load_curves_config',
    'get_curve_config',
    'add_custom_curve'
]


# ==================== HELP INFORMATION ====================
def show_available_functions():
    """
    Shows all available functions in the library.
    """
    print("=" * 70)
    print("AVAILABLE FUNCTIONS IN sonic_ml_utils")
    print("=" * 70)
    print("\n📊 PLOTTING:")
    print("  - plot_well                     : Plots well logs")
    print("  - plot_prediction_vs_actual     : Predicted vs actual plot")
    print("  - plot_residuals                : Residual analysis")
    print("  - plot_cv_results               : Cross-validation results")
    print("  - plot_feature_importance       : Feature importance")
    print("  - plot_comparison_boxplot       : Model comparison")
    print("  - plot_learning_curve           : Learning curve")
    print("  - plot_well_profile_and_scatter : Well profile + scatter plot")
    print("  - plot_petrophysical_diagnosis_panel : Petrophysical diagnosis panel")
    print("  - LITHO_COLORS                  : Color dictionary for lithologies")
    print("  - FALLBACK_COLOR                : Default color for unknown lithologies")
    print("\n📈 STATISTICS:")
    print("  - calculate_metrics          : Calculates metrics (R², RMSE, MAE, etc)")
    print("  - cross_validation_analysis  : Cross-validation analysis")
    print("  - compare_models             : Compares multiple models")
    print("  - statistical_test_models    : Statistical test between models")
    print("  - residual_analysis          : Detailed residual analysis")
    print("  - feature_importance_analysis: Feature ranking")
    print("  - summary_report             : Complete model report")
    print("  - print_summary_report       : Prints formatted report")

    print("\n⚙️  CONFIGS:")
    print("  - load_curves_config         : Loads curve configurations")
    print("  - get_curve_config           : Gets config for a specific curve")
    print("  - add_custom_curve           : Adds a custom curve")

    print("\n" + "=" * 70)
    print("For help on a specific function, use: help(function_name)")
    print("Example: help(plot_well)")
    print("=" * 70)


# Adds the help function to __all__
__all__.append('show_available_functions')


# ==================== WELCOME MESSAGE ====================
def _welcome_message():
    """Welcome message (optional — comment out if not needed)."""
    import sys
    # Only displayed when imported interactively (not in scripts)
    if hasattr(sys, 'ps1'):
        print(f"\n✓ sonic_ml_utils v{__version__} loaded successfully!")
        print("  Use show_available_functions() to see all available functions.")

# Uncomment the line below if you want the welcome message
# _welcome_message()
