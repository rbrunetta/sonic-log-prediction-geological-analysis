"""
Plotting functions for ML results in sonic log prediction.

Author: Rodrigo Brunetta
Date: December 2025

Bilingual support: pass lang='pt' to any function to get Portuguese labels.
Default is lang='en' (English).
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

# ==============================================================================
# BILINGUAL LABELS DICTIONARY
# ==============================================================================
_LABELS = {
    'en': {
        # General
        'depth'             : 'Depth (m)',
        'dt_measured'       : 'DT Measured (µs/ft)',
        'dt_predicted'      : 'DT Predicted (µs/ft)',
        'dt_well'           : 'DT Well (µs/ft)',
        'dt_well_label'     : 'DT Well',
        'dt_pred_label'     : 'DT Predicted',
        'dt_axis'           : 'DT (μs/ft)',
        'residual'          : 'Residual (µs/ft)',
        'abs_error'         : 'Absolute Error (µs/ft)',
        'frequency'         : 'Frequency',
        'importance'        : 'Importance',
        'score'             : 'Score',
        'fold'              : 'Fold',
        'sample_count'      : 'Sample Count',
        'data'              : 'Data',
        'ideal_line'        : '1:1 line (ideal)',
        'ref_line'          : '1:1 reference',
        'bin_median'        : 'Bin median',
        'training'          : 'Training (26 wells)',
        # plot_prediction_vs_actual
        'pred_vs_actual_title' : 'Predicted vs Actual',
        'ideal_line_label'     : 'Ideal Line',
        # plot_residuals
        'residuals_title'      : 'Residual Analysis',
        'res_vs_pred'          : 'Residuals vs Predicted',
        'res_dist'             : 'Residual Distribution',
        'mean_label'           : 'Mean',
        'std_label'            : 'Std',
        # plot_cv_results
        'cv_title'             : 'Cross-Validation Results - {}',
        'mean_label_cv'        : 'Mean: {:.4f}',
        'mean_std_text'        : 'Mean ± Std = {:.4f} ± {:.4f}',
        # plot_feature_importance
        'feat_imp_title'       : 'Top {} Most Important Features',
        # plot_comparison_boxplot
        'model_comp_title'     : 'Model Comparison - {}',
        # plot_learning_curve
        'lc_title'             : 'Learning Curve',
        'lc_train'             : 'Train',
        'lc_val'               : 'Validation',
        'lc_xlabel'            : 'Training Set Size',
        # plot_well_profile_and_scatter / hexabin
        'plotting_wells'       : 'Plotting {} well(s)...',
        'well_not_found'       : "⚠️  Well '{}' not found in the DataFrame!",
        'well_saved'           : '[{}/{}] {} - Saved: {}',
        'well_r2'              : '[{}/{}] {} - R²={:.4f}',
        'trend_label'          : 'Trend: y={:.3f}x+{:.3f}',
        'trend_label2'         : 'Trend (y={:.3f}x{:+.2f})',
        'mae_global'           : 'MAE global = {:.2f}',
        'mae_bin'              : 'MAE median (200m bin)',
        # plot_petrophysical_diagnosis_panel
        'petro_space_title'    : 'Petrophysical Space RHOB × NPHI\n(gray = training | color = absolute error)',
        'outside_training'     : 'Outside training P5-P95:\n  NPHI: {:.1f}%\n  RHOB: {:.1f}%',
        'dt_nphi_title'        : 'DT Measured × NPHI by Lithology\n(velocity-porosity relationship)',
        'dt_dt_title'          : 'DT Measured × DT Predicted\n(color = NPHI; higher NPHI → higher porosity)',
        'err_depth_title'      : 'Absolute Error × Depth\n(color = RHOB; green = compacted | red = porous)',
        'spearman_nphi_dt'     : 'Spearman ρ(NPHI×DT) = {:+.3f}  (p={:.3f})',
        'spearman_box'         : 'ρ(depth×error) = {:+.3f}  (p={:.3f})\nρ(NPHI×error) = {:+.3f}  (p={:.3f})\nρ(RHOB×error) = {:+.3f}  (p={:.3f})',
        'cb_abs_error'         : 'Absolute Error (µs/ft)',
        'cb_rhob'              : 'RHOB (g/cm³)',
        'cb_nphi'              : 'NPHI (pu)',
        'nphi_axis'            : 'NPHI (pu)',
        'rhob_axis'            : 'RHOB (g/cm³)',
        'well_not_found_pred'  : "⚠️  Well '{}' not found in df_pred.",
        'saved'                : 'Saved: {}',
    },
    'pt': {
        # General
        'depth'             : 'Profundidade (m)',
        'dt_measured'       : 'DT Poço (µs/ft)',
        'dt_predicted'      : 'DT Predito (µs/ft)',
        'dt_well'           : 'DT Poço (µs/ft)',
        'dt_well_label'     : 'DT Poço',
        'dt_pred_label'     : 'DT Predito',
        'dt_axis'           : 'DT (μs/ft)',
        'residual'          : 'Resíduo (µs/ft)',
        'abs_error'         : 'Erro Absoluto (µs/ft)',
        'frequency'         : 'Frequência',
        'importance'        : 'Importância',
        'score'             : 'Pontuação',
        'fold'              : 'Fold',
        'sample_count'      : 'Contagem de Amostras',
        'data'              : 'Dados',
        'ideal_line'        : 'Linha 1:1 (ideal)',
        'ref_line'          : 'Referência 1:1',
        'bin_median'        : 'Mediana por bin',
        'training'          : 'Treinamento (26 poços)',
        # plot_prediction_vs_actual
        'pred_vs_actual_title' : 'Predito vs Medido',
        'ideal_line_label'     : 'Linha Ideal',
        # plot_residuals
        'residuals_title'      : 'Análise de Resíduos',
        'res_vs_pred'          : 'Resíduos vs Predito',
        'res_dist'             : 'Distribuição dos Resíduos',
        'mean_label'           : 'Média',
        'std_label'            : 'Desvio Padrão',
        # plot_cv_results
        'cv_title'             : 'Resultados da Validação Cruzada - {}',
        'mean_label_cv'        : 'Média: {:.4f}',
        'mean_std_text'        : 'Média ± DP = {:.4f} ± {:.4f}',
        # plot_feature_importance
        'feat_imp_title'       : 'Top {} Atributos Mais Importantes',
        # plot_comparison_boxplot
        'model_comp_title'     : 'Comparação de Modelos - {}',
        # plot_learning_curve
        'lc_title'             : 'Curva de Aprendizado',
        'lc_train'             : 'Treino',
        'lc_val'               : 'Validação',
        'lc_xlabel'            : 'Tamanho do Conjunto de Treino',
        # plot_well_profile_and_scatter / hexabin
        'plotting_wells'       : 'Plotando {} poço(s)...',
        'well_not_found'       : "⚠️  Poço '{}' não encontrado no DataFrame!",
        'well_saved'           : '[{}/{}] {} - Salvo: {}',
        'well_r2'              : '[{}/{}] {} - R²={:.4f}',
        'trend_label'          : 'Tendência: y={:.3f}x+{:.3f}',
        'trend_label2'         : 'Tendência (y={:.3f}x{:+.2f})',
        'mae_global'           : 'MAE global = {:.2f}',
        'mae_bin'              : 'MAE mediano (bin 200m)',
        # plot_petrophysical_diagnosis_panel
        'petro_space_title'    : 'Espaço Petrofísico RHOB × NPHI\n(cinza = treino | cor = erro absoluto)',
        'outside_training'     : 'Fora do P5-P95 de treino:\n  NPHI: {:.1f}%\n  RHOB: {:.1f}%',
        'dt_nphi_title'        : 'DT Poço × NPHI por Litologia\n(relação velocidade-porosidade)',
        'dt_dt_title'          : 'DT Poço × DT Predito\n(cor = NPHI; NPHI alto → maior porosidade)',
        'err_depth_title'      : 'Erro Absoluto × Profundidade\n(cor = RHOB; verde = compactado | vermelho = poroso)',
        'spearman_nphi_dt'     : 'Spearman ρ(NPHI×DT) = {:+.3f}  (p={:.3f})',
        'spearman_box'         : 'ρ(prof×erro) = {:+.3f}  (p={:.3f})\nρ(NPHI×erro) = {:+.3f}  (p={:.3f})\nρ(RHOB×erro) = {:+.3f}  (p={:.3f})',
        'cb_abs_error'         : 'Erro Absoluto (µs/ft)',
        'cb_rhob'              : 'RHOB (g/cm³)',
        'cb_nphi'              : 'NPHI (pu)',
        'nphi_axis'            : 'NPHI (pu)',
        'rhob_axis'            : 'RHOB (g/cm³)',
        'well_not_found_pred'  : "⚠️  Poço '{}' não encontrado em df_pred.",
        'saved'                : 'Salvo: {}',
    }
}

def _L(lang, key):
    """Helper: returns the label string for the given language and key."""
    return _LABELS.get(lang, _LABELS['en']).get(key, _LABELS['en'].get(key, key))


# Default lithology color palette for the project
LITHO_COLORS = {
    'shale'                  : '#228B22',
    'claystone'              : '#90EE90',
    'siltstone'              : '#8A3301',
    'sandstone'              : '#FFFF00',
    'conglomeratic_sandstone': '#FFD700',
    'conglomerate'           : '#FFA500',
    'breccia'                : '#FF8C00',
    'diamictite'             : '#F0E68C',
    'calcilutite'            : '#87CEEB',
    'limestone'              : '#ADD8E6',
    'dolomite'               : '#4682B4',
    'marl'                   : '#5F9EA0',
    'mudstone'               : '#808080',
    'igneous'                : '#FF0000',
}
FALLBACK_COLOR = '#BFC9CA'


def plot_well(well, columns=None, pred=None, figsize=(700, 800),
              config_file=None, lang='en', **kwargs):
    """
    Plots well log curves flexibly.

    Parameters:
    -----------
    well : pd.DataFrame
    columns : list, optional
    pred : array-like, optional
    figsize : tuple, optional
    config_file : str, optional
    lang : str, optional
        Language for axis labels. 'en' (default) or 'pt'.
    **kwargs : dict
    """
    if config_file is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'configs', 'config_curves.json')

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Configuration file {config_file} not found. "
            f"Make sure config_curves.json is located in sonic_ml_utils/configs/"
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error reading {config_file}: invalid JSON format. Details: {e}")

    for col, settings in kwargs.items():
        if col in config:
            config[col].update(settings)

    well = well.sort_values(by='DEPTH').copy()

    available_columns = [col for col in well.columns if col in config]
    if columns:
        columns_to_plot = [col for col in columns if col in available_columns]
    else:
        default_cols = ['GR', 'RT90', 'RHOB', 'NPHI', 'CALI', 'DT']
        columns_to_plot = [col for col in default_cols if col in available_columns]

    if pred is not None:
        well['DT_pred'] = np.squeeze(pred)

    if not columns_to_plot:
        print(f"No valid columns to plot for well {well.iloc[0]['Well_Name']}.")
        return None

    fig = make_subplots(
        rows=1,
        cols=len(columns_to_plot),
        subplot_titles=[config[col]['nome'] for col in columns_to_plot],
        shared_yaxes=True,
        horizontal_spacing=0.05
    )

    for i, col in enumerate(columns_to_plot):
        showlegend = (col == 'DT') and (pred is not None)
        fig.add_trace(
            go.Scatter(
                x=well[col], y=well['DEPTH'], mode='lines',
                line=dict(color=config[col]['cor'], width=config[col]['line_width']),
                name=config[col]['nome'], showlegend=showlegend
            ), row=1, col=i+1
        )
        if col == 'DT' and pred is not None:
            fig.add_trace(
                go.Scatter(
                    x=well['DT_pred'], y=well['DEPTH'], mode='lines',
                    line=dict(color=config['DT_pred']['cor'], width=config['DT_pred']['line_width']),
                    name=config['DT_pred']['nome'], showlegend=True
                ), row=1, col=i+1
            )

    fig.update_layout(
        height=figsize[1],
        width=figsize[0] * len(columns_to_plot) / 6,
        title_text=well.iloc[0]['Well_Name'],
        title_x=0.5,
        showlegend=pred is not None,
        font=dict(size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=10)

    for i, col in enumerate(columns_to_plot):
        fig.update_xaxes(
            title_text=config[col]['unidade'], row=1, col=i+1,
            type=config[col]['tipo_eixo'],
            range=config[col]['range'] if config[col]['range'] else None
        )
        fig.update_yaxes(
            title_text=_L(lang, 'depth') if i == 0 else None,
            showticklabels=(i == 0),
            autorange='reversed', row=1, col=i+1,
            side='right' if i > 0 else 'left', dtick=50
        )

    return fig


def plot_prediction_vs_actual(y_true, y_pred, title=None,
                               figsize=(10, 6), save_path=None, lang='en'):
    """
    Plots a scatter chart of predicted vs actual values.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    L = lambda key: _L(lang, key)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
            label=L('ideal_line_label'))

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(L('dt_measured'), fontsize=12)
    ax.set_ylabel(L('dt_predicted'), fontsize=12)
    ax.set_title(title if title else L('pred_vs_actual_title'),
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_residuals(y_true, y_pred, title=None,
                   figsize=(12, 5), save_path=None, lang='en'):
    """
    Plots residual analysis: scatter and histogram.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    L = lambda key: _L(lang, key)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel(L('dt_predicted'), fontsize=11)
    axes[0].set_ylabel(L('residual'), fontsize=11)
    axes[0].set_title(L('res_vs_pred'), fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel(L('residual'), fontsize=11)
    axes[1].set_ylabel(L('frequency'), fontsize=11)
    axes[1].set_title(L('res_dist'), fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

    mean_res = np.mean(residuals)
    std_res  = np.std(residuals)
    text = f"{L('mean_label')} = {mean_res:.4f}\n{L('std_label')} = {std_res:.4f}"
    axes[1].text(0.05, 0.95, text, transform=axes[1].transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title if title else L('residuals_title'),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_cv_results(cv_scores, model_name, metric='R²',
                    figsize=(10, 6), save_path=None, lang='en'):
    """
    Plots cross-validation results.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    L = lambda key: _L(lang, key)

    fig, ax = plt.subplots(figsize=figsize)
    folds = range(1, len(cv_scores) + 1)

    ax.plot(folds, cv_scores, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=np.mean(cv_scores), color='r', linestyle='--', linewidth=2,
               label=L('mean_label_cv').format(np.mean(cv_scores)))
    ax.fill_between(folds,
                    np.mean(cv_scores) - np.std(cv_scores),
                    np.mean(cv_scores) + np.std(cv_scores),
                    alpha=0.2, color='red')

    ax.set_xlabel(L('fold'), fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(L('cv_title').format(model_name), fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(folds)

    text = L('mean_std_text').format(np.mean(cv_scores), np.std(cv_scores))
    ax.text(0.05, 0.05, text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_feature_importance(importance_dict, top_n=15,
                            figsize=(10, 8), save_path=None, lang='en'):
    """
    Plots feature importance.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    L = lambda key: _L(lang, key)

    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features    = sorted_features[:top_n]
    features, importances = zip(*top_features)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel(L('importance'), fontsize=12)
    ax.set_title(L('feat_imp_title').format(top_n), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_comparison_boxplot(results_dict, metric='R²',
                            figsize=(10, 6), save_path=None, lang='en'):
    """
    Plots a boxplot comparing multiple models.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    L = lambda key: _L(lang, key)

    fig, ax = plt.subplots(figsize=figsize)
    data   = [scores for scores in results_dict.values()]
    labels = list(results_dict.keys())

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(L('model_comp_title').format(metric), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_learning_curve(train_sizes, train_scores, val_scores,
                        title=None, figsize=(10, 6), save_path=None, lang='en'):
    """
    Plots a learning curve.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    L = lambda key: _L(lang, key)

    fig, ax = plt.subplots(figsize=figsize)

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    ax.plot(train_sizes, train_mean, 'o-', linewidth=2,
            label=L('lc_train'), markersize=6)
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.2)

    ax.plot(train_sizes, val_mean, 'o-', linewidth=2,
            label=L('lc_val'), markersize=6)
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.2)

    ax.set_xlabel(L('lc_xlabel'), fontsize=12)
    ax.set_ylabel(L('score'), fontsize=12)
    ax.set_title(title if title else L('lc_title'),
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_well_profile_and_scatter(predictions_df, well_name=None,
                                   figsize=(10, 8), save_path=None, lang='en'):
    """
    Plots depth profile + scatter plot for one or more wells.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    from matplotlib.gridspec import GridSpec
    L = lambda key: _L(lang, key)

    if well_name:
        wells_to_plot = [well_name]
    else:
        wells_to_plot = sorted(predictions_df['Well_Name'].unique())

    print(L('plotting_wells').format(len(wells_to_plot)))

    for idx, w in enumerate(wells_to_plot, 1):
        well_data = predictions_df[predictions_df['Well_Name'] == w].copy()

        if well_data.empty:
            print(L('well_not_found').format(w))
            continue

        well_data = well_data.sort_values('DEPTH')
        depth   = well_data['DEPTH'].values
        dt_real = well_data['DT_real'].values
        dt_pred = well_data['DT_pred'].values

        r2   = r2_score(dt_real, dt_pred)
        rmse = np.sqrt(mean_squared_error(dt_real, dt_pred))
        mae  = mean_absolute_error(dt_real, dt_pred)
        bias = (dt_pred - dt_real).mean()

        fig = plt.figure(figsize=figsize)
        gs  = GridSpec(1, 2, width_ratios=[0.35, 1.0], figure=fig)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        fig.suptitle(f"{w}", fontsize=13, weight="bold")

        # Profile
        ax0.plot(dt_real, depth, color="black", label=L('dt_well_label'), linewidth=0.8)
        ax0.plot(dt_pred, depth, color="red",   label=L('dt_pred_label'),
                 linewidth=0.8, alpha=0.8)
        ax0.invert_yaxis()
        ax0.set_xlabel(L('dt_axis'), fontsize=11, fontweight='bold')
        ax0.set_ylabel(L('depth'),   fontsize=11, fontweight='bold')
        ax0.legend(fontsize=9, loc='best')
        ax0.grid(True, linestyle="--", alpha=0.4)
        ax0.set_xlim([40, 240])

        # Scatter
        ax1.scatter(dt_real, dt_pred, color="blue", alpha=0.6, s=10,
                    edgecolor=None, label=L('data'))

        lims = [min(dt_real.min(), dt_pred.min()) - 5,
                max(dt_real.max(), dt_pred.max()) + 5]

        ax1.plot(lims, lims, color="red", linestyle="-", linewidth=2,
                 label=L('ideal_line'), zorder=10)

        z = np.polyfit(dt_real, dt_pred, 1)
        p = np.poly1d(z)
        ax1.plot(lims, p(lims), color="orange", linestyle="--", linewidth=1.5,
                 label=L('trend_label').format(z[0], z[1]))

        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel(L('dt_well'),      fontsize=11, fontweight='bold')
        ax1.set_ylabel(L('dt_predicted'), fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.set_aspect('equal', adjustable='box')

        stats_text = (f'R² = {r2:.4f}\n'
                      f'RMSE = {rmse:.3f}\n'
                      f'MAE = {mae:.3f}\n'
                      f'N = {len(dt_real)}\n'
                      f'Bias = {bias:.3f}')
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            save_file = save_path.replace('.png', f'_{w}.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(L('well_saved').format(idx, len(wells_to_plot), w, save_file))
        else:
            print(L('well_r2').format(idx, len(wells_to_plot), w, r2))

        plt.show()
        plt.close(fig)


def plot_petrophysical_diagnosis_panel(
    well_name,
    df_pred,
    df_train=None,
    figsize=(16, 14),
    save_path=None,
    lithology_colors=None,
    lang='en',
):
    """
    Generates a 4-subplot petrophysical diagnosis panel for a problematic well.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    L = lambda key: _L(lang, key)

    ALPHA_TRAIN  = 0.15
    S_TRAIN      = 4
    S_WELL       = 18
    ALPHA_WELL   = 0.7
    GRID_ALPHA   = 0.3
    FONTSIZE_AX  = 11
    FONTSIZE_TTL = 12
    FONTSIZE_SUP = 13

    lc = lithology_colors if lithology_colors else LITHO_COLORS

    well_data = df_pred[df_pred['Well_Name'] == well_name].copy()
    if well_data.empty:
        print(L('well_not_found_pred').format(well_name))
        return None

    well_data = well_data.sort_values('DEPTH').reset_index(drop=True)
    well_data['abs_error'] = (well_data['DT_pred'] - well_data['DT_real']).abs()

    if df_train is not None:
        train_ctx = df_train.copy()
    else:
        train_ctx = df_pred[df_pred['Well_Name'] != well_name].copy()

    r2   = r2_score(well_data['DT_real'], well_data['DT_pred'])
    rmse = np.sqrt(mean_squared_error(well_data['DT_real'], well_data['DT_pred']))
    mae  = mean_absolute_error(well_data['DT_real'], well_data['DT_pred'])
    bias = (well_data['DT_pred'] - well_data['DT_real']).mean()

    rho_nphi, p_nphi = scipy_stats.spearmanr(well_data['NPHI'], well_data['abs_error'])
    rho_rhob, p_rhob = scipy_stats.spearmanr(well_data['RHOB'], well_data['abs_error'])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"{well_name}  |  R²={r2:.3f}  RMSE={rmse:.2f} µs/ft  "
        f"MAE={mae:.2f} µs/ft  Bias={bias:+.2f} µs/ft",
        fontsize=FONTSIZE_SUP, fontweight='bold'
    )

    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Subplot 1: RHOB × NPHI ───────────────────────────────────────────────
    ax1.scatter(train_ctx['NPHI'], train_ctx['RHOB'],
                c='gray', alpha=ALPHA_TRAIN, s=S_TRAIN,
                label=L('training'), rasterized=True)

    err_vals  = well_data['abs_error'].values
    vmax_err  = np.percentile(err_vals, 95)
    sc1 = ax1.scatter(well_data['NPHI'], well_data['RHOB'],
                      c=err_vals, cmap='YlOrRd', alpha=ALPHA_WELL, s=S_WELL,
                      vmin=0, vmax=vmax_err, zorder=5, label=well_name)
    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.85, pad=0.02)
    cb1.set_label(L('cb_abs_error'), fontsize=9)

    for q, ls, lw in [(5, '--', 0.8), (95, '--', 0.8)]:
        ax1.axvline(np.percentile(train_ctx['NPHI'], q),
                    color='steelblue', linestyle=ls, linewidth=lw, alpha=0.7)
        ax1.axhline(np.percentile(train_ctx['RHOB'], q),
                    color='steelblue', linestyle=ls, linewidth=lw, alpha=0.7)

    nphi_lo, nphi_hi = (np.percentile(train_ctx['NPHI'], 5),
                        np.percentile(train_ctx['NPHI'], 95))
    rhob_lo, rhob_hi = (np.percentile(train_ctx['RHOB'], 5),
                        np.percentile(train_ctx['RHOB'], 95))
    pct_out_nphi = ((well_data['NPHI'] < nphi_lo) |
                    (well_data['NPHI'] > nphi_hi)).mean() * 100
    pct_out_rhob = ((well_data['RHOB'] < rhob_lo) |
                    (well_data['RHOB'] > rhob_hi)).mean() * 100

    ax1.text(0.97, 0.97,
             L('outside_training').format(pct_out_nphi, pct_out_rhob),
             transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.invert_yaxis()
    ax1.set_xlabel(L('nphi_axis'), fontsize=FONTSIZE_AX)
    ax1.set_ylabel(L('rhob_axis'), fontsize=FONTSIZE_AX)
    ax1.set_title(L('petro_space_title'), fontsize=FONTSIZE_TTL)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(fontsize=8, loc='lower left')

    # ── Subplot 2: DT × NPHI by lithology ───────────────────────────────────
    liths_present = well_data['Lithology_norm'].unique()
    for lith in liths_present:
        mask  = well_data['Lithology_norm'] == lith
        color = lc.get(lith, 'gray')
        ax2.scatter(well_data.loc[mask, 'NPHI'], well_data.loc[mask, 'DT_real'],
                    c=color, alpha=ALPHA_WELL + 0.1, s=S_WELL,
                    label=f'{lith} (N={mask.sum():,})', edgecolors='none')

    nphi_bins = np.percentile(well_data['NPHI'], np.linspace(5, 95, 12))
    bin_idx   = np.digitize(well_data['NPHI'], nphi_bins)
    bin_med_nphi = [well_data.loc[bin_idx == i, 'NPHI'].median()
                    for i in range(1, len(nphi_bins))]
    bin_med_dt   = [well_data.loc[bin_idx == i, 'DT_real'].median()
                    for i in range(1, len(nphi_bins))]
    valid = [i for i, v in enumerate(bin_med_nphi)
             if not np.isnan(v) and not np.isnan(bin_med_dt[i])]
    if len(valid) >= 2:
        ax2.plot([bin_med_nphi[i] for i in valid],
                 [bin_med_dt[i] for i in valid],
                 'k--', linewidth=1.2, alpha=0.6,
                 label=L('bin_median'), zorder=10)

    rho_dt_nphi, p_dt_nphi = scipy_stats.spearmanr(
        well_data['NPHI'], well_data['DT_real'])
    ax2.text(0.97, 0.03,
         L('spearman_nphi_dt').format(rho_dt_nphi, p_dt_nphi),
         transform=ax2.transAxes, fontsize=8,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel(L('nphi_axis'),    fontsize=FONTSIZE_AX)
    ax2.set_ylabel(L('dt_measured'),  fontsize=FONTSIZE_AX)
    ax2.set_title(L('dt_nphi_title'), fontsize=FONTSIZE_TTL)
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(fontsize=7, loc='upper left',
               ncol=1 if len(liths_present) <= 4 else 2)

    # ── Subplot 3: DT measured × DT predicted ───────────────────────────────
    sc3 = ax3.scatter(well_data['DT_real'], well_data['DT_pred'],
                      c=well_data['NPHI'], cmap='Blues', alpha=ALPHA_WELL,
                      s=S_WELL, vmin=well_data['NPHI'].quantile(0.05),
                      vmax=well_data['NPHI'].quantile(0.95))
    cb3 = fig.colorbar(sc3, ax=ax3, shrink=0.85, pad=0.02)
    cb3.set_label(L('cb_nphi'), fontsize=9)

    lims3 = [min(well_data['DT_real'].min(), well_data['DT_pred'].min()) - 3,
             max(well_data['DT_real'].max(), well_data['DT_pred'].max()) + 3]

    ax3.plot(lims3, lims3, 'r-', linewidth=1.8, label=L('ideal_line'), zorder=10)

    z3 = np.polyfit(well_data['DT_real'], well_data['DT_pred'], 1)
    p3 = np.poly1d(z3)
    ax3.plot(lims3, p3(lims3), color='orange', linestyle='--', linewidth=1.4,
             label=L('trend_label2').format(z3[0], z3[1]))

    ax3.set_xlim(lims3)
    ax3.set_ylim(lims3)
    ax3.set_aspect('equal', adjustable='box')

    stats3 = (f'R² = {r2:.4f}\n'
              f'RMSE = {rmse:.3f}\n'
              f'MAE = {mae:.3f}\n'
              f'Bias = {bias:+.3f}')
    ax3.text(0.05, 0.95, stats3, transform=ax3.transAxes, fontsize=8,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax3.set_xlabel(L('dt_measured'),  fontsize=FONTSIZE_AX)
    ax3.set_ylabel(L('dt_predicted'), fontsize=FONTSIZE_AX)
    ax3.set_title(L('dt_dt_title'),   fontsize=FONTSIZE_TTL)
    ax3.grid(True, alpha=GRID_ALPHA)
    ax3.legend(fontsize=8, loc='lower right')

    # ── Subplot 4: Absolute error × Depth ───────────────────────────────────
    sc4 = ax4.scatter(well_data['abs_error'], well_data['DEPTH'],
                      c=well_data['RHOB'], cmap='RdYlGn', alpha=ALPHA_WELL,
                      s=S_WELL, vmin=well_data['RHOB'].quantile(0.05),
                      vmax=well_data['RHOB'].quantile(0.95))
    cb4 = fig.colorbar(sc4, ax=ax4, shrink=0.85, pad=0.02)
    cb4.set_label(L('cb_rhob'), fontsize=9)

    well_data['depth_bin'] = (well_data['DEPTH'] // 200 * 200).astype(int)
    bin_stats = well_data.groupby('depth_bin').agg(
        mae_bin=('abs_error', 'median'),
        depth_mid=('DEPTH', 'median')
    ).reset_index()
    ax4.plot(bin_stats['mae_bin'], bin_stats['depth_mid'],
             'k-o', linewidth=1.8, markersize=5, zorder=10,
             label=L('mae_bin'))

    ax4.axvline(mae, color='red', linestyle='--', linewidth=1.2,
                label=L('mae_global').format(mae))

    rho_d, p_d = scipy_stats.spearmanr(well_data['DEPTH'], well_data['abs_error'])
    ax4.text(0.97, 0.97,
             L('spearman_box').format(rho_d, p_d, rho_nphi, p_nphi, rho_rhob, p_rhob),
             transform=ax4.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax4.invert_yaxis()
    ax4.set_xlabel(L('abs_error'),      fontsize=FONTSIZE_AX)
    ax4.set_ylabel(L('depth'),          fontsize=FONTSIZE_AX)
    ax4.set_title(L('err_depth_title'), fontsize=FONTSIZE_TTL)
    ax4.grid(True, alpha=GRID_ALPHA)
    ax4.legend(fontsize=8, loc='lower right')

    fig.subplots_adjust(top=0.92, hspace=0.38, wspace=0.32,
                        left=0.07, right=0.97, bottom=0.06)

    if save_path:
        save_file = save_path.replace('.png', f'_{well_name}.png')
        fig.savefig(save_file, dpi=150, bbox_inches='tight')
        print(L('saved').format(save_file))

    plt.show()
    plt.close(fig)


def plot_well_profile_and_hexabin(predictions_df, well_name=None,
                                   figsize=(10, 8), save_path=None,
                                   cmap='viridis', gridsize=10, lang='en'):
    """
    Plots depth profile + hexabin plot for one or more wells.

    Parameters:
    -----------
    lang : str, optional
        'en' (default) or 'pt'.
    """
    from matplotlib.gridspec import GridSpec
    L = lambda key: _L(lang, key)

    if well_name:
        wells_to_plot = [well_name]
    else:
        wells_to_plot = sorted(predictions_df['Well_Name'].unique())

    print(L('plotting_wells').format(len(wells_to_plot)))

    for idx, w in enumerate(wells_to_plot, 1):
        well_data = predictions_df[predictions_df['Well_Name'] == w].copy()

        if well_data.empty:
            print(L('well_not_found').format(w))
            continue

        well_data = well_data.sort_values('DEPTH')
        depth   = well_data['DEPTH'].values
        dt_real = well_data['DT_real'].values
        dt_pred = well_data['DT_pred'].values

        r2   = r2_score(dt_real, dt_pred)
        rmse = np.sqrt(mean_squared_error(dt_real, dt_pred))
        mae  = mean_absolute_error(dt_real, dt_pred)
        bias = (dt_pred - dt_real).mean()

        fig = plt.figure(figsize=figsize)
        gs  = GridSpec(1, 2, width_ratios=[0.35, 1.0], figure=fig)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        fig.suptitle(f"{w}", fontsize=13, weight="bold")

        # Profile
        ax0.plot(dt_real, depth, color="black", label=L('dt_well_label'), linewidth=0.8)
        ax0.plot(dt_pred, depth, color="red",   label=L('dt_pred_label'),
                 linewidth=0.8, alpha=0.8)
        ax0.invert_yaxis()
        ax0.set_xlabel(L('dt_axis'), fontsize=11, fontweight='bold')
        ax0.set_ylabel(L('depth'),   fontsize=11, fontweight='bold')
        ax0.legend(fontsize=9, loc='best')
        ax0.grid(True, linestyle="--", alpha=0.4)
        ax0.set_xlim([40, 240])

        # Hexabin
        hb   = ax1.hexbin(dt_real, dt_pred, gridsize=gridsize, cmap=cmap,
                           mincnt=1, linewidths=0.2)
        cbar = plt.colorbar(hb, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label(L('sample_count'), fontsize=11)

        lims = [min(dt_real.min(), dt_pred.min()) - 5,
                max(dt_real.max(), dt_pred.max()) + 5]

        ax1.plot(lims, lims, color="red", linestyle="-", linewidth=2,
                 label=L('ref_line'), zorder=10)

        z = np.polyfit(dt_real, dt_pred, 1)
        p = np.poly1d(z)
        ax1.plot(lims, p(lims), color="orange", linestyle="--", linewidth=1.5,
                 label=L('trend_label').format(z[0], z[1]))

        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel(L('dt_well'),      fontsize=11, fontweight='bold')
        ax1.set_ylabel(L('dt_predicted'), fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.set_aspect('equal', adjustable='box')

        stats_text = (f'R² = {r2:.4f}\n'
                      f'RMSE = {rmse:.3f}\n'
                      f'MAE = {mae:.3f}\n'
                      f'N = {len(dt_real)}\n'
                      f'Bias = {bias:.3f}')
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            save_file = save_path.replace('.png', f'_{w}.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(L('well_saved').format(idx, len(wells_to_plot), w, save_file))
        else:
            print(L('well_r2').format(idx, len(wells_to_plot), w, r2))

    

# ==============================================================================
# FORMATION COLOR PALETTE
# ==============================================================================
FORMATION_COLORS = {
    'FM. CALUMBI'       : '#2980B9',
    'FM. COTINGUIBA'    : '#27AE60',
    'FM. RIACHUELO'     : '#E67E22',
    'FM. MURIBECA'      : '#8E44AD',
    'FM. COQUEIRO SECO' : '#E74C3C',
    'out_of_range'      : '#BDC3C7',
}


# ==============================================================================
# PLOT_WELL_PETROPHYSICAL_LOG
# ==============================================================================
def plot_well_petrophysical_log(
    well_name,
    df,
    df_form,
    depth_col='DEPTH',
    lith_col='Lithology_norm',
    error_col='abs_error',
    formation_colors=None,
    lithology_colors=None,
    save_path=None,
):
    """
    Six-panel diagnostic figure for a single well.

    Panels: Formation column | GR | Density (RHOB) | Neutron Porosity (NPHI) |
            Measured vs Predicted DT | Absolute Error by Lithology

    Parameters
    ----------
    well_name         : str
    df                : DataFrame — merged predictions + lithology + formation + R²
                        Required columns: Well_Name, DEPTH, DT_real, DT_pred,
                        GR, RHOB, NPHI, <lith_col>, <error_col>, R2
    df_form           : DataFrame — formation intervals per well
                        Required columns: Well_Name, depth_start, depth_end, formation
    depth_col         : str, default 'DEPTH'
    lith_col          : str, default 'Lithology_norm'
    error_col         : str, default 'abs_error'
    formation_colors  : dict or None — overrides default FORMATION_COLORS palette
    lithology_colors  : dict or None — overrides default LITHO_COLORS palette
    save_path         : str or Path or None — if provided, saves figure here
    """
    fc = formation_colors if formation_colors is not None else FORMATION_COLORS
    lc = lithology_colors if lithology_colors is not None else LITHO_COLORS

    sub = df[df['Well_Name'] == well_name].sort_values(depth_col).copy()
    if sub.empty:
        print(f"Well '{well_name}' not found in df.")
        return

    form_intervals = df_form[df_form['Well_Name'] == well_name].sort_values('depth_start')
    r2_well   = sub['R2'].iloc[0] if 'R2' in sub.columns else float('nan')
    depth_min = sub[depth_col].min()
    depth_max = sub[depth_col].max()

    fig, axes = plt.subplots(
        1, 6, figsize=(18, 14),
        sharey=True,
        gridspec_kw={'width_ratios': [0.4, 1, 1, 1, 1.5, 2.5], 'wspace': 0.06}
    )
    ax_form, ax_gr, ax_rhob, ax_nphi, ax_dt, ax_err = axes

    # ── Helper: formation background bands ────────────────────────────────────
    def _add_form_background(ax):
        for _, row in form_intervals.iterrows():
            top  = max(row['depth_start'], depth_min)
            base = min(row['depth_end'],   depth_max)
            if base <= top:
                continue
            color = fc.get(row['formation'], '#BDC3C7')
            ax.axhspan(top, base, color=color, alpha=0.07, zorder=0)
            ax.axhline(top, color=color, lw=0.6, linestyle='--', alpha=0.4)

    # ── Panel 1: Stratigraphic column ─────────────────────────────────────────
    for _, row in form_intervals.iterrows():
        top  = max(row['depth_start'], depth_min)
        base = min(row['depth_end'],   depth_max)
        if base <= top:
            continue
        color = fc.get(row['formation'], '#BDC3C7')
        ax_form.barh(y=(top + base) / 2, width=1, height=(base - top),
                     color=color, edgecolor='white', linewidth=0.3)
        if (base - top) > 50:
            ax_form.text(0.5, (top + base) / 2,
                         row['formation'].replace('FM. ', ''),
                         ha='center', va='center', fontsize=5.5,
                         rotation=90, color='white', fontweight='bold')

    ax_form.set_xlim(0, 1)
    ax_form.set_xticks([])
    ax_form.set_ylabel('Depth [m]', fontsize=11)
    ax_form.set_title('Fm.', fontsize=8)
    form_handles = [
        mpatches.Patch(color=fc.get(r['formation'], '#BDC3C7'),
                       label=r['formation'].replace('FM. ', ''))
        for _, r in form_intervals.iterrows()
        if r['depth_end'] >= depth_min and r['depth_start'] <= depth_max
    ]
    ax_form.legend(handles=form_handles, loc='lower left',
                   fontsize=5.5, framealpha=0.9,
                   title='Formation', title_fontsize=6)

    # ── Panel 2: GR ───────────────────────────────────────────────────────────
    if 'GR' in sub.columns:
        _add_form_background(ax_gr)
        ax_gr.plot(sub['GR'], sub[depth_col], color='#5D6D7E', lw=0.8, alpha=0.85)
        ax_gr.set_xlabel('GR\n[API]', fontsize=9)
        ax_gr.set_title('Gamma Ray', fontsize=9)
        ax_gr.set_xlim(0, 200)
        ax_gr.grid(alpha=0.25)
    else:
        ax_gr.text(0.5, 0.5, 'GR\nnot available',
                   ha='center', va='center', transform=ax_gr.transAxes)

    # ── Panel 3: RHOB ─────────────────────────────────────────────────────────
    if 'RHOB' in sub.columns:
        _add_form_background(ax_rhob)
        ax_rhob.plot(sub['RHOB'], sub[depth_col], color='#8B0000', lw=0.8, alpha=0.85)
        ax_rhob.set_xlabel('RHOB\n[g/cm³]', fontsize=9)
        ax_rhob.set_title('Density', fontsize=9)
        ax_rhob.set_xlim(1.5, 3.0)
        ax_rhob.grid(alpha=0.25)
    else:
        ax_rhob.text(0.5, 0.5, 'RHOB\nnot available',
                     ha='center', va='center', transform=ax_rhob.transAxes)

    # ── Panel 4: NPHI ─────────────────────────────────────────────────────────
    if 'NPHI' in sub.columns:
        _add_form_background(ax_nphi)
        ax_nphi.plot(sub['NPHI'], sub[depth_col], color='#2471A3', lw=0.8, alpha=0.85)
        ax_nphi.set_xlabel('NPHI\n[pu]', fontsize=9)
        ax_nphi.set_title('Neutron Porosity', fontsize=9)
        ax_nphi.set_xlim(80, 0)   # reversed axis (standard petrophysical convention)
        ax_nphi.grid(alpha=0.25)
    else:
        ax_nphi.text(0.5, 0.5, 'NPHI\nnot available',
                     ha='center', va='center', transform=ax_nphi.transAxes)

    # ── Panel 5: Measured vs Predicted DT ─────────────────────────────────────
    if 'DT_real' in sub.columns or 'DT_pred' in sub.columns:
        _add_form_background(ax_dt)
        if 'DT_real' in sub.columns:
            ax_dt.plot(sub['DT_real'], sub[depth_col],
                       color='black', lw=0.9, alpha=0.85, label='Measured DT')
        if 'DT_pred' in sub.columns:
            ax_dt.plot(sub['DT_pred'], sub[depth_col],
                       color='crimson', lw=0.9, alpha=0.75,
                       linestyle='--', label='Predicted DT')
        ax_dt.set_xlabel('DT\n[µs/ft]', fontsize=9)
        ax_dt.set_title('Measured vs Predicted DT', fontsize=9)
        ax_dt.legend(fontsize=7, loc='lower right', framealpha=0.85)
        ax_dt.grid(alpha=0.25)
    else:
        ax_dt.text(0.5, 0.5, 'DT\nnot available',
                   ha='center', va='center', transform=ax_dt.transAxes)

    # ── Panel 6: Absolute Error by Lithology ──────────────────────────────────
    _add_form_background(ax_err)
    if lith_col in sub.columns:
        for lith in sub[lith_col].dropna().unique():
            s = sub[sub[lith_col] == lith]
            color = lc.get(str(lith).lower().strip(), FALLBACK_COLOR)
            ax_err.scatter(s[error_col], s[depth_col],
                           color=color, alpha=0.4, s=6,
                           label=str(lith).replace('_', ' ').capitalize(),
                           zorder=2)
    else:
        ax_err.scatter(sub[error_col], sub[depth_col],
                       color=FALLBACK_COLOR, alpha=0.4, s=6, zorder=2)

    sub = sub.copy()
    sub['_depth_bin'] = (sub[depth_col] // 100 * 100).astype(int)
    trend = (sub.groupby('_depth_bin')[error_col]
             .mean().reset_index()
             .rename(columns={error_col: '_mae_mean'}))
    ax_err.plot(trend['_mae_mean'], trend['_depth_bin'],
                color='black', lw=2, linestyle='--',
                label='Mean per bin (100m)', zorder=5)

    ax_err.set_ylim(depth_max + 20, depth_min - 20)
    ax_err.set_xlabel('Absolute Error [µs/ft]', fontsize=10)
    ax_err.set_title('Error by Lithology', fontsize=9)
    ax_err.legend(fontsize=7, loc='lower right', framealpha=0.85)
    ax_err.grid(alpha=0.25)

    r2_str = f'{r2_well:.3f}' if not np.isnan(r2_well) else 'N/A'
    fig.suptitle(
        f'{well_name}\n'
        f'Petrophysical Logs + Measured vs Predicted DT + Error  (R²={r2_str})',
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    plt.show()
    plt.close(fig)
