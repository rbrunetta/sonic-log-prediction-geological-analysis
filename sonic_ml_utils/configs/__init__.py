"""
sonic_ml_utils library configurations

This module contains configuration files used by the library functions.
"""

import os
import json

# Path to the curve configuration file
CURVES_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_curves.json')


def load_curves_config():
    """
    Loads curve configuration from the JSON file.

    Returns:
    --------
    dict : Dictionary with curve configurations
    """
    with open(CURVES_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_curve_config(curve_name):
    """
    Gets the configuration for a specific curve.

    Parameters:
    -----------
    curve_name : str
        Curve name (e.g., 'GR', 'RHOB')

    Returns:
    --------
    dict : Curve configuration or None if not found
    """
    config = load_curves_config()
    return config.get(curve_name)


def add_custom_curve(curve_name, config_dict, save=False):
    """
    Adds a custom curve to the configurations.

    Parameters:
    -----------
    curve_name : str
        Name of the new curve
    config_dict : dict
        Dictionary with curve properties:
        - nome: Display name
        - unidade: Curve unit
        - range: [min, max] or None
        - cor: Line color
        - tipo_eixo: 'linear' or 'log'
        - line_width: Line width
    save : bool
        If True, permanently saves to the JSON file

    Example:
    --------
    >>> add_custom_curve('TEMP', {
    ...     'nome': 'Temperature',
    ...     'unidade': '°C',
    ...     'range': [0, 200],
    ...     'cor': 'orange',
    ...     'tipo_eixo': 'linear',
    ...     'line_width': 1.3
    ... }, save=True)
    """
    config = load_curves_config()
    config[curve_name] = config_dict
    
    if save:
        with open(CURVES_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Curve '{curve_name}' permanently added to the configuration file.")
    
    return config


__all__ = [
    'CURVES_CONFIG_PATH',
    'load_curves_config',
    'get_curve_config',
    'add_custom_curve'
]
