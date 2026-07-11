"""
Script to create the full directory structure for the sonic_log_prediction project

Autor: [Seu Nome]
Data: Dezembro 2025

Como usar:
    python setup_project_structure.py

This will create the full folder and file structure.
"""

import os
from pathlib import Path


def create_directory_structure():
    """Creates the full project directory structure."""

    # Folder structure
    directories = [
        'data/raw',
        'data/processed',
        'sonic_ml_utils',
        'notebooks',
        'results/figures/xgboost',
        'results/figures/lgbm',
        'results/figures/random_forest',
        'results/figures/comparison',
        'results/metrics',
        'results/models',
        'results/predictions',
        'scripts',
        'tests',
        'docs',
    ]
    
    print("=" * 60)
    print("CREATING PROJECT STRUCTURE: sonic_log_prediction")
    print("=" * 60)
    
    # Create directories
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    print("\n" + "=" * 60)
    print("FOLDER STRUCTURE CREATED SUCCESSFULLY!")
    print("=" * 60)


def create_readme_files():
    """Creates README.md files in key directories."""

    print("\n" + "=" * 60)
    print("CREATING README.md FILES")
    print("=" * 60)
    
    # Main README
    readme_main = """# Sonic Log Prediction using Machine Learning

## 📋 Description

This project implements Machine Learning algorithms to predict sonic logs (DT - Delta Time)
from other well logs, using the Leave-One-Well-Out (LOWO) methodology for validation.

## 🎯 Objectives

- Predict sonic logs in wells without direct measurements
- Compare different ML algorithms (XGBoost, LightGBM, Random Forest, etc.)
- Validate models using Leave-One-Well-Out cross-validation
- Provide reusable tools for well log analysis

## 🗂️ Project Structure

```
sonic_log_prediction/
├── data/                   # Project data
├── sonic_ml_utils/         # Custom utility library
├── notebooks/              # Numbered Jupyter notebooks
├── results/                # Results (figures, metrics, models)
├── scripts/                # Standalone Python scripts
└── docs/                   # Additional documentation
```

## 🚀 How to Use

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/seu-usuario/sonic_log_prediction.git
cd sonic_log_prediction

# Create virtual environment
conda env create -f environment.yml
# OR
pip install -r requirements.txt

# Install the custom library
pip install -e .
```

### 2. Run the Notebooks

Follow the numerical order:

1. `01_exploratory_analysis.ipynb` - Exploratory analysis
2. `02_lazypredict_screening.ipynb` - Algorithm screening
3. `03_xgboost_lowo.ipynb` - XGBoost with LOWO
4. `04_lgbm_lowo.ipynb` - LightGBM with LOWO
5. `05_random_forest_lowo.ipynb` - Random Forest with LOWO
6. `06_model_comparison.ipynb` - Final comparison

### 3. Or Use the Scripts

```bash
python scripts/train_xgboost.py --data data/processed/wells.csv
```

## 📊 Results

The main results are in `results/`:
- `figures/` - Charts and visualizations
- `metrics/` - Performance metrics in CSV
- `models/` - Saved trained models
- `predictions/` - Saved predictions

## 🛠️ Technologies

- Python 3.8+
- XGBoost, LightGBM, Random Forest
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{sonic_log_prediction,
  author = {Seu Nome},
  title = {Sonic Log Prediction using Machine Learning},
  year = {2025},
  url = {https://github.com/seu-usuario/sonic_log_prediction}
}
```

## 📄 License

MIT License - see the LICENSE file for details

## 👤 Author

**[Seu Nome]**
- Instituição: [Sua Universidade]
- Email: [seu.email@exemplo.com]
- LinkedIn: [seu-perfil]

## 🔗 Related Project

- [Seismic-Well Integration](link) - Integration of predicted logs with seismic data

---

**Project Status:** 🚧 In Development / ✅ Complete
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_main)
    print("✓ Created: README.md")

    # data/ README
    readme_data = """# Data Directory

## 📂 Structure

- `raw/` - Original data, never modified
- `processed/` - Processed data ready for use

## 📊 Datasets

### raw/
- Place the original raw data here
- **Never modify these files!**

### processed/
- `wells.csv` - Data without IQR filter applied
- `wells_iqr.csv` - Data with IQR filter applied

## 🔄 Processing

To generate processed data from raw, see:
- `notebooks/01_exploratory_analysis.ipynb`

## 📝 Data Description

[Describe your features here:]

- **DEPT**: Depth (m)
- **GR**: Gamma Ray (API)
- **RHOB**: Bulk Density (g/cm³)
- **NPHI**: Neutron Porosity (fraction)
- **DT**: Sonic Log - TARGET (µs/ft)
- ... [add other features]

## ⚠️ Notes

- IQR-filtered data have gaps due to outlier removal
- For details on preprocessing, refer to the documentation
"""
    
    with open('data/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_data)
    print("✓ Created: data/README.md")

    # notebooks/ README
    readme_notebooks = """# Notebooks

Run the notebooks in the following order:

## 📓 Execution Order

### 1️⃣ `01_exploratory_analysis.ipynb`
**Goal:** Exploratory analysis of well log data
- Descriptive statistics
- Distribution visualizations
- Correlation analysis
- Outlier identification

### 2️⃣ `02_lazypredict_screening.ipynb`
**Goal:** Initial screening with LazyPredict
- Quick test of multiple algorithms
- Identification of the most promising ones
- Performance baseline

### 3️⃣ `03_xgboost_lowo.ipynb`
**Goal:** XGBoost training with Leave-One-Well-Out
- Hyperparameter optimization
- LOWO validation
- Feature importance analysis
- Model saving

### 4️⃣ `04_lgbm_lowo.ipynb`
**Goal:** LightGBM training with Leave-One-Well-Out
- LightGBM-specific configuration
- LOWO validation
- Comparison with XGBoost

### 5️⃣ `05_random_forest_lowo.ipynb`
**Goal:** Random Forest training with LOWO
- RF configuration
- LOWO validation
- Importance analysis

### 6️⃣ `06_model_comparison.ipynb`
**Goal:** Statistical comparison of all models
- Statistical tests
- Comparative visualizations
- Best model selection
- Final report generation

## 📦 Requirements

Before running, install the dependencies:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

## 💡 Tips

- Run each notebook from start to finish sequentially
- Notebooks save results automatically to `results/`
- Trained models are saved to `results/models/`
- Estimated total time: ~2-4 hours (depends on hardware)

## 🆘 Common Issues

**Error: ModuleNotFoundError: No module named 'sonic_ml_utils'**
- Solution: Run `pip install -e .` at the project root

**Error: FileNotFoundError: data/processed/wells.csv**
- Solution: Run `01_exploratory_analysis.ipynb` first
"""
    
    with open('notebooks/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_notebooks)
    print("✓ Created: notebooks/README.md")

    print("\n" + "=" * 60)
    print("README.md FILES CREATED SUCCESSFULLY!")
    print("=" * 60)


def create_gitignore():
    """Creates the .gitignore file."""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data (uncomment if needed)
# data/raw/*.csv
# data/processed/*.csv

# Models (uncomment if models are too large)
# results/models/*.pkl
# results/models/*.h5
# results/models/*.joblib

# Large result files
results/predictions/*.csv
results/figures/*.png
results/figures/*.jpg
results/figures/*.pdf

# Temporary files
*.tmp
*.bak
*.log
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("✓ Created: .gitignore")


def create_requirements():
    """Creates the requirements.txt file."""
    
    requirements_content = """# Core
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.0.0

# Utilities
tqdm>=4.62.0
joblib>=1.1.0

# Optional: LazyPredict for screening
lazypredict>=0.2.12

# Development
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    print("✓ Created: requirements.txt")


def create_environment_yml():
    """Creates the environment.yml file for conda."""
    
    env_content = """name: sonic_prediction
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scipy>=1.7.0
  - scikit-learn>=1.0.0
  - xgboost>=1.5.0
  - lightgbm>=3.3.0
  - matplotlib>=3.4.0
  - seaborn>=0.11.0
  - jupyter>=1.0.0
  - ipykernel>=6.0.0
  - tqdm>=4.62.0
  - joblib>=1.1.0
  - pip
  - pip:
    - lazypredict>=0.2.12
"""
    
    with open('environment.yml', 'w', encoding='utf-8') as f:
        f.write(env_content)
    print("✓ Created: environment.yml")


def create_license():
    """Creates the LICENSE file (MIT)."""
    
    license_content = """MIT License

Copyright (c) 2025 [Seu Nome]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open('LICENSE', 'w', encoding='utf-8') as f:
        f.write(license_content)
    print("✓ Created: LICENSE")


def create_setup_py():
    """Creates the setup.py file for library installation."""
    
    setup_content = """from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sonic_ml_utils",
    version="1.0.0",
    author="[Seu Nome]",
    author_email="[seu.email@exemplo.com]",
    description="Utility library for sonic log prediction with ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/sonic_log_prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
)
"""
    
    with open('setup.py', 'w', encoding='utf-8') as f:
        f.write(setup_content)
    print("✓ Created: setup.py")


def create_citation_cff():
    """Creates the CITATION.cff file for academic citation."""
    
    citation_content = """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "[Seu Sobrenome]"
    given-names: "[Seu Nome]"
    orcid: "https://orcid.org/0000-0000-0000-0000"
title: "Sonic Log Prediction using Machine Learning"
version: 1.0.0
date-released: 2025-12-06
url: "https://github.com/seu-usuario/sonic_log_prediction"
preferred-citation:
  type: software
  authors:
    - family-names: "[Seu Sobrenome]"
      given-names: "[Seu Nome]"
  title: "Sonic Log Prediction using Machine Learning"
  year: 2025
"""
    
    with open('CITATION.cff', 'w', encoding='utf-8') as f:
        f.write(citation_content)
    print("✓ Created: CITATION.cff")


def create_placeholder_files():
    """Creates empty placeholder files."""
    
    placeholder_files = [
        'sonic_ml_utils/__init__.py',
        'tests/__init__.py',
        'scripts/.gitkeep',
        'docs/.gitkeep',
    ]
    
    for file_path in placeholder_files:
        Path(file_path).touch()
        print(f"✓ Created: {file_path}")


def print_next_steps():
    """Prints next steps."""

    print("\n" + "=" * 60)
    print("🎉 STRUCTURE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📋 NEXT STEPS:\n")
    print("1. Copy your library files into sonic_ml_utils/:")
    print("   - plotting.py")
    print("   - statistics.py")
    print("   - preprocessing.py (if you have one)")
    print()
    print("2. Place your data in:")
    print("   - data/raw/       → original data")
    print("   - data/processed/ → processed data")
    print()
    print("3. Organize your notebooks in notebooks/:")
    print("   - Rename following the numbering convention (01_, 02_, etc)")
    print()
    print("4. Install the library in development mode:")
    print("   pip install -e .")
    print()
    print("5. Test that everything works:")
    print("   python -c 'from sonic_ml_utils import plot_prediction_vs_actual'")
    print()
    print("=" * 60)
    print("📚 CHECK the README.md in each folder for more details!")
    print("=" * 60)


def main():
    """Main function."""
    
    print("\n")
    print("🚀 " * 20)
    print("\n   SETUP: sonic_log_prediction")
    print("\n🚀 " * 20)
    print("\n")
    
    try:
        create_directory_structure()
        create_readme_files()
        create_gitignore()
        create_requirements()
        create_environment_yml()
        create_license()
        create_setup_py()
        create_citation_cff()
        create_placeholder_files()
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        print("Please report this error.")
        raise


if __name__ == "__main__":
    main()
