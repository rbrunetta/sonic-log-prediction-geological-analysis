from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sonic_ml_utils",
    version="1.0.0",
    author="Rodrigo Brunetta",
    author_email="rbrunetta.colab@gmail.com",
    description="Utility library for sonic log prediction with Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rbrunetta/sonic_log_prediction",
    packages=find_packages(exclude=["bkp*", "tests*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
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
        "plotly>=5.5.0",
        "adjustText>=0.8",
    ],
)
