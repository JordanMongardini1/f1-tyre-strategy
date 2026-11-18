F1 Tyre Strategy
Jordan Mongardini, Intro to AI Final Project

This project fits a linear regression (OLS) model of F1 lap times from the 2025
season, uses it to simulate race strategies, and runs an exhaustive search over
one and two stop strategies to compare against what teams actually did.

Requirements
Python 3.12 or newer. All packages are listed in pyproject.toml
(fastf1, pandas, numpy, scikit-learn, statsmodels, matplotlib, pyarrow, jupyter).

Setup
1. Unzip and open a terminal in the project folder
2. python -m venv .venv
3. Activate it
4. pip install -e ".[dev]"
5. jupyter notebook

Running
Open the notebooks folder and run each notebook in order.

01_data_exploration.ipynb
    Downloads the 2025 season from FastF1, cleans it, and saves it to data/.
    Needs internet and takes a long time the first time. You can skip this
    notebook because the saved data files are already included in data/.
02_degradation_model.ipynb
    Train/test split, fits the OLS model, error tables, and degradation figures.
03_counterfactuals.ipynb
    Strategy simulator, exhaustive search, and 17 race comparison.

Folders
f1strategy/   Python code (data loading, model, simulator, strategy search)
notebooks/    The three notebooks above
data/         Cleaned and raw 2025 lap data (parquet)
figures/      Figures saved by the notebooks