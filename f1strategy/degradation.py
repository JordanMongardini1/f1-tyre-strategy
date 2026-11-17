import pandas
from pandas import DataFrame
import random
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

FORMULA = "LapTime_s ~ C(Compound) + TyreLife : Compound + I(TyreLife ** 2) : Compound + TyreLife : C(EventName) + LapNumber + C(EventName) + C(Driver)"
FORMULA_WITHOUT_LAP = "LapTime_s ~ C(Compound) + TyreLife : Compound + I(TyreLife ** 2) : Compound + TyreLife : C(EventName) + C(EventName) + C(Driver)"

def pace_delta(clean_data: DataFrame) -> DataFrame:
    #Group by stint ID, take the lap with the lowest TyreLife as the stint refrence and subtract its LapTime_s from all laps in the stint
    out = clean_data.reset_index(drop=True).copy()

    # One row per stint: the lap with the lowest surviving TyreLife.
    ref_idx = out.groupby("StintId")["TyreLife"].idxmin()
    reference = out.loc[ref_idx].set_index("StintId")

    out["PaceDelta_s"] = out["LapTime_s"] - out["StintId"].map(reference["LapTime_s"])
    out["RefTyreLife"] = out["StintId"].map(reference["TyreLife"])

    return out

def training_testing_split(clean_data: DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[DataFrame, DataFrame]:
    #group by stintID and shuffle each group by random_seed and then split whole groups into training and testing sets
    #Set seed for reproducibility
    random.seed(random_state)
    grouped = clean_data.groupby("StintId")
    stint_ids = list(grouped.groups.keys())
    random.shuffle(stint_ids)

    split = int(len(stint_ids) * (1 - test_size))
    train_stints = stint_ids[:split]
    test_stints = stint_ids[split:]

    train_data = clean_data[clean_data["StintId"].isin(train_stints)].reset_index(drop=True)
    test_data = clean_data[clean_data["StintId"].isin(test_stints)].reset_index(drop=True)

    return train_data, test_data

#Train a regression model using the specified formula
def fit(training_data: DataFrame, formula: str) -> RegressionResultsWrapper:
    model = smf.ols(formula=formula, data=training_data).fit()
    return model