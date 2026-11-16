import pandas
from pandas import DataFrame

def pace_delta(clean_data: DataFrame) -> DataFrame:
    #Group by stint ID, take the lap with the lowest TyreLife as the stint refrence and subtract its LapTime_s from all laps in the stint
    out = clean_data.reset_index(drop=True).copy()

    # One row per stint: the lap with the lowest surviving TyreLife.
    ref_idx = out.groupby("StintId")["TyreLife"].idxmin()
    reference = out.loc[ref_idx].set_index("StintId")

    out["PaceDelta_s"] = out["LapTime_s"] - out["StintId"].map(reference["LapTime_s"])
    out["RefTyreLife"] = out["StintId"].map(reference["TyreLife"])

    return out