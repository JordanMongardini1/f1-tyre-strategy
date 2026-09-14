# FastF1 loading + lap cleaning   (Checkpoint 1)
from pathlib import Path
import fastf1
import pandas
from pandas import DataFrame

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "fastf1_cache"

#Column definitions for the cleaned race data
IDENTIFIERS = ["Year", "RoundNumber", "EventName", "Location", 
               "Driver", "Team", "Stint", "StintId", "LapNumber"]
MODEL = ["LapTime_s", "TyreLife", "Compound"]
FILTER = ["TrackStatus", "PitInTime_s", "PitOutTime_s", "IsAccurate"]

#Combined list of all columns for the cleaned race data
COLUMNS = IDENTIFIERS + MODEL + FILTER

def save_season(clean, raw, year):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(DATA_DIR / f"clean_laps_{year}.parquet", index=False)
    raw.to_parquet(DATA_DIR / f"raw_laps_{year}.parquet", index=False)

def load_season(year):
    return (
        pandas.read_parquet(DATA_DIR / f"clean_laps_{year}.parquet"),
        pandas.read_parquet(DATA_DIR / f"raw_laps_{year}.parquet"),
    )

#Turn on the cache for FastF1 laps
def setup_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level('WARNING')

#Function to get the session, laps, and event information for a given year and race name
def get_session(year: int, name: str|int):
    #Attempt to load the race session for the given year and name
    try:
        session = fastf1.get_session(year, name, 'Race')

        #Get the laps and event information from the session
        event = session.event
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        
    except Exception as e:
        #throw an error with the year and round and the error message
        raise RuntimeError(f"Error loading session for {year} {name}: {e}")
    
    return session, event

def get_data(session, event):
    laps = DataFrame(session.laps)

    cols = ["Driver", "LapNumber", "TyreLife", "Compound",
        "Stint", "TrackStatus", "IsAccurate", "Team"]

    data = laps[cols].copy()

    data["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    data["PitInTime_s"] = laps["PitInTime"].dt.total_seconds()
    data["PitOutTime_s"] = laps["PitOutTime"].dt.total_seconds()

    data['Location'] = event["Location"]
    data['RoundNumber'] = event["RoundNumber"]
    data['EventName'] = event["EventName"]

    data["Year"] = event["EventDate"].year

    data["StintId"] = (
        data["Year"].astype(str)
        + "_" + data["RoundNumber"].astype(str)
        + "_" + data["Driver"]
        + "_" + data["Stint"].astype("Int64").astype(str)
    )

    return data[COLUMNS]

#Function to filter the data to include only laps without pit stops, accurate laps, and laps with green flag track status
def filter_data(data: DataFrame):
    mask = (
        data['PitInTime_s'].isna()
        & data['PitOutTime_s'].isna()
        & data['IsAccurate']
        & data['TrackStatus'].eq('1')
        & data["TyreLife"].notna() 
        & data["Stint"].notna()
    )

    return data[mask].copy()

#Load race data for a given year and round number and return both the clean and raw data
def load_race(year: int, rnd: int):
    session, event = get_session(year, rnd)
    raw_df = get_data(session, event)

    #filter
    clean_df = filter_data(raw_df)

    #return both raw and clean data
    return clean_df, raw_df

def build_round_list(year: int):
    schedule = fastf1.get_event_schedule(year)

    #Exclude testing events
    schedule = schedule[schedule['EventFormat'] != 'testing']

    #make sure that event date is before today
    schedule = schedule[schedule['EventDate'] < pandas.Timestamp.today()]

    return list(schedule['RoundNumber'])

#Get all races for a given year, including clean and raw data, skipped wet races, and failed races
def get_all_races(year: int) -> tuple[DataFrame, DataFrame, list[int], list[tuple[int, str]]]:
    rounds = build_round_list(year)

    wet_compounds = {"INTERMEDIATE", "WET"}

    clean_data_list = []
    raw_data_list = []
    skipped_wet = []
    failed = []

    for rnd in rounds:
        try:
            clean_df, raw_df = load_race(year, rnd)

            print(f"Loaded race data for year {year}, round {rnd} with {len(clean_df)} clean laps and {len(raw_df)} raw laps")

            if raw_df['Compound'].isin(wet_compounds).any():
                skipped_wet.append(rnd)
                print(f"Skipped wet race for year {year}, round {rnd}")
            else:
                clean_data_list.append(clean_df)
                raw_data_list.append(raw_df)
                print(f"Appended clean and raw data for year {year}, round {rnd}")
        except Exception as e:
            failed.append((rnd, repr(e)))
            print(f"Failed to load race data for year {year}, round {rnd}: {repr(e)}")

    #Check if clean data list and raw data list are not empty before concatenating and throw an error if they are
    if not clean_data_list or not raw_data_list:
        raise ValueError(
            f"No valid race data for {year}. "
            f"{len(failed)} failed, {len(skipped_wet)} skipped wet. "
            f"First failure: {failed[0] if failed else 'none'}"
        )

    #concat and reset index with drop=True
    clean_data = pandas.concat(clean_data_list, ignore_index=True)
    raw_data = pandas.concat(raw_data_list, ignore_index=True)

    return clean_data, raw_data, skipped_wet, failed
    