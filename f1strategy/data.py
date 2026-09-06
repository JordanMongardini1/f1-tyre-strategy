# FastF1 loading + lap cleaning   (Checkpoint 1)
from pathlib import Path
import fastf1
import pandas
from pandas import DataFrame

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fastf1_cache"

#Turn on the cache for FastF1 laps
def setup_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

#Function to get the session, laps, and event information for a given year and race name
def get_session(year: int, name: str):
    #Attempt to load the race session for the given year and name
    try:
        session = fastf1.get_session(year, name, 'Race')
    except Exception as e:
        print(f"Error loading session for {year} {name}: {e}")
        return None, None

    #Get the laps and event information from the session
    event = session.event
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    
    return session, event

def get_data(session, event):
    laps = DataFrame(session.laps)

    driver = laps["Driver"]
    lap_number = laps["LapNumber"]
    lap_time = laps["LapTime"].dt.total_seconds()
    tyre_life = laps["TyreLife"]
    compound = laps["Compound"]
    stint = laps["Stint"]
    track_status = laps["TrackStatus"]
    pit_in_time = laps["PitInTime"].dt.total_seconds()
    pit_out_time = laps["PitOutTime"].dt.total_seconds()
    is_accurate = laps["IsAccurate"]

    data = [driver, lap_number, lap_time, tyre_life, compound, stint, track_status, pit_in_time, pit_out_time, is_accurate]

    data = pandas.concat(data, axis=1)

    data['Country'] = event["Country"]
    data['RoundNumber'] = event["RoundNumber"]
    data['EventName'] = event["EventName"]

    return data

#Function to filter the data to include only laps without pit stops, accurate laps, and laps with green flag track status
def filter_data(data: DataFrame):
    filtered = data[(data['PitInTime'].isna()) & (data['PitOutTime'].isna()) & (data['IsAccurate'] == True) & (data['TrackStatus'] == '1')]
    return filtered