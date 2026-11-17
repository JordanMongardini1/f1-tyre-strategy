from f1strategy.simulator import Strategy, Stint, PIT_LOSS_CONSTANT
import pandas as pd

#Minimum stint length in laps, (Assumption loww bound based on typical F1 stint lengths)
MINIMUM_STINT_LENGTH = 5
#Soft can be from 1 to 18, medium from 1 to 32, hard from 1 to 43 (Assumption upper bound based on actual F1 tyre data)
MAXIMUM_COMPOUND_LIFE = {
    "SOFT": 18,
    "MEDIUM": 32,
    "HARD": 43
}

def lookup_table(driver: str, circuit: str, race_length: int, model) -> dict:
    #Create all combinations of lap number, compound and tyre age
    lap_numbers = list(range(1, race_length + 1))
    compounds = ["SOFT", "MEDIUM", "HARD"]


    all_combinations = []

    for compound in compounds:
        for lap in lap_numbers:
            for age in range(1, MAXIMUM_COMPOUND_LIFE[compound] + 1):
                all_combinations.append((compound, lap, age))

    #Create DataFrame with all combinations and additional driver and event information
    all_predicted_times = pd.DataFrame(
        all_combinations, columns=["Compound", "LapNumber", "TyreLife"]
    )
    all_predicted_times["Driver"] = driver
    all_predicted_times["EventName"] = circuit

    #We now use the trained model to predict lap times for all combinations
    if model is not None:
        predictions = model.predict(all_predicted_times)
    else:
        raise ValueError("Model cannot be None")

    #Convert into a dictionary for O(1) average lookup
    lookup_dict = dict(zip(all_combinations, predictions))

    return lookup_dict

# Function to score a given strategy based on the lookup table, this function expects strategies only from the search that are already valid
def score_strategy(strategy: Strategy, lookup: dict) -> float:
    total_time = 0.0
    starting_stint_lap = 1

    for stint in strategy.stints:
        compound = stint.compound
        starting_lap = starting_stint_lap
        ending_lap = starting_lap + stint.laps - 1

        for lap in range(starting_lap, ending_lap + 1):
            tyre_age = lap - starting_lap + 1
            total_time += lookup[(compound, lap, tyre_age)]

        starting_stint_lap = ending_lap + 1

    #Add pit stop time between stints
    total_time += PIT_LOSS_CONSTANT * (len(strategy.stints) - 1)

    return total_time

def best_strategy(lookup: dict, race_length: int) -> tuple[list[tuple[float, Strategy]], int]:
    num_strategies_checked = 0
    best_strategies = []

    #All one stop strategies (two stints)
    for i in range(1, race_length):
        for compound1 in ["SOFT", "MEDIUM", "HARD"]:
            for compound2 in ["SOFT", "MEDIUM", "HARD"]:
                # Skip invalid strategies based on stint length and compound life constraints
                if (compound1 == compound2 
                or i < MINIMUM_STINT_LENGTH 
                or (race_length - i) < MINIMUM_STINT_LENGTH
                or i > MAXIMUM_COMPOUND_LIFE[compound1]
                or (race_length - i) > MAXIMUM_COMPOUND_LIFE[compound2]):
                    continue

                strategy = Strategy(stints=[
                    Stint(compound1, i),
                    Stint(compound2, race_length - i)
                ])

                score = score_strategy(strategy, lookup)
                best_strategies.append((score, strategy))

                num_strategies_checked += 1

    #All two stop strategies (three stints)
    for i in range(1, race_length):
        for j in range(1, race_length - i):
            for compound1 in ["SOFT", "MEDIUM", "HARD"]:
                for compound2 in ["SOFT", "MEDIUM", "HARD"]:
                    for compound3 in ["SOFT", "MEDIUM", "HARD"]:
                        if (compound1 == compound2 == compound3
                        or i < MINIMUM_STINT_LENGTH or j < MINIMUM_STINT_LENGTH or (race_length - i - j) < MINIMUM_STINT_LENGTH
                        or i > MAXIMUM_COMPOUND_LIFE[compound1]
                        or j > MAXIMUM_COMPOUND_LIFE[compound2]
                        or (race_length - i - j) > MAXIMUM_COMPOUND_LIFE[compound3]):
                            continue

                        strategy = Strategy(stints=[
                            Stint(compound1, i),
                            Stint(compound2, j),
                            Stint(compound3, race_length - i - j)
                        ])

                        score = score_strategy(strategy, lookup)
                        best_strategies.append((score, strategy))

                        num_strategies_checked += 1

    #Sort the best strategies by score in ascending order (lower time is better)
    best_strategies.sort(key=lambda x: x[0])

    return best_strategies, num_strategies_checked

#Function that returns the actual strategies drivers used in a race
def actual_strategies(raw_data: pd.DataFrame, race: str) -> tuple[dict[str, Strategy], int]:
    strategies = {}
    skipped_drivers = 0

    #get only the data for the specified race
    race_data = raw_data[raw_data["EventName"] == race]

    #get race length based on the highest LapNumber in race_data
    race_length = race_data["LapNumber"].max()

    for driver in race_data["Driver"].unique():
        driver_data = race_data[race_data["Driver"] == driver]

        #Check if the driver completed the full race length
        if driver_data["LapNumber"].max() < race_length:
            skipped_drivers += 1
            continue

        #Check that all compounds are valid
        if not set(driver_data["Compound"]).issubset({"SOFT", "MEDIUM", "HARD"}):
            skipped_drivers += 1
            continue

        driver_data = driver_data.groupby("Stint")
        stints = []

        for stint, stint_data in driver_data:
            compound = stint_data["Compound"].iloc[0]
            laps = len(stint_data)
            stints.append(Stint(compound, laps))

        strategies[driver] = Strategy(stints)

    return strategies, skipped_drivers
