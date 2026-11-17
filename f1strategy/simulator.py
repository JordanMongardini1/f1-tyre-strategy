from dataclasses import dataclass
import pandas as pd

PIT_LOSS_CONSTANT = 20  #Model assumption, represents the time lost during a pit stop in seconds, future improvements could include variable pit loss based on track and conditions

#Represents a single stint in a race, defined by the tyre compound and the number of laps
@dataclass
class Stint:
    compound: str
    laps: int

    def __str__(self):
        return f"{self.compound} stint for {self.laps} laps"

@dataclass
class Strategy:
    stints: list[Stint]

    def __str__(self):
        return " -> ".join(str(stint) for stint in self.stints)

#Simulates a race strategy and returns the total predicted race time in seconds.
def simulate_strategy(strategy: Strategy, model, circuit: str, driver: str, race_length: int) -> float:
    laps = []
    compounds = []

    for stint in strategy.stints:
        laps.append(stint.laps)
        compounds.append(stint.compound.upper())
        if stint.laps <= 0:
            raise ValueError("Each stint must have a positive number of laps.")

    #Check that the total number of laps in the strategy matches the race length
    if sum(laps) != race_length:
        raise ValueError("The total number of laps in the strategy does not match the race length.")

    #Check that the strategy is valid in terms of tyre compounds
    if len(set(compounds)) < 2:
        raise ValueError("The strategy must use at least two different tyre compounds.")

    #build dataframe for the strategy
    input_table = pd.DataFrame({
        "Driver": [driver] * race_length,   
        "EventName": [circuit] * race_length,
        "LapNumber": [lap for lap in range(1, race_length + 1)],
        "TyreLife": [lap for stint in strategy.stints for lap in range(1, stint.laps + 1)],
        "Compound": [stint.compound.upper() for stint in strategy.stints for _ in range(stint.laps)]
    })

    #Simulate the strategy using the provided model
    result = model.predict(input_table)

    #Sum the predicted lap times and add pit stop losses
    total_time = result.sum() + (len(strategy.stints) - 1) * PIT_LOSS_CONSTANT
    return float(total_time)

#Helper function to go from raw race data to a dictionary of race lengths
#The keys are the event names and the values are the maximum lap numbers for each event
def get_race_lengths(raw_data: pd.DataFrame) -> dict[str, int]:
    race_lengths = raw_data.groupby("EventName")["LapNumber"].max()

    # Convert event names to strings and lap numbers to integers.
    return {str(event): int(laps) for event, laps in race_lengths.items()}
