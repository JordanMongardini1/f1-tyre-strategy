import pytest
import pandas as pd

from f1strategy.data import (
    setup_cache,
    load_race,
    get_all_races,
    COLUMNS,
    IDENTIFIERS,
    MODEL,
)

SLICKS = {"SOFT", "MEDIUM", "HARD"}

# Chosen for how they differ, not for coverage: high-deg desert,
# stint-capped low-deg, street circuit.
FAST_RACES = [(2025, 4), (2025, 23), (2025, 8)]


# ---------------------------------------------------------------- fast suite

@pytest.fixture(scope="module", params=FAST_RACES, ids=lambda p: f"{p[0]}r{p[1]}")
def race(request):
    setup_cache()
    return load_race(*request.param)


@pytest.fixture(scope="module")
def clean(race):
    return race[0]


@pytest.fixture(scope="module")
def raw(race):
    return race[1]


def test_columns_and_order(clean):
    assert list(clean.columns) == COLUMNS


def test_identifier_dtypes(clean):
    assert pd.api.types.is_integer_dtype(clean["Year"])
    assert pd.api.types.is_integer_dtype(clean["RoundNumber"])


def test_no_column_all_nan(clean):
    assert not clean[IDENTIFIERS + MODEL].isna().all().any()


def test_compounds_are_slicks(clean):
    assert clean["Compound"].isin(SLICKS).all()


def test_filter_columns_are_constant(clean):
    assert clean["TrackStatus"].eq("1").all()
    assert clean["IsAccurate"].eq(True).all()
    assert clean[["PitInTime_s", "PitOutTime_s"]].isna().all().all()


def test_stint_id_is_unique_key(clean):
    grouped = clean.groupby("StintId")[["Driver", "Stint"]].nunique()
    assert (grouped == 1).all().all()


def test_retention_is_reasonable(clean, raw):
    retention = len(clean) / len(raw)
    assert 0.6 < retention < 0.95


# ------------------------------------------------- Bahrain only: degradation

@pytest.fixture(scope="module")
def bahrain():
    setup_cache()
    clean, _ = load_race(2025, 4)
    return clean


def test_degradation_is_visible(bahrain):
    stint = bahrain[(bahrain["Driver"] == "VER") & (bahrain["Stint"] == 1)]
    stint = stint.sort_values("TyreLife")
    assert len(stint) >= 6          # so the head and tail windows don't overlap
    assert stint["LapTime_s"].iloc[-3:].mean() > stint["LapTime_s"].iloc[:3].mean()


# ------------------------------------- slow: whole season, pooled (-m slow)

@pytest.fixture(scope="module")
def season():
    setup_cache()
    return get_all_races(2025)


@pytest.mark.slow
def test_season_no_failures(season):
    *_, failed = season
    assert failed == [], f"{len(failed)} races failed, first: {failed[:2]}"


@pytest.mark.slow
def test_season_wet_skips_are_plausible(season):
    *_, skipped_wet, _ = season
    assert len(skipped_wet) <= 6, f"skipped {len(skipped_wet)} as wet: {skipped_wet}"


@pytest.mark.slow
def test_season_schema(season):
    clean, *_ = season
    assert list(clean.columns) == COLUMNS
    assert clean["Compound"].isin(SLICKS).all()
    assert not clean[IDENTIFIERS + MODEL].isna().all().any()


@pytest.mark.slow
def test_season_stint_ids_do_not_collide(season):
    clean, *_ = season
    keys = ["Year", "RoundNumber", "Driver", "Stint"]
    grouped = clean.groupby("StintId")[keys].nunique()
    assert (grouped == 1).all().all()


@pytest.mark.slow
def test_season_one_location_per_race(season):
    clean, *_ = season
    assert clean["Location"].nunique() == clean["RoundNumber"].nunique()


@pytest.mark.slow
def test_season_size(season):
    clean, *_ = season
    assert 10_000 < len(clean) < 25_000