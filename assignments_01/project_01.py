# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 1

# Part 2: Mini-Project: World Happiness Pipeline


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from prefect import task, flow
from prefect.logging import get_run_logger

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@task
def process_year_from_file(filepath: Path) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Reading data from {filepath}")
    df = pd.read_csv(filepath, sep=";")
    year = int(filepath.name.split("_")[-1].split(".")[0])
    df["Year"] = year
    logger.info(f"Results for year {year} completed.")
    return df


@task(retries=3, retry_delay_seconds=2)
def process_all_years() -> pd.DataFrame:
    logger = get_run_logger()
    all_years = pd.DataFrame()

    for filepath in DATA_DIR.glob("*.csv"):
        year_df = process_year_from_file(filepath)
        logger.info(f"Adding year rows to DataFrame.")
        all_years = pd.concat([all_years, year_df])

    print(f"All Rows: {all_years}")
    logger.info(f"All rows added to DataFrame.")
    all_years.to_csv(BASE_DIR / "outputs/merged_happiness.csv")
    logger.info(f"Merged DataFrame saved to outputs/merged_happiness.csv")
    return all_years


if __name__ == "__main__":
    processed_all_years = process_all_years()
