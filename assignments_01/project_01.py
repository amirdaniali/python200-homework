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
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@task
def retrieve_year_from_file(filepath: Path) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Reading data from {filepath}")
    df = pd.read_csv(filepath, sep=";")
    year = int(filepath.name.split("_")[-1].split(".")[0])
    df["Year"] = year
    logger.info(f"Results for year {year} completed.")
    return df


@task(retries=3, retry_delay_seconds=2)
def retrieve_all_years() -> pd.DataFrame:
    """Load data from all ten yearly CSV files into a single DataFrame. Your implementation should not duplicate code for each year -- iterate over a list of file paths and load them in a loop.
    You discovered some quirks when you inspected the raw files. Make sure you account for those when calling pd.read_csv(). There is also something missing from each file that you will need to add before merging: each row needs to know which year it came from. Think about where to add that information.

    After loading and merging, save the combined dataset to:

    assignments_01/outputs/merged_happiness.csv

    Add retries=3, retry_delay_seconds=2 to this task's decorator. File I/O is exactly the kind of operation that can fail intermittently in production pipelines, and this is where retries earn their keep.
    """

    logger = get_run_logger()
    all_years = pd.DataFrame()

    logger.info(f"Checking to see if the data has previously been processed.")
    if (BASE_DIR / "outputs/merged_happiness.csv").exists():
        logger.info(f"Data has already been processed.")
        logger.info(f"Skipping to next step.")
        return pd.read_csv(BASE_DIR / "outputs/merged_happiness.csv")

    for filepath in DATA_DIR.glob("*.csv"):
        year_df = retrieve_year_from_file(filepath)
        logger.info(f"Adding year rows to DataFrame.")
        all_years = pd.concat([all_years, year_df])

    print(f"All Rows: {all_years}")
    logger.info(f"All rows added to DataFrame.")
    all_years.to_csv(BASE_DIR / "outputs/merged_happiness.csv")
    logger.info(f"Merged DataFrame saved to outputs/merged_happiness.csv")
    return all_years


@task
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """processes the data to be used in later stages"""

    # if Happiness score is missing for 2024 and ladder score exists:
    df["Happiness Score"] = df.get("Happiness score", pd.NA)  # in case column missing
    df.loc[df["Year"] == 2024, "Happiness score"] = df.loc[
        df["Year"] == 2024, "Ladder score"
    ]

    df["Happiness score"] = (
        df["Happiness score"].astype(str).str.replace(",", ".", regex=False)
    )
    df["Happiness score"] = pd.to_numeric(df["Happiness score"], errors="coerce")

    # todo: add more process
    # df = df.copy()
    # df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    # df = df.dropna(subset=["Score", "Class"])
    # logger.info(" Data cleaned")

    return df


@task
def describe_data(df):
    """Compute and log overall descriptive statistics for happiness_score: mean, median, and standard deviation.

    Then compute and log the mean happiness score grouped by year and by region. Looking at the regional breakdown is often the most interesting part of this dataset -- you may already have a hypothesis about which regions rank highest before you run the numbers.
    """

    logger = get_run_logger()
    logger.info(f"Computing descriptive statistics.")
    print(f"Mean: {df['Happiness score'].mean()}")
    print(f"Median: {df['Happiness score'].median()}")
    print(f"Standard Deviation: {df['Happiness score'].std()}")
    print(
        f"Mean by Region: {df.groupby('Regional indicator')['Happiness score'].mean()}"
    )
    logger.info(f"Happiness score computed.")
    print(df.groupby("Year")["Happiness score"].describe())


@task()
def happiness_histogram(df: pd.DataFrame) -> None:
    """Create and save the following visualizations to assignments_01/outputs/:

    A histogram of all happiness scores across all years. Save as happiness_histogram.png.
    """
    logger = get_run_logger()
    sns.histplot(
        data=df,
        x="Happiness score",
        hue="Year",
        bins=30,
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.title("Happiness score distribution by year")
    plt.tight_layout()
    plt.show()
    plt.savefig(BASE_DIR / "outputs/happiness_histogram.png")
    logger.info('Plot saved to "happiness_histogram.png"')
    plt.close()


@task()
def happiness_boxplot(df: pd.DataFrame) -> None:
    """Create and save the following visualizations to assignments_01/outputs/:
    A boxplot comparing happiness score distributions across years (one box per year). Save as happiness_by_year.png
    """
    logger = get_run_logger()
    ax = df.boxplot(by="Year", column="Happiness score", return_type="axes")
    plt.title("Happiness score distribution across years")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "outputs/happiness_by_year.png")
    logger.info('Plot saved to "happiness_by_year.png"')
    plt.show()
    plt.close()


if __name__ == "__main__":
    raw_data = retrieve_all_years()
    processed_data = process_data(raw_data)
    describe_data(processed_data)
    happiness_histogram(processed_data)
    happiness_boxplot(processed_data)
