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
from scipy.stats import ttest_ind, pearsonr

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
    """Task 1: Load Multiple Years of Data

    Load data from all ten yearly CSV files into a single DataFrame. Your
    implementation should not duplicate code for each year -- iterate over a list
    of file paths and load them in a loop.

    You discovered some quirks when you inspected the raw files. Make sure you
    account for those when calling pd.read_csv(). There is also something missing
    from each file that you will need to add before merging: each row needs to
    know which year it came from. Think about where to add that information.

    After loading and merging, save the combined dataset to:
    assignments_01/outputs/merged_happiness.csv

    Add retries=3, retry_delay_seconds=2 to this task's decorator. File I/O is
    exactly the kind of operation that can fail intermittently in production
    pipelines, and this is where retries earn their keep.
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
    """Processes the data to be used in later stages:
    - Fix column naming for happiness score.
    - For 2024, fill happiness score from ladder score if needed.
    - Convert comma decimals to floats for all numeric columns.
    """

    logger = get_run_logger()
    logger.info("Starting data processing.")

    # if happiness score is missing for 2024 and ladder score exists:
    if "Ladder score" in df.columns:
        logger.info("Filling 2024 happiness scores from ladder score where available.")
        mask_2024 = df["Year"] == 2024
        df.loc[mask_2024, "Happiness score"] = df.loc[mask_2024, "Ladder score"]

    # numeric-like columns using comma decimals in raw CSV
    numeric_like_cols = [
        "Happiness score",
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption",
    ]

    for col in numeric_like_cols:
        if col in df.columns:
            logger.info(f"Cleaning numeric column '{col}' (comma decimals -> floats).")
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ensure Year is a plain integer type for plotting/grouping
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)

    logger.info("Data processing complete.")
    return df


@task
def describe_data(df):
    """Compute and log overall descriptive statistics for happiness_score:
    mean, median, and standard deviation.

    Then compute and log the mean happiness score grouped by year and by region.
    Looking at the regional breakdown is often the most interesting part of this
    dataset -- you may already have a hypothesis about which regions rank highest
    before you run the numbers.
    """

    logger = get_run_logger()
    logger.info("Computing descriptive statistics for Happiness score.")

    happiness = df["Happiness score"].dropna()
    mean = happiness.mean()
    median = happiness.median()
    std = happiness.std()

    logger.info(f"Overall mean happiness score: {mean:.3f}")
    logger.info(f"Overall median happiness score: {median:.3f}")
    logger.info(f"Overall standard deviation of happiness score: {std:.3f}")

    if "Regional indicator" in df.columns:
        mean_by_region = (
            df.groupby("Regional indicator")["Happiness score"]
            .mean()
            .sort_values(ascending=False)
        )
        logger.info("Mean happiness score by region:")
        logger.info(mean_by_region.to_string())

    mean_by_year = df.groupby("Year")["Happiness score"].mean()
    logger.info("Mean happiness score by year:")
    logger.info(mean_by_year.to_string())


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
    plt.close()


@task()
def gdp_vs_happiness(df: pd.DataFrame) -> None:
    """Create and save a scatter plot showing the relationship between GDP per capita     and happiness score. Save as gdp_vs_happiness.png."""

    logger = get_run_logger()
    logger.info("Creating GDP vs Happiness scatter plot.")

    plot_df = df.copy()
    plot_df["Year"] = plot_df["Year"].astype(
        int
    )  # we use hue=year but we need to convert to int
    plt.figure()
    sns.scatterplot(
        data=plot_df,
        x="GDP per capita",
        y="Happiness score",
        hue="Year",
        alpha=0.7,
    )
    plt.title("GDP per capita vs Happiness score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness score")
    plt.tight_layout()

    plt.savefig(BASE_DIR / "outputs/gdp_vs_happiness.png")
    logger.info(f"Plot saved")
    plt.close()


@task()
def correlation_heatmap(df: pd.DataFrame) -> None:
    """Create and save a correlation heatmap (using sns.heatmap() with annot=True)
    showing the Pearson correlations between all numeric columns.
    Save as correlation_heatmap.png.

    Log a message after the plot is saved so you can see the progress in the
    Prefect dashboard.

    """
    logger = get_run_logger()
    logger.info("Creating correlation heatmap.")

    # select numeric columns only
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    output_path = BASE_DIR / "outputs" / "correlation_heatmap.png"
    plt.savefig(output_path)
    logger.info(f'Plot saved to "{output_path.name}"')
    plt.close()


@task()
def hypothesis_tests(df: pd.DataFrame) -> dict:
    """The pandemic began in early 2020. Did it affect global happiness scores?
    Test this directly: run an independent samples t-test comparing happiness
    scores from 2019 to 2020.

    Log the t-statistic, p-value, the mean happiness for each group, and a
    plain-language interpretation of the result at alpha = 0.05. Your
    interpretation should say something meaningful -- not just "we reject the
    null hypothesis" but what that actually means in terms of this data.

    Add a second test of your choice (for example, comparing two specific regions
    that you expect to differ based on the descriptive statistics you computed
    earlier).

    Returns a dict summarizing key results for use in the summary report.
    """

    logger = get_run_logger()
    logger.info("Running hypothesis tests.")

    results = {}

    # t-test 2019 vs 2020
    happiness_2019 = df.loc[df["Year"] == 2019, "Happiness score"].dropna()
    happiness_2020 = df.loc[df["Year"] == 2020, "Happiness score"].dropna()

    if len(happiness_2019) > 1 and len(happiness_2020) > 1:
        t_stat, p_value = ttest_ind(
            happiness_2019, happiness_2020, equal_var=False
        )  # Welch's t-test [12]
        mean_2019 = happiness_2019.mean()
        mean_2020 = happiness_2020.mean()

        logger.info(
            f"T-test 2019 vs 2020: t-statistic={t_stat:.3f}, p-value={p_value:.3f}, "
            f"mean_2019={mean_2019:.3f}, mean_2020={mean_2020:.3f}"
        )

        alpha = 0.05
        if p_value < alpha:
            interpretation = (
                "At alpha = 0.05, we find a statistically significant difference in "
                "mean happiness between 2019 and 2020, suggesting the pandemic period "
                "was associated with a change in global happiness scores."
            )
        else:
            interpretation = (
                "At alpha = 0.05, we do not find a statistically significant difference "
                "in mean happiness between 2019 and 2020, so any change in scores could "
                "be due to random variation in this dataset."
            )

        logger.info(f"Interpretation (2019 vs 2020): {interpretation}")

        results["t_2019_2020"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_2019": float(mean_2019),
            "mean_2020": float(mean_2020),
            "interpretation": interpretation,
        }
    else:
        logger.info("Insufficient data for 2019 or 2020 to run t-test.")

    # second test: compare two specific regions (e.g., Western Europe vs Sub-Saharan Africa)
    if "Regional indicator" in df.columns:
        regions = df["Regional indicator"].unique()
        # choose two commonly present regions if available
        region_a = None
        region_b = None
        for r in regions:
            if "Western Europe" in str(r):
                region_a = r
            if "Sub-Saharan Africa" in str(r):
                region_b = r

        # fallback: first two regions
        if region_a is None or region_b is None:
            unique_regions = list(regions)
            if len(unique_regions) >= 2:
                region_a = unique_regions[0]
                region_b = unique_regions[1]

        if region_a is not None and region_b is not None:
            logger.info(f"Running t-test between regions: {region_a} vs {region_b}")
            happiness_a = df.loc[
                df["Regional indicator"] == region_a, "Happiness score"
            ].dropna()
            happiness_b = df.loc[
                df["Regional indicator"] == region_b, "Happiness score"
            ].dropna()

            if len(happiness_a) > 1 and len(happiness_b) > 1:
                t_reg, p_reg = ttest_ind(happiness_a, happiness_b, equal_var=False)
                mean_a = happiness_a.mean()
                mean_b = happiness_b.mean()

                logger.info(
                    f"T-test {region_a} vs {region_b}: t-statistic={t_reg:.3f}, "
                    f"p-value={p_reg:.3f}, mean_{region_a}={mean_a:.3f}, "
                    f"mean_{region_b}={mean_b:.3f}"
                )

                alpha = 0.05
                if p_reg < alpha:
                    interpretation_reg = (
                        f"At alpha = 0.05, {region_a} and {region_b} have significantly "
                        "different mean happiness scores, supporting the idea that "
                        "regional context is strongly associated with happiness."
                    )
                else:
                    interpretation_reg = (
                        f"At alpha = 0.05, we do not find a significant difference in "
                        f"mean happiness between {region_a} and {region_b}; any apparent "
                        "difference could be due to chance."
                    )

                logger.info(f"Interpretation (region comparison): {interpretation_reg}")

                results["t_regions"] = {
                    "region_a": region_a,
                    "region_b": region_b,
                    "t_stat": float(t_reg),
                    "p_value": float(p_reg),
                    "mean_a": float(mean_a),
                    "mean_b": float(mean_b),
                    "interpretation": interpretation_reg,
                }
            else:
                logger.info(
                    "Insufficient data for selected regions to run second t-test."
                )
        else:
            logger.info("Could not identify two regions for second t-test.")

    return results


@task()
def correlation_and_multiple_comparisons(df: pd.DataFrame) -> dict:
    """For each numeric explanatory variable, compute the Pearson correlation with
    happiness score using scipy.stats.pearsonr and log the coefficient and
    p-value.

    Count how many correlation tests you performed, then compute:
    adjusted_alpha = 0.05 / number_of_tests

    Log which correlations are significant at the original alpha = 0.05, and
    which remain significant after applying the Bonferroni correction. You may
    find that some results that looked significant at first don't hold up under
    the stricter threshold -- that's a useful finding in itself.

    Return a summary dict of correlations and significance for use in the final
    report.
    """

    logger = get_run_logger()
    logger.info(
        "Computing Pearson correlations with Happiness score and applying Bonferroni correction."
    )

    numeric_df = df.select_dtypes(include=["number"])
    target = "Happiness score"
    if target not in numeric_df.columns:
        logger.info(
            "Happiness score column not found in numeric data; skipping correlations."
        )
        return {}

    correlations = {}
    p_values = {}
    significant_original = []
    significant_bonferroni = []

    # run correlation for each explanatory numeric variable (excluding target itself)
    for col in numeric_df.columns:
        if col == target:
            continue
        x = df[col].dropna()
        y = df[target].dropna()

        # align indices to ensure pairs match
        aligned = df[[col, target]].dropna()
        if aligned.shape[0] < 2:
            continue

        r, p = pearsonr(aligned[col], aligned[target])
        correlations[col] = r
        p_values[col] = p

        logger.info(
            f"Pearson correlation {col} vs Happiness score: r={r:.3f}, p-value={p:.3f}"
        )

    num_tests = len(correlations)
    if num_tests == 0:
        logger.info("No numeric explanatory variables found for correlation tests.")
        return {}

    alpha = 0.05
    adjusted_alpha = alpha / num_tests
    logger.info(
        f"Original alpha: {alpha:.3f}, Bonferroni-adjusted alpha: {adjusted_alpha:.5f}"
    )

    for col, p in p_values.items():
        if p < alpha:
            significant_original.append(col)
        if p < adjusted_alpha:
            significant_bonferroni.append(col)

    logger.info(f"Variables significant at alpha=0.05: {significant_original}")
    logger.info(
        f"Variables remaining significant after Bonferroni correction: {significant_bonferroni}"
    )

    # find variable most strongly correlated with happiness score after Bonferroni
    strongest_var = None
    strongest_r = None
    for col in significant_bonferroni:
        r = correlations[col]
        if strongest_r is None or abs(r) > abs(strongest_r):
            strongest_r = r
            strongest_var = col

    if strongest_var is not None:
        logger.info(
            f"Most strongly correlated variable after Bonferroni: {strongest_var} "
            f"with r={strongest_r:.3f}"
        )

    return {
        "correlations": correlations,
        "p_values": p_values,
        "alpha": alpha,
        "adjusted_alpha": adjusted_alpha,
        "significant_original": significant_original,
        "significant_bonferroni": significant_bonferroni,
        "strongest_variable": strongest_var,
        "strongest_r": strongest_r,
    }


@task()
def summary_report(df: pd.DataFrame, t_results: dict, corr_results: dict) -> None:
    """Log a human-readable summary of the key findings from the entire pipeline.
    Think of it as the "report" step -- the thing you'd share with a
    non-technical colleague. It should include:

    - Total number of countries and years in the merged dataset.
    - The top 3 and bottom 3 regions by mean happiness score.
    - The result of the pre/post-2020 t-test in plain language.
    - The variable most strongly correlated with happiness score (after
      Bonferroni correction).

    Log each of these as a separate logger.info() message so they're easy to
    find in the Prefect dashboard.
    """

    logger = get_run_logger()
    logger.info("Generating summary report.")

    # Total number of countries and years
    num_countries = df["Country"].nunique() if "Country" in df.columns else None
    num_years = df["Year"].nunique() if "Year" in df.columns else None

    logger.info(
        f"Summary: total number of countries in merged dataset = {num_countries}"
    )
    logger.info(f"Summary: total number of years in merged dataset = {num_years}")

    # Top 3 and bottom 3 regions by mean happiness score
    if "Regional indicator" in df.columns:
        region_means = (
            df.groupby("Regional indicator")["Happiness score"]
            .mean()
            .sort_values(ascending=False)
        )

        top_3 = region_means.head(3)
        bottom_3 = region_means.tail(3)

        logger.info("Summary: top 3 regions by mean happiness score:")
        logger.info(top_3.to_string())

        logger.info("Summary: bottom 3 regions by mean happiness score:")
        logger.info(bottom_3.to_string())
    else:
        logger.info(
            "Summary: 'Regional indicator' column not found; cannot compute region means."
        )

    # Pre/post-2020 t-test in plain language
    if "t_2019_2020" in t_results:
        t_info = t_results["t_2019_2020"]
        logger.info(
            "Summary: pre/post-2020 t-test result: "
            f"mean_2019 = {t_info['mean_2019']:.3f}, "
            f"mean_2020 = {t_info['mean_2020']:.3f}, "
            f"t-statistic = {t_info['t_stat']:.3f}, "
            f"p-value = {t_info['p_value']:.3f}"
        )
        logger.info(
            f"Summary: interpretation (2019 vs 2020): {t_info['interpretation']}"
        )
    else:
        logger.info("Summary: no t-test results for 2019 vs 2020 were available.")

    # Strongest correlation after Bonferroni correction
    strongest_var = corr_results.get("strongest_variable")
    strongest_r = corr_results.get("strongest_r")
    adjusted_alpha = corr_results.get("adjusted_alpha")
    if (
        strongest_var is not None
        and strongest_r is not None
        and adjusted_alpha is not None
    ):
        logger.info(
            "Summary: most strongly correlated variable with happiness score "
            f"after Bonferroni correction is '{strongest_var}' "
            f"with Pearson r = {strongest_r:.3f} at adjusted alpha = {adjusted_alpha:.5f}."
        )
    else:
        logger.info(
            "Summary: no variable remained significant after Bonferroni correction, "
            "or correlation results were unavailable."
        )


@flow
def happiness_pipeline() -> None:
    """Orchestrates the full World Happiness analysis pipeline. This flow should:

    - Load and merge multiple years of data (Task 1).
    - Process and clean the dataset for analysis.
    - Compute descriptive statistics (Task 2).
    - Generate visualizations (Task 3).
    - Run hypothesis tests (Task 4).
    - Compute correlations and apply Bonferroni correction (Task 5).
    - Log a final summary report of key findings (Task 6).

    The full pipeline should be runnable with:

        python project_01.py

    When you run it, it should execute all tasks in order, produce all outputs,
    and save them to the specified locations. It should be safe to run multiple
    times, overwriting previous outputs cleanly."""

    logger = get_run_logger()
    logger.info("Starting World Happiness pipeline.")

    # Task 1 & processing: load and clean data
    raw_data = retrieve_all_years()
    processed_data = process_data(raw_data)

    # Task 2: descriptive statistics
    describe_data(processed_data)

    # Task 3: visual exploration
    happiness_histogram(processed_data)
    happiness_boxplot(processed_data)
    gdp_vs_happiness(processed_data)
    correlation_heatmap(processed_data)

    # Task 4: hypothesis testing
    t_results = hypothesis_tests(processed_data)

    # Task 5: correlations and Bonferroni correction
    corr_results = correlation_and_multiple_comparisons(processed_data)

    # Task 6: summary report
    summary_report(processed_data, t_results, corr_results)

    logger.info("World Happiness pipeline completed.")


if __name__ == "__main__":
    happiness_pipeline()
