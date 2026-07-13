# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 1

# --- Prefect ---

# Prefect Q2

"""The answer to this question should go in prefect_warmup.py, not warmups_01.py.
Rebuild the pipeline from Q1 using Prefect. Copy your three functions from Pipeline Question 1 (create_series, clean_data, summarize_data) into this file and turn them into Prefect tasks using @task.
Turn data_pipeline() into a Prefect flow using @flow. Inside the flow, call the three tasks in order and return the summary dictionary.

Add this block at the bottom of the file so the flow runs when you execute the script directly:

if __name__ == "__main__":
    pipeline_flow()

Run your workflow from the terminal:

python prefect_warmup.py

The summary values should match what you got in Question 1.
Finally, add a comment block at the bottom of prefect_warmup.py answering these two questions:

    This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
    Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
"""

import numpy as np
import pandas as pd
from prefect import task, flow

arr = np.array(
    [12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0]
)


@task
def create_series(arr):
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }


@flow
def pipeline_flow(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary


if __name__ == "__main__":

    for key, value in pipeline_flow(arr).items():
        print(f"{key}: {value}")


"""
Prefect question 1:
This pipeline is very simple: three short steps on a single in-memory array with no schedules, retries, or external systems.
For this kind of one-off script, Prefect adds overhead (extra configuration, runtime, dashboard, etc.) without providing much benefit beyond what plain Python functions and prints already give.

Prefect question 2:
Even for simple pipelines, a framework like Prefect becomes useful when:
- The pipeline needs to run on a schedule (e.g., hourly or daily) and we want automatic retries on failures.
- Steps depend on external resources (files, APIs, databases) where logging, monitoring, and error handling are important.
- You want a central place (UI/API) to see runs, logs, and statuses, rather than reading local print output.
- Multiple small pipelines need to be orchestrated together across environments or machines.

In those scenarios, keeping the logic simple but using Prefect for orchestration, logging, and observability can still be a big win.
"""
