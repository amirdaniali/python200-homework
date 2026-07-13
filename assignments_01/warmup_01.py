# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 1

# --- Pandas ---

"""Pandas Question 1

Create the following DataFrame and print the first three rows, the shape, and the data types of each column.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# Pandas Q1
data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}
df = pd.DataFrame(data)

print(f"First 3 rows:\n{df.head(3)}")
print(f"Shape: {df.shape}")
print(f"Data Types:\n{df.dtypes}")

# Pandas Q2
"""Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result."""
top_students = df[(df["passed"] == True) & (df["grade"] > 80)]
print(f"Top Students:\n{top_students}")

# Pandas Q3
"""Add a new column called 'grade_curved' that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows)."""
df["grade_curved"] = df["grade"] + 5
print(f"Curved Grade DataFrame:\n{df}")

# Pandas Q4
"""Add a new column called 'name_upper' that contains each student's name in uppercase, using the .str accessor. Print the 'name' and 'name_upper' columns together."""
df["name_upper"] = df["name"].str.upper()
print(f"Name Upper:\n{df[['name', 'name_upper']]}")

# Pandas Q5
"""Group the DataFrame by 'city' and compute the mean grade for each city. Print the result."""
cities = df.groupby("city").agg({"grade": "mean"})
print(f"Cities Mean Grade:\n{cities}")

# Pandas Q6
"""Replace the value 'Austin' in the 'city' column with 'Houston'. Print the 'name' and 'city' columns to confirm the change."""
df["city"] = df["city"].str.replace("Austin", "Houston")
print(f"Names and Cities with 'Austin' replaced:\n{df[['name', 'city']]}")

# Pandas Q7
"""Sort the DataFrame by 'grade' in descending order and print the top 3 rows."""
df_sorted = df.sort_values(by="grade", ascending=False)
print(f"Top 3 students by grade:\n{df_sorted.head(3)}")

# --- NumPy ---

# NumPy Q1
"""Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim."""
arr = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr}")
print(f"Array Shape: {arr.shape}")
print(f"Array dtype: {arr.dtype}")
print(f"Array ndim: {arr.ndim}")

# NumPy Q2
"""Create the following 2D array and print its shape and size (total number of elements)."""
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(f"2D Array:\n{arr2d}")
print(f"2D Array Shape: {arr2d.shape}")
print(f"2D Array Size: {arr2d.size}")

# NumPy Q3
"""Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]]."""
sliced_arr = arr2d[:2, :2]
print(f"Sliced 2x2 Array:\n{sliced_arr}")

# NumPy Q4
"""Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both."""
array_of_zeros = np.zeros((3, 4))
array_of_ones = np.ones((2, 5))
print(f"Array of zeros:\n{array_of_zeros}")
print(f"Array of ones:\n{array_of_ones}")

# NumPy Q5
"""Create an array using np.arange(0, 50, 5). Print the array, its shape, mean, sum, and standard deviation."""
arranged_array = np.arange(0, 50, 5)
print(f"Arange Array: {arranged_array}")
print(f"Arange Array Shape: {arranged_array.shape}")
print(f"Arange Array Mean: {arranged_array.mean()}")
print(f"Arange Array Sum: {arranged_array.sum()}")
print(f"Arange Array Std: {arranged_array.std()}")

# NumPy Q6
"""Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1. Print the mean and standard deviation."""
normal_array = np.random.normal(0, 1, 200)
print(f"Random Normal Mean: {normal_array.mean()}")
print(f"Random Normal Std: {normal_array.std()}")

# --- Matplotlib ---

# Matplotlib Q1
"""Plot the following data as a line plot with title 'Squares', x-axis label 'x', and y-axis label 'y'."""
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(title="Squares", xlabel="x", ylabel="y")
plt.show()

# Matplotlib Q2
"""Create a bar plot for subject scores with title 'Subject Scores' and labeled axes."""
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

fig, ax = plt.subplots()
ax.bar(subjects, scores, color="purple")
ax.set_title("Subject Scores")
ax.set_xlabel("Subjects")
ax.set_ylabel("Scores")
plt.show()

# Matplotlib Q3
"""Plot the two datasets as a scatter plot on the same figure with different colors, legend, and labeled axes."""
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

fig, ax = plt.subplots()
ax.scatter(x1, y1, color="red", label="Dataset 1")
ax.scatter(x2, y2, color="blue", label="Dataset 2")
ax.legend()
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_title("Scatter Plot of Two Datasets")
plt.show()

# Matplotlib Q4
"""Create a figure with 1 row and 2 subplots: left line plot (x vs y from Q1), right bar plot (subjects and scores from Q2)."""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(x, y)
ax1.set_title("Squares Line Plot")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.bar(subjects, scores)
ax2.set_title("Subject Scores Bar Plot")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")

plt.tight_layout()
plt.show()

# --- Descriptive Statistics ---

# Descriptive Stats Q1
"""Given the list below, use NumPy to compute mean, median, variance, and standard deviation."""
data_ds = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data_ds)}")
print(f"Median: {np.median(data_ds)}")
print(f"Variance: {np.var(data_ds)}")
print(f"Standard Deviation: {np.std(data_ds)}")

# Descriptive Stats Q2
"""Generate 500 random values from N(65, 10) and plot a histogram with 20 bins."""
scores_dist = np.random.normal(65, 10, 500)
plt.hist(scores_dist, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3
"""Create a boxplot comparing two groups with labels and title 'Score Comparison'."""
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

fig, ax = plt.subplots()
ax.boxplot([group_a, group_b], labels=["Group A", "Group B"])
ax.set_ylabel("Scores")
ax.set_xlabel("Groups")
ax.set_title("Score Comparison")
plt.show()

# Descriptive Stats Q4
"""Side-by-side boxplots for normal vs exponential distributions with comments on skew and central tendency."""
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.boxplot(normal_data, labels=["Normal"])
ax1.set_ylabel("Values")

ax2.boxplot(skewed_data, labels=["Exponential"])
ax2.set_ylabel("Values")

fig.suptitle("Distribution Comparison")
plt.tight_layout()
plt.show()

# Exponential distribution is more skewed than the normal distribution.
# Mean is a good measure of central tendency for the normal distribution.
# Median is more appropriate for the exponential distribution because it is less affected by extreme values.

# Descriptive Stats Q5
"""Print mean, median, and mode of data1 and data2, and explain why mean and median differ for data2."""
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

# Convert to pandas Series to get mode easily
series1 = pd.Series(data1)
series2 = pd.Series(data2)

print(f"Data1 Mean: {series1.mean()}")
print(f"Data1 Median: {series1.median()}")
print(f"Data1 Mode: {series1.mode()[0]}")

print(f"Data2 Mean: {series2.mean()}")
print(f"Data2 Median: {series2.median()}")
print(f"Data2 Mode: {series2.mode()[0]}")

# For data2, the single extreme value (150) pulls the mean upward, but the median stays near
# the center of the bulk of the data. This makes the mean much larger than the median.

# --- Hypothesis Testing ---

# Hypothesis Q1
"""Run an independent samples t-test on two groups."""
group_a_ht = [72, 68, 75, 70, 69, 73, 71, 74]
group_b_ht = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a_ht, group_b_ht)
print("Independent t-test (Q1) t-statistic:", t_stat)
print("Independent t-test (Q1) p-value:", p_val)

# Hypothesis Q2
"""Using the p-value from Q1, print whether the result is significant at alpha = 0.05."""
alpha = 0.05
if p_val < alpha:
    print("The difference is statistically significant at alpha = 0.05.")
else:
    print("No statistically significant difference detected at alpha = 0.05.")

# Hypothesis Q3
"""Paired t-test on before/after scores."""
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat_paired, p_val_paired = stats.ttest_rel(before, after)
print(f"Paired t-test t-statistic: {t_stat_paired:.3f}")
print(f"Paired t-test p-value: {p_val_paired:.6f}")

# Hypothesis Q4
"""One-sample t-test against benchmark 70."""
scores_bench = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat_one, p_val_one = stats.ttest_1samp(scores_bench, 70)
print(f"One-sample t-test t-statistic: {t_stat_one:.3f}")
print(f"One-sample t-test p-value: {p_val_one:.6f}")

# Hypothesis Q5
"""One-tailed test: check whether group_a_ht scores are less than group_b_ht scores."""
t_stat_one_tail, p_val_one_tail = stats.ttest_ind(group_a_ht, group_b_ht, alternative="less")
print(f"One-tailed t-test p-value (group_a < group_b): {p_val_one_tail:.6f}")

# Hypothesis Q6
"""Plain-language conclusion for Q1."""
print(
    "Group B scores are higher on average than Group A, and the very small p-value "
    "suggests this difference is unlikely to be due to random chance alone."
)

# --- Correlation ---

# Correlation Q1
"""Compute Pearson correlation between x and y using np.corrcoef() and interpret the result."""
x_corr1 = [1, 2, 3, 4, 5]
y_corr1 = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x_corr1, y_corr1)
print(f"Full Correlation Matrix:\n{corr_matrix}")
print(f"Correlation Coefficient (0,1): {corr_matrix[0, 1]}")

# We expect the correlation to be 1 because y is exactly 2 * x, a perfect linear relationship.

# Correlation Q2
"""Use pearsonr() to compute correlation and p-value."""
x_corr2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_corr2 = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

r_val, p_val_corr = pearsonr(x_corr2, y_corr2)
print("pearsonr Correlation:", r_val)
print("pearsonr p-value:", p_val_corr)

# Correlation Q3
"""Create DataFrame and compute correlation matrix."""
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28],
}
df_people = pd.DataFrame(people)
df_corr = df_people.corr()
print("DataFrame Correlation Matrix:\n", df_corr)

# Correlation Q4
"""Create scatter plot of negatively related x and y."""
x_neg = [10, 20, 30, 40, 50]
y_neg = [90, 75, 60, 45, 30]

plt.scatter(x_neg, y_neg, color="orange")
plt.title("Negative Correlation")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

# Correlation Q5
"""Create correlation heatmap from df_corr."""
sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines ---

# Pipeline Q1
"""Plain Python data pipeline with create_series, clean_data, summarize_data, and data_pipeline()."""

arr_pipeline = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
                         18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


def create_series(arr):
    """Takes a NumPy array and returns a pandas Series named 'values'."""
    return pd.Series(arr, name="values")


def clean_data(series):
    """Removes NaN values from the Series and returns the cleaned Series."""
    return series.dropna()


def summarize_data(series):
    """Returns a dictionary with mean, median, std, and mode of the Series."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }


def data_pipeline(arr):
    """Connects create_series, clean_data, and summarize_data in sequence."""
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary


summary_result = data_pipeline(arr_pipeline)
for key, value in summary_result.items():
    print(f"{key}: {value}")