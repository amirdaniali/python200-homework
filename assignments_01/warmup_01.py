# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 1

# --- Pandas ---
# Pandas Q1
"""Pandas Question 1

Create the following DataFrame and print the first three rows, the shape, and the data types of each column.
"""

import pandas as pd

data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}
df = pd.DataFrame(data)

print(f"Num Rows: {df.head(3)}")
print(f"Shape: {df.shape}")
print(f"Data Types: {df.dtypes}")

# Pandas Q2

"""Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result."""

top_students = df[(df["passed"] == True) & (df["grade"] > 80)]
print(f"Top Students: {top_students}")


# Pandas Q3

"""Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows)."""

df["grade_curved"] = df["grade"] + 5
print(f"Curved Grade: {df[:]}")

# Pandas Q4

"""Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together."""

df["name_upper"] = df["name"].str.upper()
print(f"Name Upper: {df[["name","name_upper"]]}")

# Pandas Q5

"""Group the DataFrame by "city" and compute the mean grade for each city. Print the result."""

cities = df.groupby("city").agg({"grade": "mean"})
print(f"Cities Mean Grade: {cities}")

# Pandas Q6


"""Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change."""

df["city"] = df["city"].str.replace("Austin", "Houston")
print(f"Names with Cities While Austin is  Replaced: {df[["name","city"]]}")

# Pandas

"""Sort the DataFrame by "grade" in descending order and print the top 3 rows."""

df_sorted = df.sort_values(by="grade", ascending=False)
print(f"Sorted by Grade: {df_sorted.head(3)}")


# --- Numpy ---
# Numpy Q1

"""Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim."""

import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print(f"Array Shape: {np.shape(arr)}")
print(f"Array Type: {arr.dtype}")
print(f"Array ndim: {np.ndim(arr)}")

# Numpy Q2

"""Create the following 2D array and print its shape and size (total number of elements)."""

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"Array Shape: {np.shape(arr)}")
print(f"Array Size: {arr.size}")

# Numpy Q3

"""Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]]."""

sliced_arr = arr[:2, :2]
print(f"Sliced 2x2 Array: {sliced_arr}")

# Numpy Q4

"""Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both."""

array_of_zeros = np.zeros((3, 4))
print(f"Array of zeros: {array_of_zeros}")

array_of_ones = np.ones((2, 5))
print(f"Array of ones: {array_of_ones}")

# Numpy Q5

"""Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation."""

arraged_array = np.arange(0, 50, 5)
print(f"Array: {arraged_array}")
print(f"Array Shape: {np.shape(arraged_array)}")
print(f"Array Mean: {np.mean(arraged_array)}")
print(f"Array Sum: {np.sum(arraged_array)}")
print(f"Array Standard Deviation: {np.std(arraged_array)}")

# Numpy Q6

"""Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result."""

generated_array = np.random.normal(0, 1, 200)
print(f"Generated Random Values Mean: {np.mean(generated_array)}")
print(f"Generated Random Values Standard Deviation: {np.std(generated_array)}")

# --- Matplotlib ---

# Matplotlib Q1

"""Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]"""

import matplotlib.pyplot as plt

data = {"x": [0, 1, 2, 3, 4, 5], "y": [0, 1, 4, 9, 16, 25]}

fig, ax = plt.subplots()
ax.plot(data["x"], data["y"])
ax.set(title="Squares", xlabel="x", ylabel="y")
plt.show()


# Matplotlib Q2

"""Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]"""

data.update(
    {
        "subjects": ["Math", "Science", "English", "History"],
        "scores": [88, 92, 75, 83],
    }
)


plt.bar(data["subjects"], data["scores"], color="purple")
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3

"""Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]"""

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

fig, ax = plt.subplots()
ax.plot(x1, y1, color="red", label="Dataset 1")
ax.plot(x2, y2, color="blue", label="Dataset 2")
ax.legend()
ax.set(xlabel="X-Axis", ylabel="Y-Axis")
plt.show()


# Matplotlib Q4

"""Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing."""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(data["x"], data["y"])
ax1.set_title("Line Plot")
ax2.bar(data["subjects"], data["scores"])
ax2.set_title("Bar Plot")
plt.tight_layout()
plt.show()

# --- Descriptive Statistics ---

# Stats Q1

"""Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value."""

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")
