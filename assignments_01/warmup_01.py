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
print(f"Shape: {df.info()}")
print(f"Data Types: {df.dtypes}")

# Pandas Q2

"""Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result."""

top_students = df[(df["passed"] == True) & (df["grade"] >= 80)]
print(f"Top Students: {top_students}")


# Pandas Q3

"""Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows)."""

df["grade_curved"] = df["grade"] + 5
print(f"Curved Grade: {df[:]}")

# Pandas Q4

"""Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together."""

df["name_upper"] = df["name"].str.upper()
print(f"Name Upper: {df[["name_upper","name"]]}")

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
# todo

"""Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]]."""

sliced_arr = arr[:2]
# print(f"Array Shape: {sliced_arr}")

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
