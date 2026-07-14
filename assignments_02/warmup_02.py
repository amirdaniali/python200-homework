# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 2

# --- The scikit-learn API ---

# Sklearn Q1

"""The core pattern in scikit-learn is create → fit → predict. Practice it here with a simple dataset: years of work experience versus annual salary.

import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Salary for someone with 4 years of experience: {model.predict([[4]])}")
print(f"Salary for someone with 8 years of experience: {model.predict([[8]])}")

# Sklearn Q2

"""scikit-learn requires the feature array X to be 2D even when you only have one feature. Start with this 1D array:

x = np.array([10, 20, 30, 40, 50])

Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D."""

x = np.array([10, 20, 30, 40, 50])

x_modified = x.reshape(-1, 1)
print(f"Original Shape: {x.shape}")
print(f"Modified Shape: {x_modified.shape}")

"""
Scikit-learn needs X to be 2D to distinguish between features and samples. The x[0] is the first feature, and x[1] is the second feature and so on. Each feature has multiple samples. Scikit needs to know if we have 5 features total with 1 sample each or if we have 1 feature with 5 samples."""

# Sklearn Q3

"""K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn. Use the code below to generate a synthetic dataset with three natural clusters:

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).
Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png."""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# plot: what K-Means discovers
kmeans = KMeans(n_clusters=3, random_state=42)  # 1. Create the model
kmeans.fit(X_clusters)  # 2. Fit -- find cluster centers
labels = kmeans.predict(X_clusters)  # 3. Predict a label for each point

for i, center in enumerate(kmeans.cluster_centers_):
    plt.plot(center[0], center[1], "kx")
    print(f"Cluster Center No {i}: {center}")
    print(f"Number of points in cluster {i}: {np.bincount(labels)[i]}")

plt.scatter(
    X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", s=60, alpha=0.7
)

plt.title("Clusters Found by K-Means")
plt.xlabel("Data Points")


plt.tight_layout()
plt.show()

# Linear Regression

# Linear Regression Q1

"""Linear Regression

The questions below all use the same synthetic medical costs dataset: 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), and an annual medical cost as the target. Generate it once and reuse the variables throughout.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

Linear Regression Question 1
Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.
Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?"""
