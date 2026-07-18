# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 3


# --- Preprocessing ---

"""Preprocessing Question 1

Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42.
Print the shapes of all four arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.multiclass import OneVsRestClassifier

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- Preprocessing ---

"""Preprocessing Question 2

Fit a StandardScaler on X_train and use it to transform both X_train and X_test.
Print the mean of each column in X_train_scaled -- they should all be very close to 0.
Add a comment explaining in one sentence why you fit the scaler on X_train only.
"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Mean of each column in X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# Fit the scaler on X_train only to avoid data leakage from the test set.

# --- KNN ---

"""KNN Question 1

Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train),
and predict on the test set. Print the accuracy score and the full classification report.
"""

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print(f"KNN accuracy (unscaled): {accuracy_score(y_test, y_pred_knn)}")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# --- KNN ---

"""KNN Question 2

Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled).
Print the accuracy score. Add a comment: does scaling improve performance, hurt it, or make no difference?
Why might that be for this particular dataset?
"""

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print(f"KNN accuracy (scaled): {accuracy_score(y_test, y_pred_knn_scaled)}")

# Scaling slightly hurt performance for this train/test split.
# The Iris features are already on similar scales, so scaling did not provide a benefit here.

# --- KNN ---

"""KNN Question 3

Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data.
Print each fold score, the mean, and the standard deviation.
Add a comment: is this result more or less trustworthy than a single train/test split, and why?
"""

cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=5)

print("5-fold CV scores:")
print(cv_scores)
print(f"Mean CV score: {cv_scores.mean()}")
print(f"Std CV score: {cv_scores.std()}")

# This is more trustworthy than a single train/test split because it averages performance across multiple folds.

# --- KNN ---

"""KNN Question 4

Loop over k values [1, 3, 5, 7, 9, 11, 13, 15].
For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score.
Add a comment identifying which k you would choose and why.
"""

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = -1

for k in k_values:
    scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5
    )
    mean_score = scores.mean()
    print(f"k={k}, mean CV score={mean_score}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

"""Result:
k=1, mean CV score=0.9416666666666668
k=3, mean CV score=0.9583333333333334
k=5, mean CV score=0.975
k=7, mean CV score=0.975
k=9, mean CV score=0.9583333333333334
k=11, mean CV score=0.9583333333333334
k=13, mean CV score=0.9583333333333334
k=15, mean CV score=0.9666666666666666
"""

print(f"\nBest k: {best_k}")


# I would choose the k = 5 or k = 7 for their highest mean CV scores because it performs best on validation data.

# --- Classifier Evaluation ---

"""Classifier Evaluation Question 1

Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay,
passing display_labels=iris.target_names. Save the figure to outputs/knn_confusion_matrix.png.
Add a comment: which pair of species does the model most often confuse (if any)?
"""

cm = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format=".0f")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

# The model does not confuse any species on this test set because every sample was classified correctly.

# --- The sklearn API: Decision Trees ---

"""Decision Trees Question 1

Create a DecisionTreeClassifier(max_depth=3, random_state=42), fit it on the unscaled training data,
and predict on the test set. Print the accuracy score and classification report.
Add a comment comparing the Decision Tree accuracy to KNN. Then add a second comment:
given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?
"""

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print(f"Decision Tree accuracy: {accuracy_score(y_test, y_pred_tree)}")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))

# The Decision Tree achieved 96.7% accuracy, which is slightly lower than the 100% accuracy from KNN.
# Scaling should have little to no effect on Decision Trees because they split using feature thresholds rather than distance calculations.

# --- Logistic Regression and Regularization ---

"""Logistic Regression Question 1

Train three logistic regression models on the scaled Iris data, identical in every way except for the C parameter:
C=0.01, C=1.0, and C=100. Use max_iter=1000 and solver='liblinear' for all three.
For each model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum().
Add a comment: what happens to the total coefficient magnitude as C increases?
What does this tell you about what regularization is doing?
"""


for C in [0.01, 1.0, 100]:
    model = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=1000,
        )
    )

    model.fit(X_train_scaled, y_train)

    coef_sum = sum(np.abs(est.coef_).sum() for est in model.estimators_)

    print(f"C={C}")
    print(f"Total coefficient magnitude: {coef_sum}")

# As C increases, the total coefficient magnitude increases because regularization becomes weaker.
# Smaller values of C apply stronger regularization, which shrinks the model coefficients.

# --- PCA ---

"""PCA Question 1

Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9),
using cmap='gray_r' with each digit's label as the title. Save the figure to outputs/sample_digits.png.
"""

digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

print(f"X_digits shape: {X_digits.shape}")
print(f"images shape: {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()

"""PCA Question 2

Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits).
Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar.
Save the figure to outputs/pca_2d_projection.png.
"""

pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("outputs/pca_2d_projection.png")
plt.close()

# Same-digit images tend to cluster somewhat, but there is overlap because 2 components cannot capture all variation.

"""PCA Question 3

Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components using
np.cumsum(pca.explained_variance_ratio_). Save to outputs/pca_variance_explained.png.
"""

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

# The cumulative variance plot shows how many components are needed to retain
# most of the information. Around 20 components capture approximately 80% of
# the variance, while fewer components lose more information.

"""PCA Question 4

Using this function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits in X_digits
using reconstruction through principal components n = 2, 5, 15, and 40.
Build a grid of subplots where rows correspond to each n value and columns show those 5 digits.
Add an "Original" row at the top. Save to outputs/pca_reconstructions.png.
"""


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_values = [2, 5, 15, 40]
fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 10))

for col in range(5):
    axes[0, col].imshow(images[col], cmap="gray_r")
    axes[0, col].axis("off")
    axes[0, col].set_title("Original")

for row, n in enumerate(n_values, start=1):
    for col in range(5):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap="gray_r")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(f"n={n}")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()

# Digits become clearly recognizable around 5 to 15 components, and that roughly matches where the variance curve starts to level off.
