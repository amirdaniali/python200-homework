import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

os.makedirs("outputs", exist_ok=True)

# ============================================================
# Task 1 - Load and Explore
# ============================================================

"""
Task 1

Load the Spambase dataset.

Explore the dataset by:
- printing the number of emails
- printing the class balance
- explaining what that means for interpreting accuracy
- creating boxplots for:
    word_freq_free
    char_freq_!
    capital_run_length_total
- commenting on feature distributions and feature scales.
"""

spam = fetch_ucirepo(id=94)

X = spam.data.features
y = spam.data.targets.squeeze()

print(f"Dataset shape: {X.shape}")
print(f"Number of emails: {len(X)}")

print("\nClass counts:")
print(y.value_counts())

print("\nClass percentages:")
print(y.value_counts(normalize=True) * 100)

# The dataset is moderately imbalanced, with about 61% ham emails and 39% spam emails.
# Accuracy is useful, but it should not be the only metric because predicting every email
# as ham would still achieve around 61% accuracy.

features = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total",
]

for feature in features:

    plt.figure(figsize=(6, 4))

    plt.boxplot(
        [
            X.loc[y == 0, feature],
            X.loc[y == 1, feature],
        ],
        tick_labels=["Ham", "Spam"],
    )

    plt.title(feature)
    plt.ylabel(feature)

    plt.tight_layout()

    filename = feature.replace("/", "_").replace("!", "exclamation")

    plt.savefig(f"outputs/{filename}_boxplot.png")
    plt.close()

# Many word-frequency features contain mostly zeros because most emails never
# contain a particular word. The features are heavily skewed.

# Feature scales vary widely. Word frequencies are small percentages while
# capital letter statistics can be much larger values. This difference matters
# for distance-based models such as KNN and Logistic Regression, so scaling is
# important before training those models.


# ============================================================
# Task 2 - Prepare Your Data
# ============================================================

"""
Task 2

Split the data into training and testing sets.

Scale the features using StandardScaler.

Fit PCA using only the scaled training data.

Plot cumulative explained variance and determine how many principal
components explain at least 90% of the variance.

Transform the train and test data using those components.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Fit the scaler only on the training data to avoid leaking information
# from the test set into the preprocessing step.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training means:")
print(X_train_scaled.mean(axis=0))

# PCA should also be fit only on the training data.

pca = PCA()

pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
)

plt.axhline(
    y=0.90,
    color="red",
    linestyle="--",
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/pca_explained_variance.png")
plt.close()

n_components = np.argmax(cumulative_variance >= 0.90) + 1

print(f"\nComponents needed for 90% variance: {n_components}")

# About 43 principal components are needed to explain 90% of the variance.

X_train_pca = pca.transform(X_train_scaled)[:, :n_components]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components]

print("\nPCA training shape:", X_train_pca.shape)
print("PCA testing shape:", X_test_pca.shape)

# The full scaled data and the PCA-reduced data are both kept because
# different classifiers may benefit from different preprocessing choices.

# --- Task 3 ---

"""
Task 3: A Classifier Comparison

Build and evaluate the following five classifiers. For each, print the
accuracy and the full classification report.

- KNeighborsClassifier(n_neighbors=5) trained on the unscaled data
- KNeighborsClassifier(n_neighbors=5) trained on the scaled data, and again
  on the PCA-reduced data from Task 2
- DecisionTreeClassifier(random_state=42) with max_depth values of
  3, 5, 10, and None
- LogisticRegression(C=1.0, max_iter=1000) trained on the scaled data
  and again on the PCA-reduced data

Compare the models and determine which settings perform best.
"""

from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

print("\n-----------------------------------------")
print("KNN (Unscaled)")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# KNN works directly on distances between samples.
# This version uses the original feature values.


print("\n-----------------------------------------")
print("KNN (Scaled)")

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))
print(classification_report(y_test, y_pred_knn_scaled))

# Scaling puts every feature on a similar scale so large-valued
# features do not dominate the distance calculation.


print("\n-----------------------------------------")
print("KNN (PCA)")

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

y_pred_knn_pca = knn_pca.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred_knn_pca))
print(classification_report(y_test, y_pred_knn_pca))

# PCA reduces the number of dimensions while keeping most of the variance.
# This may improve or slightly reduce performance depending on the dataset.


print("\n-----------------------------------------")
print("Decision Tree Depth Comparison")

depths = [3, 5, 10, None]

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    train_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)

    print(f"Depth: {depth}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy : {test_accuracy:.4f}")
    print()

# Although the unlimited tree has slightly higher test accuracy, its training
# accuracy is almost perfect (0.9997), showing strong overfitting.
# max_depth=10 provides nearly the same test performance while reducing the
# risk of memorizing training examples, so it is a better production choice.


best_depth = 10

print("\nUsing max_depth =", best_depth)

tree = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42,
)

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# As tree depth increases, the training accuracy usually increases.
# If test accuracy stops improving while training accuracy keeps rising,
# the tree is beginning to overfit.

# The unlimited tree gives the highest test accuracy, but the training
# accuracy is almost perfect, showing signs of overfitting.
# A depth of 10 provides nearly the same test performance while reducing
# the gap between training and testing accuracy, so it may generalize better.


print("\n-----------------------------------------")
print("Logistic Regression (Scaled)")

log_scaled = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=1000,
)

log_scaled.fit(X_train_scaled, y_train)

y_pred_log_scaled = log_scaled.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_log_scaled))
print(classification_report(y_test, y_pred_log_scaled))

# Logistic Regression benefits from scaling because it learns
# coefficients directly from the feature values.


print("\n-----------------------------------------")
print("Logistic Regression (PCA)")

log_pca = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=1000,
)

log_pca.fit(X_train_pca, y_train)

y_pred_log_pca = log_pca.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred_log_pca))
print(classification_report(y_test, y_pred_log_pca))

# Logistic Regression performed the best among the tested models.
# Scaling improved both KNN and Logistic Regression because these models
# depend on feature distances or coefficient calculations.
#
# PCA slightly reduced performance compared with using all scaled features.
# This suggests that the removed features contained useful information for
# spam classification.
#
# For a spam filter, accuracy alone is not enough. False negatives
# (spam reaching the user's inbox) are inconvenient, but false positives
# (important emails being marked as spam) can be more harmful.
# A good spam filter should balance precision and recall rather than only
# maximizing accuracy.


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os

print("\n-----------------------------------------")
print("Random Forest")


rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("Accuracy:", rf_accuracy)
print(classification_report(y_test, y_pred_rf))


# Random Forest uses many decision trees and combines their predictions.
# This usually reduces overfitting compared with a single decision tree.


print("\n-----------------------------------------")
print("Feature Importances")


feature_names = X.columns


# Decision Tree feature importance
tree_importance = pd.DataFrame(
    {"Feature": feature_names, "Importance": tree.feature_importances_}
)

tree_importance = tree_importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Decision Tree Features:")
print(tree_importance.head(10))


# Random Forest feature importance
rf_importance = pd.DataFrame(
    {"Feature": feature_names, "Importance": rf.feature_importances_}
)

rf_importance = rf_importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Random Forest Features:")
print(rf_importance.head(10))


# The two models may identify similar important features,
# but Random Forest usually provides more stable importance estimates.


print("\nSaving feature importance plot...")


os.makedirs("outputs", exist_ok=True)


plt.figure(figsize=(10, 6))

top_features = rf_importance.head(10)

plt.barh(top_features["Feature"], top_features["Importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("outputs/feature_importances.png")

plt.close()

""" 
Model & Accuracy Results so far:
Random Forest   94.46%
Logistic Regression (Scaled) 92.94%
Logistic Regression (PCA) 91.86%
Decision Tree (None) 91.10%
KNN (Scaled 90.77%
KNN (PCA) 90.66%
Decision Tree (Depth 10) 90.88%
Decision Tree (Depth 5) 89.90%
KNN (Unscaled) 79.91%
"""
# --- Best Model Confusion Matrix ---

"""
For the best-performing classifier, create a confusion matrix.
The Random Forest achieved the highest accuracy in Task 3,
so it is used as the final model.
"""

print("\n-----------------------------------------")
print("Best Model Confusion Matrix")


best_model = rf
best_predictions = y_pred_rf


best_accuracy = accuracy_score(y_test, best_predictions)

print("Best Model Accuracy:", best_accuracy)


cm = confusion_matrix(y_test, best_predictions)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])


fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(ax=ax, cmap="Blues")

plt.title("Best Model Confusion Matrix")

plt.tight_layout()

plt.savefig("outputs/best_model_confusion_matrix.png")

plt.close()


print("\nConfusion Matrix:")
print(cm)


"""Random Forest is the best-performing classifier with the highest accuracy.

The confusion matrix shows:
True negatives (ham correctly identified): 549
False positives (ham marked as spam): 9
False negatives (spam marked as ham): 32
True positives (spam correctly identified): 331

The model makes more false negative errors than false positive errors.
For a spam filter, false positives can be more harmful because important
emails could be incorrectly sent to the spam folder.

Random Forest performed best on the Spambase dataset.
It achieved higher accuracy than Logistic Regression and Decision Trees.

Scaling helped KNN and Logistic Regression because those algorithms are
affected by feature magnitudes. Decision Trees and Random Forests do not
require scaling because they split data using feature thresholds.

PCA did not improve KNN or Logistic Regression performance. This suggests
that some of the removed components contained useful information for
distinguishing spam from ham.

The Random Forest feature importances match expectations:
dollar signs, exclamation marks, free-related words, and capital letter
patterns are strong indicators of spam.

Accuracy is not the only important metric for a spam filter. False positives
are especially costly because legitimate emails may be hidden from users.
A production spam filter should balance precision and recall depending on
the cost of each type of mistake.

# Summary:
#
# Random Forest performed best overall because it achieved the highest test
# accuracy and strong cross-validation performance.
#
# PCA improved neither KNN nor Logistic Regression because reducing dimensions
# removed some useful information. Scaling was more important than PCA for these
# models.
#
# For spam detection, accuracy alone is not enough. False positives are costly
# because legitimate emails may be incorrectly filtered as spam. False negatives
# are also important because spam can reach the user. A good spam filter should
# balance precision and recall depending on the user's priorities."""

# --- Task 4: Cross-Validation ---

"""
Task 4: Cross-Validation

Using cross_val_score with cv=5, run cross-validation on the training data
for each classifier from Task 3. For each, print the mean and standard
deviation of the fold scores.

Determine which model is the most accurate and which is the most stable.
"""

from sklearn.model_selection import cross_val_score

print("\n-----------------------------------------")
print("Task 4: Cross-Validation")


from sklearn.pipeline import Pipeline

models_for_cv = {
    "KNN Scaled": KNeighborsClassifier(n_neighbors=5),
    "KNN PCA": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression Scaled": LogisticRegression(
        C=1.0, max_iter=1000, solver="liblinear"
    ),
    "Logistic Regression PCA": LogisticRegression(
        C=1.0, max_iter=1000, solver="liblinear"
    ),
}


cv_results = {}


for name, model in models_for_cv.items():

    print("\n------------------------------")
    print(name)

    if name == "KNN Unscaled":
        X_cv = X_train

    elif name == "KNN Scaled":
        X_cv = X_train_scaled

    elif name == "KNN PCA":
        X_cv = X_train_pca

    elif name == "Logistic Regression":
        X_cv = X_train_scaled

    elif name == "Logistic Regression PCA":
        X_cv = X_train_pca

    else:
        X_cv = X_train

    scores = cross_val_score(model, X_cv, y_train, cv=5)

    cv_results[name] = scores

    print("Fold scores:")
    print(scores)

    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")


print("\n------------------------------")
print("CV Summary")


for name, scores in cv_results.items():
    print(f"{name}: " f"Mean={scores.mean():.4f}, " f"Std={scores.std():.4f}")


# The model with the highest mean CV score is the most accurate.
# The model with the lowest standard deviation is the most stable.
#
# Cross-validation gives a more reliable estimate than a single train/test
# split because the model is evaluated on multiple different subsets of data.
#
# Random Forest is expected to have lower variance than a single Decision Tree
# because it averages predictions from many different trees.

"""
Task 5: Building a Prediction Pipeline

Build two pipelines:
1. Best tree-based classifier
2. Best non-tree-based classifier

Fit both pipelines on the training data and print classification reports.
Explain why the pipelines have different structures and the practical value
of packaging preprocessing and models together.
"""

from sklearn.pipeline import Pipeline

print("\n-----------------------------------------")
print("Task 5: Prediction Pipelines")


#
# Random Forest Pipeline
#
# Random forests do not require scaling because they split data using feature
# thresholds rather than distance calculations.

rf_pipeline = Pipeline(
    [("classifier", RandomForestClassifier(n_estimators=100, random_state=42))]
)

rf_pipeline.fit(X_train, y_train)

rf_pipeline_pred = rf_pipeline.predict(X_test)

print("\nRandom Forest Pipeline Report:")
print(classification_report(y_test, rf_pipeline_pred))


#
# Logistic Regression Pipeline
#
# PCA was tested earlier but slightly reduced Logistic Regression accuracy,
# so the final pipeline includes only StandardScaler before the classifier.
# If PCA had improved performance, it would have been added as another
# pipeline step between the scaler and the classifier.

logreg_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
    ]
)


logreg_pipeline.fit(X_train, y_train)

logreg_pipeline_pred = logreg_pipeline.predict(X_test)

print("\nLogistic Regression Pipeline Report:")
print(classification_report(y_test, logreg_pipeline_pred))


#
# Pipeline Comments
#

# The pipelines have different structures because tree-based models do not
# require feature scaling, while logistic regression performs better when
# features are standardized.
#
# Pipelines reduce errors by automatically applying preprocessing learned from
# training data before making predictions. This prevents accidental data
# leakage and makes the model easier to deploy and share.
