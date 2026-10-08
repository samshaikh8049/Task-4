import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("LOGISTIC REGRESSION - BINARY CLASSIFICATION")
print("=" * 60)

data = data.csv

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target, name="target")


print("\nDataset loaded successfully.")
print(f"Number of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

print("\nTarget classes:")
print(data.target_names)

print("\nFirst 5 rows:")
print(X.head())
print("\nDataset information:")
print(X.info())

print("\nMissing values:")
print(X.isnull().sum().sum())

print("\nClass distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures standardized successfully.")
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully.")
y_pred = model.predict(X_test_scaled)

# Probability of class 1
y_probability = model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Logistic Regression")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.show()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-10, 10, 200)

sigmoid_values = sigmoid(z)

plt.figure(figsize=(8, 6))

plt.plot(
    z,
    sigmoid_values
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Threshold = 0.5"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Input (z)")
plt.ylabel("Sigmoid Probability")

plt.title("Sigmoid Function")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/sigmoid_curve.png",
    dpi=300
)

plt.show()
thresholds_to_test = np.arange(
    0.1,
    1.0,
    0.1
)

threshold_results = []

for threshold in thresholds_to_test:

    y_threshold_pred = (
        y_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        y_threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        y_threshold_pred,
        zero_division=0
    )

    threshold_accuracy = accuracy_score(
        y_test,
        y_threshold_pred
    )

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": threshold_accuracy,
        "Precision": threshold_precision,
        "Recall": threshold_recall
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

print(threshold_df.to_string(index=False))
plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.xlabel("Classification Threshold")

plt.ylabel("Score")

plt.title("Threshold Tuning")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/threshold_tuning.png",
    dpi=300
)

plt.show()

threshold_df.to_csv(
    "results/threshold_results.csv",
    index=False
)

print("\nThreshold results saved to:")
print("results/threshold_results.csv")

best_row = threshold_df.loc[
    threshold_df["Accuracy"].idxmax()
]

best_threshold = best_row["Threshold"]
best_accuracy = best_row["Accuracy"]

print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(f"Best threshold: {best_threshold:.1f}")
print(f"Best accuracy : {best_accuracy:.4f}")

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nAll plots have been saved in the 'results' folder.")

print("\nTask 4 completed successfully!")