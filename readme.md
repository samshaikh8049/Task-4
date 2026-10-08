# AI & ML Internship – Task 4

## Classification with Logistic Regression

---

## 📌 Task Overview

This project is completed as part of the **AI & ML Internship – Task 4**.

The objective of this task is to build a **binary classification model using Logistic Regression** and evaluate its performance using different classification metrics.

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The dataset is used to predict whether a breast tumor is **Malignant** or **Benign**.

The task includes:

* Selecting a binary classification dataset
* Splitting the dataset into training and testing sets
* Standardizing the features
* Training a Logistic Regression model
* Evaluating the model
* Creating a confusion matrix
* Calculating precision and recall
* Calculating ROC-AUC
* Plotting the ROC curve
* Understanding the sigmoid function
* Tuning the classification threshold

These activities are based on the requirements provided in the internship task document.

---

## 🎯 Objective

The main objective of this project is to:

> **Build a binary classifier using Logistic Regression.**

Logistic Regression is a supervised machine learning algorithm commonly used for classification problems where the target variable has two possible classes.

---

## 📊 Dataset

### Breast Cancer Wisconsin (Diagnostic) Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The dataset contains features computed from digitized images of breast mass samples.

The target variable represents two classes:

| Value | Class     |
| ----- | --------- |
| 0     | Benign    |
| 1     | Malignant |

The dataset contains multiple numerical features describing characteristics of the cell nuclei.

Examples include:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

### Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

---

## 📁 Project Structure

```text
Task 4/
│
├── logistic_regression.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── breast_cancer.csv
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── sigmoid_curve.png
    ├── threshold_tuning.png
    └── threshold_results.csv
```

---

## 🛠️ Technologies Used

The following tools and Python libraries are used:

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Git
* GitHub

---

## 💻 Requirements

Make sure Python is installed on your computer.

Check the Python version using:

```bash
python --version
```

Install the required libraries using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

# 🔄 Project Workflow

The complete workflow of the project is:

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Feature and Target Separation
   ↓
Train-Test Split
   ↓
Feature Standardization
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
Precision & Recall
   ↓
ROC-AUC
   ↓
ROC Curve
   ↓
Sigmoid Function
   ↓
Threshold Tuning
```

---

# 1. Import Required Libraries

The project imports the required Python libraries:

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

The libraries are used for:

* Data manipulation
* Numerical calculations
* Visualization
* Data preprocessing
* Model training
* Model evaluation

---

# 2. Load the Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv("dataset/breast_cancer.csv")
```

The first few records can be viewed using:

```python
print(df.head())
```

The dataset information can be checked using:

```python
print(df.info())
```

---

# 3. Data Cleaning

Before training the model, unnecessary columns are removed.

For example:

```python
df = df.drop(columns=["id"], errors="ignore")
```

If the dataset contains an unwanted column such as:

```text
Unnamed: 32
```

it can be removed using:

```python
df = df.drop(columns=["Unnamed: 32"], errors="ignore")
```

The target column is converted into numerical values:

```python
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})
```

Here:

```text
M → 1 → Malignant
B → 0 → Benign
```

---

# 4. Separate Features and Target

The dataset is divided into input features and the target variable.

```python
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
```

Where:

```text
X = Input features
y = Target variable
```

---

# 5. Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The dataset is divided into:

```text
80% → Training data
20% → Testing data
```

The training data is used to train the model.

The testing data is used to evaluate how well the model performs on unseen data.

---

# 6. Feature Standardization

The numerical features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Standardization converts the features to a comparable scale.

The scaler is fitted only on the training data and then applied to the test data.

---

# 7. Train Logistic Regression Model

A Logistic Regression model is created using Scikit-learn:

```python
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The model is trained using:

```python
model.fit(X_train_scaled, y_train)
```

---

# 8. Make Predictions

The trained model is used to predict the test data:

```python
y_pred = model.predict(X_test_scaled)
```

The model's probability predictions are also obtained:

```python
y_probability = model.predict_proba(X_test_scaled)[:, 1]
```

The probability values are useful for ROC-AUC calculation and threshold tuning.

---

# 📈 Model Evaluation

The model is evaluated using several classification metrics.

---

## 9. Accuracy

Accuracy represents the percentage of correctly classified samples.

Formula:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Python:

```python
accuracy_score(y_test, y_pred)
```

---

## 10. Confusion Matrix

A confusion matrix shows the number of correct and incorrect predictions.

It contains:

```text
True Positive  (TP)
True Negative  (TN)
False Positive (FP)
False Negative (FN)
```

The confusion matrix is generated using:

```python
cm = confusion_matrix(y_test, y_pred)
```

A visualization is created using Seaborn.

The output is saved as:

```text
results/confusion_matrix.png
```

---

## 11. Precision

Precision measures how many predicted positive samples were actually positive.

Formula:

```text
Precision = TP / (TP + FP)
```

Python:

```python
precision = precision_score(y_test, y_pred)
```

A higher precision means fewer false-positive predictions.

---

## 12. Recall

Recall measures how many actual positive samples were correctly identified.

Formula:

```text
Recall = TP / (TP + FN)
```

Python:

```python
recall = recall_score(y_test, y_pred)
```

Recall is particularly useful when missing a positive case is costly.

---

# 📉 ROC Curve and ROC-AUC

## 13. ROC Curve

ROC stands for:

**Receiver Operating Characteristic**

The ROC curve plots:

```text
True Positive Rate
        vs
False Positive Rate
```

The ROC curve is generated using:

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)
```

The ROC curve is saved as:

```text
results/roc_curve.png
```

---

## 14. ROC-AUC

AUC stands for:

**Area Under the Curve**

ROC-AUC measures how well the model separates the two classes.

It is calculated using:

```python
roc_auc = roc_auc_score(
    y_test,
    y_probability
)
```

A value closer to `1.0` generally indicates better class separation, while a value around `0.5` indicates performance close to random classification.

---

# 📊 Sigmoid Function

Logistic Regression uses the sigmoid function to convert a model's output into a probability between 0 and 1.

The sigmoid function is:

```text
σ(z) = 1 / (1 + e^-z)
```

The Python implementation is:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The sigmoid curve is plotted and saved as:

```text
results/sigmoid_curve.png
```

The value of the sigmoid function ranges between:

```text
0 and 1
```

A commonly used classification threshold is:

```text
Probability >= 0.5 → Class 1
Probability < 0.5  → Class 0
```

---

# 🎚️ Threshold Tuning

Instead of always using a threshold of `0.5`, different thresholds can be tested.

The project tests thresholds such as:

```text
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
```

For every threshold, the following metrics are calculated:

* Accuracy
* Precision
* Recall

Example:

```python
y_threshold_pred = (
    y_probability >= threshold
).astype(int)
```

The results are stored in:

```text
results/threshold_results.csv
```

A graph showing the effect of the threshold is saved as:

```text
results/threshold_tuning.png
```

Threshold selection depends on the application.

For example, if recall is more important, a lower threshold may be considered to identify more positive cases.

---

# ▶️ How to Run the Project

### Step 1: Open the project folder

```bash
cd "Task 4"
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Python program

```bash
python logistic_regression.py
```

### Step 4: Check the output

The program displays:

* Dataset information
* Training/testing sample counts
* Accuracy
* Precision
* Recall
* ROC-AUC
* Classification report
* Confusion matrix
* Threshold results

The generated plots are saved inside:

```text
results/
```

---

# 📂 Generated Results

After running the program, the following files are generated:

| File                    | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `confusion_matrix.png`  | Confusion matrix visualization                         |
| `roc_curve.png`         | ROC curve                                              |
| `sigmoid_curve.png`     | Sigmoid function                                       |
| `threshold_tuning.png`  | Accuracy, precision and recall at different thresholds |
| `threshold_results.csv` | Threshold evaluation results                           |

---

# 📋 Model Results

After executing the program, record the actual results generated by your model here.

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | Add result |
| Precision | Add result |
| Recall    | Add result |
| ROC-AUC   | Add result |

> **Note:** Replace the values above with the actual values displayed when `logistic_regression.py` is executed.

---

# 🧠 What I Learned

Through this task, I learned:

1. How binary classification works.
2. How to prepare a dataset for machine learning.
3. How to split data into training and testing sets.
4. Why feature standardization is important.
5. How Logistic Regression performs classification.
6. How the sigmoid function converts model output into probabilities.
7. How to calculate accuracy, precision and recall.
8. How to create and interpret a confusion matrix.
9. How to evaluate a classifier using ROC-AUC.
10. How changing the classification threshold affects model performance.

The internship task identifies binary classification, evaluation metrics, and the sigmoid curve as key learning outcomes.

---

# ❓ Interview Questions and Answers

## 1. How does Logistic Regression differ from Linear Regression?

Linear Regression predicts continuous numerical values, whereas Logistic Regression is commonly used for classification problems.

Linear Regression produces a continuous output, while Logistic Regression uses the sigmoid function to produce a probability between 0 and 1.

---

## 2. What is the sigmoid function?

The sigmoid function converts a numerical value into a probability between 0 and 1.

Formula:

```text
σ(z) = 1 / (1 + e^-z)
```

It is used by Logistic Regression to estimate the probability of a class.

---

## 3. What is precision vs recall?

**Precision** measures how many predicted positive samples are actually positive.

**Recall** measures how many actual positive samples were correctly identified.

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)
```

---

## 4. What is the ROC-AUC curve?

The ROC curve shows the relationship between the True Positive Rate and False Positive Rate at different classification thresholds.

AUC represents the area under the ROC curve and indicates how well the model separates the classes.

---

## 5. What is a confusion matrix?

A confusion matrix is a table used to evaluate classification predictions.

It contains:

```text
True Positive
True Negative
False Positive
False Negative
```

---

## 6. What happens if classes are imbalanced?

When classes are imbalanced, accuracy alone may not provide a reliable picture of model performance.

Metrics such as:

* Precision
* Recall
* F1-score
* ROC-AUC

can provide additional information about model performance.

---

## 7. How do you choose the threshold?

The default threshold is commonly `0.5`, but it can be changed depending on the requirements of the application.

For example, if detecting positive cases is more important, a threshold that improves recall may be preferred.

---

## 8. Can Logistic Regression be used for multi-class problems?

Yes.

Logistic Regression can be extended to multi-class classification using approaches such as:

* One-vs-Rest
* Multinomial Logistic Regression

The interview questions above are the questions specified in the Task 4 document.

---

# ⚠️ Important Notes

* Do not train the model before splitting the data.
* Fit the `StandardScaler` only on training data.
* Apply the fitted scaler to the test data.
* Do not use the target column as an input feature.
* Check for missing values before training.
* Use probability predictions for ROC-AUC and threshold tuning.
* Replace the model results in this README with the actual values generated by your program.

---

# 🚀 Future Improvements

Possible improvements to this project include:

* Hyperparameter tuning
* Cross-validation
* F1-score evaluation
* Precision-Recall curve
* Handling class imbalance
* Comparing Logistic Regression with other classification algorithms
* Feature selection
* Model deployment using Flask or Streamlit

---

# 📤 GitHub Submission

The internship instructions require a separate GitHub repository for each task and ask for the code, dataset, screenshots if applicable, and a short README file.

Recommended repository structure:

```text
AI-ML-Internship-Task-4/
│
├── logistic_regression.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── breast_cancer.csv
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── sigmoid_curve.png
    ├── threshold_tuning.png
    └── threshold_results.csv
```

After completing the project, copy the GitHub repository link and submit it through the internship submission form.

---

# 👨‍💻 Project Status

```text
Task 4: Classification with Logistic Regression
Status: Completed
```

---

## 📚 References

1. UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic) Dataset
2. Scikit-learn Documentation
3. Pandas Documentation
4. NumPy Documentation
5. Matplotlib Documentation
6. Seaborn Documentation
