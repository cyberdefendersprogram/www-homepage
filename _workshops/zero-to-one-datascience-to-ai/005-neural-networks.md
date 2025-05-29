---
layout: workshop
order: 005
slug: logistic-regression-with-scikit-learn
title: Logistic Regression With Scikit-learn
module_category: Machine Learning
workshop_name: zero-to-one-datascience-to-ai
date: 2025-05-23
---
## Lesson 5: Logistic Regression with Scikit-learn 🎯

**Objective:**

* Understand the basic concept of Logistic Regression for classification.
* Implement Logistic Regression using Scikit-learn on the Titanic dataset.
* Understand its output (probabilities and class labels).
* Evaluate the predictions using standard classification metrics.

**Recap of Lesson 3:**
* Data was preprocessed: missing values handled, categorical features one-hot encoded, numerical features scaled.
* Data was split into `X_train`, `y_train`, `X_val`, `y_val`. These are Pandas DataFrames/Series.

---

### 1. What is Logistic Regression? (Conceptual - 10 min)
Logistic Regression is a fundamental algorithm used for **binary classification problems** (where the outcome has two classes, e.g., Survived/Died, Yes/No). While its name includes "Regression," it's a classification algorithm.

* **Core Idea:** Instead of fitting a straight line directly to the 0/1 outcomes, Logistic Regression models the **probability** that an instance belongs to a particular class.
* **Sigmoid Function (Logistic Function):** It uses the sigmoid function to S-shape a linear combination of features, squashing the output into a range between 0 and 1.
    * $P(y=1|X) = \frac{1}{1 + e^{-z}}$
    * where $z = W_1X_1 + W_2X_2 + \dots + W_nX_n + b$ (the linear part, same as in linear regression).
    * $P(y=1|X)$ is the probability of the instance belonging to class 1, given its features $X$.
    * $W_i$ are the weights (coefficients) learned by the model.
    * $b$ is the bias term (intercept).
* **Log-odds (Logit):** Logistic Regression models the log-odds of the outcome:
    * $log(\frac{P(y=1|X)}{1 - P(y=1|X)}) = z = W_1X_1 + W_2X_2 + \dots + W_nX_n + b$
* **Goal:** To find the weights ($W_i$) and bias ($b$) that maximize the likelihood of observing the given training data.
* **Output:**
    * It can output probabilities (e.g., 0.7 means 70% chance of belonging to class 1).
    * These probabilities are then typically converted to class labels (0 or 1) by applying a threshold (usually 0.5).

---

### 2. Applying Logistic Regression to a Classification Problem (Titanic) (Conceptual - 10 min)
The Titanic 'Survived' variable is binary (0 for Died, 1 for Survived), making Logistic Regression a very suitable algorithm.

* **Suitability:**
    * Designed specifically for binary (and can be extended to multi-class) classification.
    * Outputs probabilities, which are interpretable and useful (e.g., assessing confidence).
    * Provides a good baseline model for classification tasks.
* **How it works with 'Survived':**
    * The model will learn to map the input features (Pclass, Age, Sex, etc.) to the probability of survival.
    * For example, it might learn that being female increases the probability of survival, or being in a lower class decreases it.
* **Interpreting Output for Classification:**
    * `model.predict_proba(X)` will output two columns of probabilities for each instance: P(Survived=0) and P(Survived=1).
    * `model.predict(X)` will directly output the class label (0 or 1) by using a default threshold of 0.5 on P(Survived=1).
        * If P(Survived=1) $\ge 0.5$, predict 1 (Survived).
        * If P(Survived=1) $< 0.5$, predict 0 (Died).

---

### 3. Implementation with Scikit-learn (Practical - 20 min)
We'll use the `LogisticRegression` model from `sklearn.linear_model`.

```python
# Ensure necessary libraries and data are loaded from previous lessons
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression # Changed from LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt # For potential plotting if needed

# --- Assume X_train, y_train, X_val, y_val are pre-existing Pandas DataFrames/Series ---
# For this script to be runnable standalone, let's create dummy preprocessed data
# In your actual workflow, you'd use the data from your previous lessons.
if 'X_train' not in locals() or 'y_train' not in locals():
    print("--- X_train, y_train not found. Creating dummy preprocessed data for Lesson 5. ---")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    num_samples = 891
    num_features = 8
    dummy_X_data = np.random.rand(num_samples, num_features)
    feature_names = [f'feature_{i}' for i in range(num_features)]
    X_full = pd.DataFrame(dummy_X_data, columns=feature_names)
    scaler = StandardScaler()
    X_scaled_data = scaler.fit_transform(X_full)
    X_full_scaled = pd.DataFrame(X_scaled_data, columns=feature_names)
    y_full = pd.Series(np.random.randint(0, 2, num_samples), name='Survived')
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_scaled, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    print(f"Dummy X_train shape: {X_train.shape}, Dummy y_train shape: {y_train.shape}")

# 1. Initialize the Logistic Regression model
# You can specify a solver and max_iter if you get convergence warnings.
# 'liblinear' is good for small datasets. 'lbfgs' is a common default.
log_reg_model = LogisticRegression(solver='liblinear', random_state=42) # Changed model

# 2. Train the model
# Scikit-learn's LogisticRegression expects y_train to be a 1D array or Series.
print("\n--- Training Logistic Regression Model (Scikit-learn) ---")
log_reg_model.fit(X_train, y_train)
print("Logistic Regression model trained.")

# You can inspect the learned coefficients (weights) and intercept if interested
# print(f"Coefficients (W): {log_reg_model.coef_}")
# print(f"Intercept (b): {log_reg_model.intercept_}")

# 3. Make predictions (class labels)
# .predict() directly gives class labels (0 or 1) using a 0.5 threshold internally.
y_pred_class_train = log_reg_model.predict(X_train)
y_pred_class_val = log_reg_model.predict(X_val)

print("\nFirst 5 class predictions on validation set:")
print(y_pred_class_val[:5])

# 4. Optionally, get probabilities
y_pred_proba_train = log_reg_model.predict_proba(X_train)
y_pred_proba_val = log_reg_model.predict_proba(X_val)

print("\nFirst 5 probability predictions on validation set (P(Died), P(Survived)):")
print(y_pred_proba_val[:5])
# The second column is P(Survived=1)
print("\nProbability of survival for first 5 validation samples:")
print(y_pred_proba_val[:5, 1])
```

4. Evaluation (Practical - 15 min)
We use standard classification metrics on the predicted class labels.

```python
# (Continuing from the previous code block)

print("\n--- Logistic Regression Model Evaluation (Classification Metrics) ---")

# Accuracy
train_accuracy_logreg = accuracy_score(y_train, y_pred_class_train)
val_accuracy_logreg = accuracy_score(y_val, y_pred_class_val)
print(f"Training Accuracy: {train_accuracy_logreg*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy_logreg*100:.2f}%")

# Confusion Matrix for Validation Set
print("\nValidation Set Confusion Matrix:")
cm_logreg = confusion_matrix(y_val, y_pred_class_val)
print(cm_logreg)
# For better display:
# TN = cm_logreg[0,0]; FP = cm_logreg[0,1]
# FN = cm_logreg[1,0]; TP = cm_logreg[1,1]
# print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

# Precision, Recall, F1-Score for Validation Set
# Ensure there are positive predictions and more than one class in true labels for meaningful metrics
if len(np.unique(y_val)) > 1 and len(np.unique(y_pred_class_val)) > 0:
    precision_logreg = precision_score(y_val, y_pred_class_val, zero_division=0)
    recall_logreg = recall_score(y_val, y_pred_class_val, zero_division=0)
    f1_logreg = f1_score(y_val, y_pred_class_val, zero_division=0)
    print(f"Validation Precision: {precision_logreg:.4f}")
    print(f"Validation Recall: {recall_logreg:.4f}")
    print(f"Validation F1-Score: {f1_logreg:.4f}")
else:
    print("Could not calculate Precision/Recall/F1 for validation set (e.g., no positive predictions or only one class in true labels).")

# You can also explore metrics like ROC AUC score
# from sklearn.metrics import roc_auc_score
# val_roc_auc = roc_auc_score(y_val, y_pred_proba_val[:, 1]) # Use probabilities for ROC AUC
# print(f"Validation ROC AUC Score: {val_roc_auc:.4f}")
```

5. Discussion (Conceptual - 5 min)
-  Performance: How does Logistic Regression perform? It often provides a strong baseline for binary classification tasks. Its performance will depend on how linearly separable the data is in the feature space transformed by the logit function.
- Why this is suitable for classification:
- Probabilistic Output: Provides calibrated probabilities, which are more informative than just class labels.
- Interpretable Coefficients (to some extent): The sign and magnitude of coefficients can give insights into feature importance and their relationship with the outcome (though interpretation needs care, especially with scaled features).
- Efficiency: Generally fast to train, especially with solvers like 'liblinear' or 'saga' for larger datasets.
Less Prone to Overfitting (than complex models on small datasets): It's a simpler model compared to, say, deep neural networks, which can be an advantage with limited data.
- Value of this exercise: It demonstrates the standard Scikit-learn workflow for a robust and widely-used classification algorithm. It sets a benchmark against which more complex models can be compared.

Wrap-up & Next Steps:
- Recap: We've implemented Logistic Regression using Scikit-learn, applied it to the Titanic classification task, understood how to get both class predictions and probabilities, and evaluated it using standard classification metrics. We've also discussed its suitability for this type of problem.
- Teaser for Lesson 6: Having explored a powerful classical classification algorithm, we'll now delve into the Core Concepts of Neural Networks. This will lay the groundwork for understanding more powerful and flexible models capable of learning complex non-linear relationships, which are often needed to further improve performance on tasks like the Titanic challenge.
