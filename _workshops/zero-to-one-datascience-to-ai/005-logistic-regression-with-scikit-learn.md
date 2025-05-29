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

> [Github Notebook]()
>

Objective:
- Understand the basic concept of Logistic Regression and its suitability for classification.
- Implement Logistic Regression using Scikit-learn on the Titanic dataset.
- Evaluate the model's predictions using common classification metrics.
- Appreciate the ease of use of Scikit-learn for standard models.

Recap of Lesson 3 & 4:
- Data was preprocessed: missing values handled, categorical features one-hot encoded, numerical features scaled.
- Data was split into X_train, y_train, X_val, y_val. These are Pandas DataFrames/Series.

We've already built Logistic Regression from scratch to understand its mechanics. Now we see the library version.

### 1. What is Logistic Regression? (Conceptual Recap - 10 min)
Logistic Regression is a statistical model used for binary classification problems (predicting one of two outcomes, e.g., Survived/Died). Despite "Regression" in its name, it's a classification algorithm.

Core Idea: It models the probability that an input belongs to a particular class.
Sigmoid (Logistic) Function: It uses the sigmoid function to squash the output of a linear equation (similar to linear regression: z=W⋅X+b) into a probability between 0 and 1.

$$σ(z)=1+e−z$$

The output σ(z) is P(y=1∣X;W,b).
Decision Boundary: A threshold (typically 0.5) is applied to this probability to make a class prediction:
If P(y=1)≥0.5, predict class 1.
If P(y=1)<0.5, predict class 0.

>Suitability: It's well-suited for binary classification and provides probability estimates, which can be very useful. It assumes a linear relationship between the features and the log-odds of the outcome.

### 2. Why Use Scikit-learn's Logistic Regression? (Conceptual - 5 min)
While building from scratch (Lesson 4) is excellent for understanding, Scikit-learn offers:

- Optimization: Highly optimized algorithms (solvers like 'liblinear', 'lbfgs', 'saga') for faster training.
- Features: Built-in support for regularization (L1, L2), handling multi-class problems, and more.
- Ease of Use: A consistent API for training, predicting, and evaluating models.
- Robustness: Well-tested and widely used.
This lesson demonstrates how quickly we can implement and test a standard model using such a library.

### 3. Implementation with Scikit-learn (Practical - 20 min)
We'll use the LogisticRegression model from sklearn.linear_model.

```python
# Ensure necessary libraries and data are loaded from previous lessons
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression # Changed from LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt # For potential plotting if needed

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
# Common parameters:
# solver: Algorithm to use in the optimization problem. Default is 'lbfgs'.
#         'liblinear' is good for small datasets. 'saga' supports L1 and L2.
# C: Inverse of regularization strength; smaller values specify stronger regularization. Default is 1.0.
# max_iter: Maximum number of iterations taken for the solvers to converge.
log_reg_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
# 2. Train the model
# Scikit-learn's LogisticRegression expects y_train to be a 1D array or Series.
print("\n--- Training Logistic Regression Model (Scikit-learn) ---")
log_reg_model.fit(X_train, y_train)
print("Logistic Regression model trained.")

# You can inspect the learned coefficients (weights) and intercept if interested
# print(f"Coefficients (W): {log_reg_model.coef_}")
# print(f"Intercept (b): {log_reg_model.intercept_}")

# 3. Make predictions
# .predict() gives class labels directly (0 or 1)
y_pred_class_train_logreg = log_reg_model.predict(X_train)
y_pred_class_val_logreg = log_reg_model.predict(X_val)

# .predict_proba() gives probabilities for each class [P(class=0), P(class=1)]
y_pred_proba_val_logreg = log_reg_model.predict_proba(X_val)

print("\nFirst 5 class predictions (0 or 1) on validation set:")
print(y_pred_class_val_logreg[:5])
print("\nFirst 5 probability predictions [P(Died), P(Survived)] on validation set:")
print(y_pred_proba_val_logreg[:5])
# We are interested in P(Survived), which is the second column
print("\nFirst 5 probabilities of 'Survived' on validation set:")
print(y_pred_proba_val_logreg[:5, 1])
```

4. Evaluation (Practical - 15 min)
We use standard classification metrics.

```python
print("\n--- Logistic Regression Model Evaluation (Scikit-learn) ---")

# Accuracy
train_accuracy_logreg = accuracy_score(y_train, y_pred_class_train_logreg)
val_accuracy_logreg = accuracy_score(y_val, y_pred_class_val_logreg)
print(f"Training Accuracy: {train_accuracy_logreg*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy_logreg*100:.2f}%")

# Confusion Matrix for Validation Set
print("\nValidation Set Confusion Matrix:")
cm_logreg = confusion_matrix(y_val, y_pred_class_val_logreg)
print(cm_logreg)
# For better display:
TN = cm_logreg[0,0]; FP = cm_logreg[0,1]
FN = cm_logreg[1,0]; TP = cm_logreg[1,1]
print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")


# Precision, Recall, F1-Score for Validation Set
if len(np.unique(y_val)) > 1: # Ensure there are positive and negative classes in true labels
    precision_logreg = precision_score(y_val, y_pred_class_val_logreg, zero_division=0)
    recall_logreg = recall_score(y_val, y_pred_class_val_logreg, zero_division=0)
    f1_logreg = f1_score(y_val, y_pred_class_val_logreg, zero_division=0)
    print(f"Validation Precision: {precision_logreg:.4f}")
    print(f"Validation Recall: {recall_logreg:.4f}")
    print(f"Validation F1-Score: {f1_logreg:.4f}")
else:
    print("Could not calculate Precision/Recall/F1 for validation set (e.g., only one class in true labels).")

# (Optional) ROC Curve and AUC Score - Good for evaluating probability-based classifiers
# from sklearn.metrics import roc_curve, roc_auc_score
# y_pred_proba_survived_val = y_pred_proba_val_logreg[:, 1] # Probabilities for the positive class (Survived)
# fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba_survived_val)
# auc_score = roc_auc_score(y_val, y_pred_proba_survived_val)
# print(f"Validation AUC Score: {auc_score:.4f}")

# plt.figure(figsize=(8,6))
# plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.2f})")
# plt.plot([0, 1], [0, 1], 'k--') # Random guessing line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()
```

5. Discussion (Conceptual - 5 min)
- Performance: Compare the Scikit-learn Logistic Regression's performance (accuracy, F1-score) to the from-scratch version (Lesson 4). The Scikit-learn version is likely to be similar or slightly better due to optimized solvers and default regularization.
- Ease of Use: Note how few lines of code are needed to train and predict with Scikit-learn.
- Interpretability: Logistic Regression coefficients can be interpreted in terms of log-odds, providing some insight into feature importance (though less direct than linear regression coefficients for continuous outcomes).
- Baseline Model: Logistic Regression is often a good baseline model for classification tasks. If more complex models (like neural networks) don't significantly outperform it, the simpler model might be preferred.

Wrap-up & Next Steps:
- Recap: We've implemented Logistic Regression using Scikit-learn, a powerful and convenient library. We trained the model, made predictions (both class labels and probabilities), and evaluated its performance using various classification metrics.
- Teaser for Lesson 6 (formerly Lesson 5): Now that we have experience with a robust library-based classifier (Logistic Regression) and a from-scratch implementation, we'll transition to understanding the Core Concepts of Neural Networks. This will prepare us to build more complex and potentially more powerful models that can capture non-linear relationships in data.
