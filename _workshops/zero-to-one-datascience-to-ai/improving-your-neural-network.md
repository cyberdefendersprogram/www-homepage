---
layout: workshop
order: 8
slug: improving-your-neural-network
title: Improving Your Neural Network
workshop_name: zero-to-one-datascience-to-machinelearning
module_category: Advanced Machine Learning
date: 2025-05-23
---

## Lesson 8: Evaluating and Improving the Neural Network 📈🎯

**Objective:**
* Understand and implement more robust evaluation metrics beyond accuracy (Confusion Matrix, Precision, Recall, F1-Score).
* Learn to diagnose overfitting and underfitting in the neural network.
* Conceptually understand techniques to improve model performance and combat overfitting, such as regularization and hyperparameter tuning.

---

### Recap of Lesson 7:
* Successfully built, trained, and made initial predictions with a 1-hidden-layer neural network from scratch.
* Calculated training and validation accuracy for this neural network (`train_accuracy_nn`, `val_accuracy_nn`).
* Our key outputs from the previous lesson for evaluation are:
    * `Y_pred_train_nn`, `Y_pred_val_nn` (predictions from our NN model)
    * `Y_train_for_comp`, `Y_val_for_comp` (true labels, shaped as `(1, m_samples)`)
    * `final_trained_nn_params` (the learned weights and biases)

---

### 1. Beyond Accuracy: More Robust Evaluation Metrics (Conceptual & Practical - 30 min)

While accuracy (percentage of correct predictions) is a common metric, it can be misleading, especially for datasets where classes are imbalanced (e.g., if 90% of passengers died, a model predicting everyone died would be 90% accurate but useless).

Let's define:
* **Positive Class:** Typically the class of interest (e.g., Survived = 1).
* **Negative Class:** The other class (e.g., Died = 0).

* **True Positives (TP):** Correctly predicted positive (e.g., predicted Survived, actually Survived).
* **True Negatives (TN):** Correctly predicted negative (e.g., predicted Died, actually Died).
* **False Positives (FP):** Incorrectly predicted positive (e.g., predicted Survived, actually Died). Also called a "Type I error."
* **False Negatives (FN):** Incorrectly predicted negative (e.g., predicted Died, actually Survived). Also called a "Type II error."

#### a) Confusion Matrix

A table that summarizes the performance of a classification model by showing TP, TN, FP, FN.

|                   | Predicted Died (0) | Predicted Survived (1) |
| :---------------- | :----------------- | :--------------------- |
| **Actual Died (0)** | TN                 | FP                     |
| **Actual Survived (1)** | FN                 | TP                     |

Let's calculate these for our validation set predictions (`Y_pred_val_nn`) and true labels (`Y_val_for_comp`).
Remember, `Y_pred_val_nn` and `Y_val_for_comp` should both be NumPy arrays of shape `(1, m_samples)`.

```python
# (Ensure numpy is imported: import numpy as np)
# (Assume Y_pred_val_nn and Y_val_for_comp are available from Lesson 7)
# import numpy as np # Typically at the start of your script/notebook

# For demonstration, let's create dummy predictions and labels if not available
if 'Y_pred_val_nn' not in locals() or 'Y_val_for_comp' not in locals() or \
   Y_pred_val_nn.size == 0 or Y_val_for_comp.size == 0:
    print("--- Y_pred_val_nn or Y_val_for_comp not found. Using dummy data for metrics. ---")
    # Example: 10 samples, Y_val_for_comp shape (1,10)
    Y_val_for_comp = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 0]])
    Y_pred_val_nn  = np.array([[1, 1, 0, 1, 0, 0, 1, 1, 0, 1]]) # Some correct, some FP, some FN
else:
    print("Using Y_pred_val_nn and Y_val_for_comp from previous lesson.")

# Ensure they are 1D arrays or (1,m) for easier boolean indexing if needed
# For direct comparison, ensure they are flattened or consistently shaped.
# Our current shapes are (1, m_samples), which is fine for direct comparison.

if Y_pred_val_nn.shape == Y_val_for_comp.shape and Y_pred_val_nn.size > 0:
    TP = np.sum((Y_pred_val_nn == 1) & (Y_val_for_comp == 1))
    TN = np.sum((Y_pred_val_nn == 0) & (Y_val_for_comp == 0))
    FP = np.sum((Y_pred_val_nn == 1) & (Y_val_for_comp == 0))
    FN = np.sum((Y_pred_val_nn == 0) & (Y_val_for_comp == 1))

    print("\n--- Confusion Matrix Components (Validation Set) ---")
    print(f"True Positives (TP): {TP} (Predicted Survived, Actually Survived)")
    print(f"True Negatives (TN): {TN} (Predicted Died, Actually Died)")
    print(f"False Positives (FP): {FP} (Predicted Survived, Actually Died) - Type I Error")
    print(f"False Negatives (FN): {FN} (Predicted Died, Actually Survived) - Type II Error")

    # Verify total predictions
    total_predictions = TP + TN + FP + FN
    print(f"Total predictions in validation set: {total_predictions} (should match Y_val_for_comp.size: {Y_val_for_comp.size})")

    # Display as a simple matrix
    confusion_matrix_scratch = np.array([[TN, FP], [FN, TP]])
    print("\nConfusion Matrix (Scratch):")
    print("             Predicted Died | Predicted Survived")
    print(f"Actual Died:    {TN:^14} | {FP:^17}")
    print(f"Actual Survived:  {FN:^14} | {TP:^17}")

else:
    print("\nCannot calculate confusion matrix: Y_pred_val_nn or Y_val_for_comp have issues.")
    TP, TN, FP, FN = 0,0,0,0 # Default if calculation fails
```

Scikit-learn has a convenient sklearn.metrics.confusion_matrix(y_true, y_pred) function for this too.

b) Precision
Precision answers the question: "Of all passengers we predicted would survive, how many actually survived?"
It's a measure of a classifier's exactness. Low precision indicates a high number of False Positives.

$$ \text{Precision} = \frac{TP}{TP + FP} $$

```python
# (Continuing from above, assuming TP, FP are calculated)
if (TP + FP) > 0:
    precision = TP / (TP + FP)
    print(f"\nPrecision: {precision:.4f}")
    print("Interpretation: Of those predicted to survive, this fraction actually did.")
else:
    precision = 0.0 # Or undefined, handle as appropriate (e.g., if no positive predictions made)
    print("\nPrecision: Not calculable (no positive predictions made or TP+FP is zero).")

```

c) Recall (Sensitivity or True Positive Rate - TPR)
Recall answers the question: "Of all the passengers who actually survived, how many did our model correctly identify?"
It's a measure of a classifier's completeness. Low recall indicates a high number of False Negatives.

$$ \text{Recall} = \frac{TP}{TP + FN} $$

```python
# (Continuing from above, assuming TP, FN are calculated)
if (TP + FN) > 0:
    recall = TP / (TP + FN)
    print(f"\nRecall (Sensitivity): {recall:.4f}")
    print("Interpretation: Of those who actually survived, this fraction was correctly identified.")
else:
    recall = 0.0 # Or undefined, handle as appropriate (e.g., if no actual positive instances)
    print("\nRecall: Not calculable (no actual positive instances or TP+FN is zero).")
```

d) F1-Score
The F1-Score is the harmonic mean of Precision and Recall. It tries to find a balance between them. It's useful when you want good performance on both metrics. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.

$$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

```python
# (Continuing from above, assuming precision and recall are calculated)
if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"\nF1-Score: {f1_score:.4f}")
    print("Interpretation: A combined measure of precision and recall.")
else:
    f1_score = 0.0
    print("\nF1-Score: Not calculable (precision + recall is zero).")

# Also, let's recalculate accuracy using TP, TN, FP, FN to verify
if total_predictions > 0 :
    accuracy_from_cm = (TP + TN) / total_predictions
    print(f"\nAccuracy (calculated from CM components): {accuracy_from_cm:.4f}")
    if 'val_accuracy_nn' in locals(): # If available from Lesson 7
        print(f"(Should be close to previous val_accuracy_nn: {val_accuracy_nn/100:.4f})")

```

Discussion:

* Trade-off: Often, there's a trade-off between precision and recall. Improving one might worsen the other (this depends on the classification threshold).
* Which metric is more important? It depends on the problem:
    * Spam detection: High precision is important (you don't want to classify important emails as spam - minimize FP).
    * Medical diagnosis for a serious disease: High recall is important (you want to identify as many actual patients as possible, even if it means some healthy people get a false alarm - minimize FN). For the Titanic, what might be more critical? Missing a survivor (FN) or wrongly predicting someone survived who didn't (FP)? This is open to interpretation and depends on the application's hypothetical goal.

---
*(Continuing Lesson 8: Evaluating and Improving the Neural Network)*

### 2. Understanding and Diagnosing Overfitting and Underfitting (Conceptual - 20 min) 📉📈

After evaluating our model, we need to understand if its performance is optimal or if it's suffering from common issues like overfitting or underfitting.

* **Underfitting (High Bias):**
    * **What it is:** The model is too simple to capture the underlying patterns in the training data. It performs poorly on both the training set and the validation/test set.
    * **Symptoms:**
        * High training error (low training accuracy, high training cost).
        * High validation/test error (low validation accuracy, high validation cost).
        * The learning curve for the cost might flatten out at a high value quickly.
    * **Possible Causes:**
        * Model is too simple (e.g., too few hidden units/layers, linear model for non-linear data).
        * Not enough training iterations.
        * Features don't have enough predictive power.
    * **What we saw:** If our from-scratch NN had very low accuracy on both training and validation sets, it might be underfitting.

* **Overfitting (High Variance):**
    * **What it is:** The model learns the training data *too well*, including its noise and specific details that don't generalize to new, unseen data. It performs very well on the training set but poorly on the validation/test set.
    * **Symptoms:**
        * Low training error (high training accuracy, low training cost).
        * Significantly higher validation/test error (much lower validation accuracy, higher validation cost).
        * A large gap between training accuracy and validation accuracy.
        * The training cost might continue to decrease while the validation cost starts to increase or plateaus.
    * **Possible Causes:**
        * Model is too complex (e.g., too many hidden units/layers for the amount of data).
        * Insufficient training data.
        * Training for too many epochs.
    * **What we saw:** If `train_accuracy_nn` was, for example, 95% but `val_accuracy_nn` was only 75%, that would be a strong sign of overfitting.

* **Good Fit (Just Right):**
    * The model generalizes well from the training data to unseen data.
    * **Symptoms:**
        * Low training error.
        * Validation/test error is close to the training error (and also low).
        * The validation cost decreases and then stabilizes, closely following the training cost.

**Diagnosing with Learning Curves:**
* **Cost Curves:** Plotting training cost and validation cost against the number of iterations/epochs.
    * If both are high and flat: Underfitting.
    * If training cost is low and decreasing, but validation cost is significantly higher or starts increasing: Overfitting.
    * If both converge to a low value and are close: Good fit.
* **Accuracy Curves:** Plotting training accuracy and validation accuracy.
    * Similar interpretations regarding the gap and overall performance.

---

### 3. Techniques to Improve Model Performance / Combat Overfitting (Conceptual - 25 min)

If our model is underfitting or overfitting, here are some strategies:

**If Underfitting (Model too simple / High Bias):**
1.  **Increase Model Complexity:**
    * Add more hidden layers (make the network deeper).
    * Increase the number of hidden units in existing layers.
    * Choose a more complex model architecture.
2.  **Add More Features / Better Feature Engineering:** The current features might not be informative enough.
3.  **Train Longer:** Ensure the model has had enough iterations to learn (if the cost is still steadily decreasing).
4.  **Decrease Regularization (if applied too strongly):** Regularization (discussed next) can sometimes lead to underfitting if overdone.
5.  **Try a Different Optimization Algorithm:** Though less common for basic underfitting.

**If Overfitting (Model too complex for data / High Variance):**
1.  **Get More Training Data:** More data is often the best way to help a model generalize better. This is not always feasible.
2.  **Reduce Model Complexity:**
    * Decrease the number of hidden units or layers.
    * Use a simpler model.
3.  **Early Stopping:** Monitor the validation error during training and stop training when the validation error starts to increase, even if the training error is still decreasing.
4.  **Feature Selection:** Remove less important features that might be contributing to noise.
5.  **Regularization:** This is a very common and effective technique to reduce overfitting by adding a penalty to the cost function for large weights.
    * **a) L2 Regularization (Weight Decay):**
        * **Concept:** Adds a term to the cost function proportional to the sum of the squares of the weights:
            $$J_{regularized} = J_{original} + \frac{\lambda}{2m} \sum_{l} ||W^{[l]}||^2_F$$
            where $\lambda$ (lambda) is the regularization hyperparameter (controls the strength of the penalty), $m$ is the number of training examples, and $||W^{[l]}||^2_F$ is the squared Frobenius norm of the weight matrix for layer $l$.
        * **Effect:** It discourages large weights, leading to a "simpler" model where individual features have less extreme influence. The decision boundary becomes smoother.
        * **Implementation:** The backward propagation step for weights also gets modified:
            $dW^{[l]}_{regularized} = dW^{[l]}_{original} + \frac{\lambda}{m} W^{[l]}$
    * **b) Dropout:**
        * **Concept:** During each training iteration, randomly "drop out" (temporarily remove) a fraction of neurons in the hidden layers (along with their connections). Each neuron is dropped with some probability $p$ (e.g., $p=0.5$).
        * **Effect:** Neurons cannot rely too much on any specific other neurons, as they might be dropped out. This forces the network to learn more robust and redundant representations. It's like training many different thinned networks simultaneously.
        * **Implementation:** During forward propagation, activations of dropped neurons are set to zero. During backpropagation, the same mask is applied. At test time, dropout is typically turned off, and the weights are scaled by the keep probability (1-p) to compensate for the larger number of active neurons.

---

### 4. Hyperparameter Tuning Strategies (Conceptual - 15 min) 🎛️

Many choices we've made are **hyperparameters** – settings for the learning algorithm itself, not learned from the data. Examples:
* Learning rate ($\alpha$)
* Number of hidden units ($n_h$)
* Number of hidden layers
* Number of iterations/epochs
* Activation functions (though often chosen based on common practice)
* Regularization parameter ($\lambda$)
* Dropout keep probability

Finding good hyperparameters is crucial for model performance and can be an iterative process:

1.  **Identify Key Hyperparameters:** Start with the most impactful ones (e.g., learning rate, number of hidden units).
2.  **Choose a Search Strategy:**
    * **Manual Tuning:** Use your intuition and experience, try different values, observe the impact on the validation set performance (cost, accuracy). This is common for learning and smaller projects.
    * **Grid Search:** Define a grid of possible values for each hyperparameter. Train and evaluate the model for every combination in the grid. Computationally expensive but systematic.
    * **Random Search:** Define a range (or distribution) for each hyperparameter. Randomly sample combinations from these ranges. Often more efficient than grid search, especially when some hyperparameters are more important than others.
    * **More Advanced Methods:** Bayesian optimization, evolutionary algorithms (beyond the scope here).
3.  **Use a Validation Set:** **Crucially**, tune hyperparameters based on their performance on a separate validation set, *not* the test set. The test set is reserved for the final, unbiased evaluation of the *chosen* model with its *final* hyperparameters.
4.  **Iterate:** Adjust hyperparameters based on results, re-train, re-evaluate on the validation set. Look at learning curves to guide your choices.

---

### Wrap-up & Next Steps for Lesson 8:

* **Recap:** We've explored essential evaluation metrics beyond accuracy (Confusion Matrix, Precision, Recall, F1-Score). We've discussed how to identify overfitting and underfitting by comparing training and validation performance and by examining learning curves. We've also conceptually introduced powerful techniques to combat overfitting, like L2 regularization and dropout, and discussed strategies for hyperparameter tuning.
* **Looking Ahead:** We now have a solid foundation in building and evaluating a simple neural network from scratch. Future steps could involve:
    * Actually implementing regularization or dropout in our from-scratch model.
    * Experimenting with different network architectures (more layers, more units).
    * Transitioning to using a deep learning framework like TensorFlow/Keras or PyTorch, which handles many of the low-level details (like backpropagation and optimization) efficiently and allows for building much more complex models with ease.
    * Applying our knowledge to other datasets or more complex problems.

This concludes our journey of building a neural network from scratch for the Titanic dataset!
