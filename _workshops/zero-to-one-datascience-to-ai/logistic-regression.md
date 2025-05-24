---
layout: workshop
order: 004
slug: logistic-regression
title: Logistic Regression
module_category: Data Science
workshop_name: zero-to-one-datascience-to-ai
date: 2025-05-23
---
## Lesson 4: Your First Predictive Model - Logistic Regression (FROM SCRATCH) ⚙️🧠

**Objective:**
* Understand the mathematical intuition behind Logistic Regression.
* Implement the core components of Logistic Regression (sigmoid function, cost function, gradient descent) from scratch using NumPy.
* Train the from-scratch Logistic Regression model on our prepared Titanic training data.
* Evaluate its performance on the validation data.

**Important Note:** For this lesson, we are building Logistic Regression from scratch to understand its inner workings. In practice, for most applications, you would use optimized and well-tested implementations from libraries like Scikit-learn.

---

### Recap of Lesson 3:
* Converted categorical features to numerical (one-hot encoding).
* Scaled numerical features (Standardization).
* Split data into `X_train`, `y_train`, `X_val`, `y_val`. *(If running standalone, ensure these are defined, possibly with dummy data if Lesson 3 wasn't fully executed).*

---

### 1. Logistic Regression: Intuition (Conceptual - 15 min)

Logistic Regression is a statistical model used for **binary classification** problems (predicting one of two outcomes). Despite its name, it's a classification algorithm, not a regression algorithm for continuous values.

* **The Problem with Linear Regression for Classification:** If we directly used linear regression ($y = WX + b$) for a binary outcome (0 or 1), the output could be any real number (e.g., <0 or >1), which isn't interpretable as a probability.
* **The Sigmoid (Logistic) Function:** To solve this, Logistic Regression passes the linear output through a sigmoid function:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    Where $z$ is the linear combination: $z = W \cdot X + b$ (often written as $W^T X + b$ or $\theta^T X$).
    * The sigmoid function squashes any real-valued input $z$ into an output between 0 and 1.
    * This output, $\sigma(z)$, can be interpreted as the **probability** of the positive class (e.g., $P(y=1 | X; W, b)$).
* **Hypothesis ($h(X)$):**
    For Logistic Regression, our hypothesis is:
    $$h_{W,b}(X) = \sigma(W \cdot X + b)$$
* **Decision Boundary:**
    We make a prediction based on this probability:
    * If $h_{W,b}(X) \ge 0.5$, predict $y=1$.
    * If $h_{W,b}(X) < 0.5$, predict $y=0$.
    This threshold of 0.5 for the probability corresponds to $z \ge 0$ for predicting $y=1$. The equation $z = W \cdot X + b = 0$ defines the decision boundary, which is linear for Logistic Regression.

---

### 2. Building Blocks (From Scratch with NumPy) (Practical)

We'll use NumPy for efficient array operations. Ensure it's imported:
```python
import numpy as np
import pandas as pd # For creating dummy data if needed
import matplotlib.pyplot as plt # For plotting cost curve later
```
---
*(Continuing Section 2: Building Blocks)*

* **e) Backward Propagation: Compute Gradients**
    This step calculates the derivatives (gradients) of the cost function with respect to $W$ and $b$. These tell us how to update $W$ and $b$ to reduce the cost.
    * $dZ = A - Y$ (Derivative of cost w.r.t $Z$, shape $(m, 1)$)
    * $dW = \frac{1}{m} X^T \cdot dZ$ (Derivative of cost w.r.t $W$, shape $(n_f, 1)$)
    * $db = \frac{1}{m} \sum_{i=1}^{m} (A - Y)^{(i)}$ (Derivative of cost w.r.t $b$, scalar)

    ```python
    def backward_propagation_logistic(X, Y, A):
        """
        Implements backward propagation to calculate gradients.
        Arguments:
        X -- data of size (m_samples, n_features)
        Y -- true "label" vector (0 or 1) of size (m_samples, 1)
        A -- activations (predictions) from forward_propagation, shape (m_samples, 1)
        Returns:
        dW -- gradient of the cost with respect to W, shape (n_features, 1)
        db -- gradient of the cost with respect to b, scalar
        """
        m = X.shape[0] # Number of samples

        dZ = A - Y  # Shape (m, 1)
        dW = (1/m) * np.dot(X.T, dZ) # X.T is (n_f, m), dZ is (m, 1) -> dW is (n_f, 1)
        db = (1/m) * np.sum(dZ)      # Sum over all samples

        return dW, db
    ```

* **f) Optimization: Gradient Descent Update Rule**
    Update the parameters $W$ and $b$ in the opposite direction of their gradients to minimize the cost.
    * $W = W - \alpha \cdot dW$
    * $b = b - \alpha \cdot db$
    Where $\alpha$ is the **learning rate**, a hyperparameter that controls the step size.

    ```python
    def update_parameters_logistic(W, b, dW, db, learning_rate):
        """
        Updates parameters using the gradient descent rule.
        Arguments:
        W, b -- parameters
        dW, db -- gradients
        learning_rate -- alpha
        Returns:
        W, b -- updated parameters
        """
        W = W - learning_rate * dW
        b = b - learning_rate * db
        return W, b
    ```

---

### 3. Training the Logistic Regression Model (Practical - 25 min)

Now, we'll combine these functions into a training loop.

```python
def logistic_regression_model_scratch(X_train, Y_train, num_iterations=2000, learning_rate=0.01, print_cost_every=100):
    """
    Builds and trains the logistic regression model from scratch.
    Arguments:
    X_train -- training data, shape (m_samples, n_features)
    Y_train -- training labels, shape (m_samples, 1)
    num_iterations -- number of iterations for gradient descent
    learning_rate -- learning rate for gradient descent
    print_cost_every -- print cost at this interval
    Returns:
    parameters -- dictionary containing final W and b
    costs -- list of costs recorded during training
    """
    if X_train.size == 0 or Y_train.size == 0:
        print("Error: Training data is empty. Cannot train model.")
        return {"W": None, "b": None}, []
        
    num_features = X_train.shape[1]
    W, b = initialize_parameters_logistic(num_features)
    costs = []

    for i in range(num_iterations):
        # Forward propagation
        A, cost = forward_propagation_logistic(X_train, Y_train, W, b)

        # Backward propagation
        dW, db = backward_propagation_logistic(X_train, Y_train, A)

        # Update parameters
        W, b = update_parameters_logistic(W, b, dW, db, learning_rate)

        # Record and print cost
        if i % print_cost_every == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost:.4f}")
        if i == num_iterations - 1: # Also print final cost
             if i % print_cost_every != 0: # Avoid double printing if last iter is a print_cost_every iter
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost:.4f}")


    parameters = {"W": W, "b": b}
    return parameters, costs

# Train the model (using X_train_np and y_train_np from earlier)
trained_parameters_scratch = {"W": None, "b": None} # Initialize to handle potential errors
costs_history_scratch = []

if 'X_train_np' in locals() and X_train_np.size > 0 and 'y_train_np' in locals() and y_train_np.size > 0:
    print("\n--- Training Logistic Regression Model (From Scratch) ---")
    trained_parameters_scratch, costs_history_scratch = logistic_regression_model_scratch(
        X_train_np, y_train_np,
        num_iterations=5000, # Increased iterations
        learning_rate=0.05,  # Experiment with learning rate
        print_cost_every=500
    )

    if trained_parameters_scratch["W"] is not None:
        print("\n--- Training Complete ---")
        print(f"Final W (head): {trained_parameters_scratch['W'][:5].T if trained_parameters_scratch['W'].size > 0 else 'N/A'}")
        print(f"Final b: {trained_parameters_scratch['b']:.4f}")

        # Plot the learning curve (cost vs. iterations)
        plt.figure(figsize=(10,6))
        # X-axis for plot should match iterations where cost was recorded
        plot_iterations = np.arange(0, len(costs_history_scratch)) * 500
        plt.plot(plot_iterations, costs_history_scratch)
        plt.ylabel('Cost')
        plt.xlabel(f'Iterations')
        plt.title('Learning Curve for Logistic Regression (From Scratch)')
        plt.show() # Ensure plot is displayed
    else:
        print("Training did not complete successfully.")
else:
    print("\nCannot train model: X_train_np or y_train_np is not defined or empty.")
```

4. Making Predictions (Practical - 10 min)
Use the learned $W$ and $b$ to predict on new data.

```python 
def predict_logistic_scratch(X, W, b, threshold=0.5):
    """
    Predicts binary labels (0 or 1) using learned logistic regression parameters.
    Arguments:
    X -- data to predict on, shape (m_samples, n_features)
    W, b -- learned parameters
    threshold -- probability threshold for classifying as 1
    Returns:
    Y_predictions -- numpy array of shape (m_samples, 1) with 0/1 predictions
    """
    if W is None or b is None or X.size == 0:
        print("Error: Model parameters or data missing for prediction.")
        return np.array([]) # Return empty array or handle error appropriately

    m = X.shape[0]
    Y_predictions = np.zeros((m, 1)) # Initialize predictions

    # Compute activations (probabilities)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)

    # Convert probabilities to 0/1 predictions
    Y_predictions = (A >= threshold).astype(int) # Boolean to int

    return Y_predictions

# Make predictions if model was trained
if trained_parameters_scratch["W"] is not None and \
   'X_train_np' in locals() and X_train_np.size > 0 and \
   'X_val_np' in locals() and X_val_np.size > 0 and \
   'y_train_np' in locals() and y_train_np.size > 0 and \
   'y_val_np' in locals() and y_val_np.size > 0:

    W_final_scratch = trained_parameters_scratch["W"]
    b_final_scratch = trained_parameters_scratch["b"]

    Y_pred_train_scratch = predict_logistic_scratch(X_train_np, W_final_scratch, b_final_scratch)
    Y_pred_val_scratch = predict_logistic_scratch(X_val_np, W_final_scratch, b_final_scratch)

    if Y_pred_train_scratch.size > 0 and Y_pred_val_scratch.size > 0:
        # Calculate Accuracy
        # Accuracy = (Number of correct predictions) / (Total number of predictions)
        train_accuracy_scratch = np.mean(Y_pred_train_scratch == y_train_np) * 100
        val_accuracy_scratch = np.mean(Y_pred_val_scratch == y_val_np) * 100

        print(f"\n--- From Scratch Model Performance ---")
        print(f"Training Accuracy: {train_accuracy_scratch:.2f}%")
        print(f"Validation Accuracy: {val_accuracy_scratch:.2f}%")
    else:
        print("Predictions could not be made or resulted in empty arrays.")
else:
    print("\nCannot make predictions: Model was not trained successfully or necessary data (X_train_np, X_val_np, y_train_np, y_val_np) is missing.")

```

### 5. Discussion & Evaluation (Conceptual - 10 min)

* **Accuracy:** How well did our from-scratch model do? Compare training and validation accuracy. A large gap might indicate overfitting (model learned training data too well, doesn't generalize).
* **Learning Rate ($\alpha$):** This is a crucial hyperparameter.
    * If too small: Gradient descent will be very slow to converge.
    * If too large: The cost might oscillate or even diverge (overshoot the minimum).
    Experimenting with different learning rates (e.g., 0.001, 0.01, 0.05, 0.1, 0.3) is common.
* **Number of Iterations:** How many steps of gradient descent? More isn't always better if the learning rate is not well-tuned or if the model starts overfitting. The cost curve helps monitor this.
* **Convergence:** Did the cost consistently decrease and then flatten out? This indicates convergence. If the cost is still decreasing significantly at the end, more iterations might be needed (or a better learning rate).
* **Limitations of this Scratch Implementation:**
    * **Basic Gradient Descent:** We used batch gradient descent (computes gradient over all training examples). For very large datasets, Stochastic Gradient Descent (SGD) or Mini-batch Gradient Descent are more efficient.
    * **No Regularization:** We haven't included regularization (L1 or L2), which can help prevent overfitting by penalizing large weights.
    * **Manual Hyperparameter Tuning:** We manually set the learning rate and number of iterations. More advanced techniques like cross-validation are used for robust hyperparameter tuning.
    * **Numerical Stability:** While we added a small epsilon in the cost function, more sophisticated numerical stability measures might be needed in complex scenarios.
* **Benefit of "From Scratch":** The primary benefit is a deep understanding of the algorithm's mechanics! This knowledge is invaluable when using library implementations or when building more complex models like neural networks. You know what's happening "under the hood."

---

### Wrap-up & Next Steps:

* **Recap:** We've successfully built a Logistic Regression model *entirely from scratch* using NumPy! We implemented the sigmoid function, cost calculation, gradient computation, and the gradient descent update rule. We trained it on our preprocessed Titanic data and evaluated its performance.
* **Teaser for Lesson 5:** Now that we've seen how a simple linear classifier learns, we are ready to explore the core concepts of **Neural Networks**. We'll see how neural networks extend these ideas with layers of neurons and non-linear activation functions to learn much more complex patterns in data.
