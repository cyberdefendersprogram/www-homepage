---
layout: workshop
order: 7
slug: putting-a-neural-network-together
title: Putthing A Neural Network Together
module_category: Machine Learning
workshop_name: zero-to-one-datascience-to-machinelearning
date: 2025-05-23
---
## Lesson 7: Neural Network from Scratch - Part 2 (Cost, Backward Propagation & Training Loop) 🧠⚙️📉

**Objective:**
* Implement the cost function (Binary Cross-Entropy) for our 2-layer neural network.
* Understand the principles of backward propagation and derive/state the necessary gradient formulas.
* Implement the backward propagation steps to calculate gradients for all parameters.
* Implement the parameter update rule (gradient descent).
* Combine all parts into a complete training loop for the neural network.
* Train and evaluate the neural network.

---

### Recap of Lesson 6:
* Defined a 2-layer neural network architecture (Input -> Hidden (ReLU) -> Output (Sigmoid)).
* Implemented `initialize_parameters_nn` to set up $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$.
* Implemented `forward_propagation_nn` which computes $A^{[2]}$ (the prediction) and a `cache` containing intermediate values ($Z^{[1]}, A^{[1]}, Z^{[2]}, A^{[2]}$).
* Remembered the importance of input data shape for NN functions: `(n_features, m_samples)`.

---

### 1. Computing the Cost Function (Practical - 15 min) 💸

After forward propagation, we get $A^{[2]}$, which is a vector of predicted probabilities (shape `(1, m_samples)` because $n_y=1$). We need to compare these predictions to the true labels $Y$ (shape `(1, m_samples)`) to see how well our network is doing.

The **cost function** (or loss function) quantifies this "error." For binary classification, we use the **Binary Cross-Entropy** loss, which is the same as we used for Logistic Regression, but now applied to the output of our neural network $A^{[2]}$.

The formula for $m$ training examples is:
$$J = -\frac{1}{m} \sum_{i=1}^{m} (y^{(i)}\log(a^{[2](i)}) + (1-y^{(i)})\log(1-a^{[2](i)}))$$

Where:
* $m$ is the number of training examples.
* $y^{(i)}$ is the true label for the $i$-th example.
* $a^{[2](i)}$ is the predicted probability for the $i$-th example (from $A^{[2]}$).

```python
# (Ensure numpy is imported: import numpy as np)
# (Ensure X_train_np, y_train_np, params_example, A2_example might be available from previous lessons for context)
# import numpy as np # Already imported

def compute_cost_nn(A2, Y):
    """
    Computes the cross-entropy cost given predictions A2 and true labels Y.

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, m_samples) (predictions)
    Y -- "true" labels vector of shape (1, m_samples) (e.g., 0 if non-survived, 1 if survived)

    Returns:
    cost -- cross-entropy cost (scalar)
    """
    m = Y.shape[1] # Number of examples (Y has shape (1, m_samples))

    # Add a small epsilon for numerical stability to avoid log(0)
    epsilon = 1e-15

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2 + epsilon), Y) + np.multiply(np.log(1 - A2 + epsilon), (1 - Y))
    cost = - (1/m) * np.sum(logprobs)

    cost = np.squeeze(cost) # To make sure cost is a scalar (e.g. turns [[17]] into 17)
    assert(isinstance(cost, float))

    return cost

# Example of using the cost function:
# Assuming A2_example (from Lesson 6 forward prop example) and y_train_nn_input_example are available
if 'A2_example' in locals() and 'y_train_np' in locals() and y_train_np.size > 0:
    # y_train_np from Lesson 4 was (m_samples, 1). For NN cost, we usually want (1, m_samples)
    # if y_train_np.shape[0] > 1 and y_train_np.shape[1] == 1: # if it's (m,1)
    #    y_train_nn_input_example = y_train_np.T # Transpose to (1, m)
    # else: # if it's already (1,m) or a 1D array that needs reshaping
    #    y_train_nn_input_example = y_train_np.reshape(1, -1)

    # Let's assume y_train_np from previous lesson is (m_samples, 1)
    # For consistency with A2 shape (1, m_samples), Y should also be (1, m_samples)
    # If y_train_np is (m_samples,1), then y_train_np.T is (1,m_samples)
    if y_train_np.shape[0] > 1 and y_train_np.shape[1] == 1: # Check if it's (m_samples, 1)
        y_for_cost_example = y_train_np.T
    elif y_train_np.ndim == 1: # if it is a 1D array
        y_for_cost_example = y_train_np.reshape(1, -1)
    else: # Assuming it's already (1, m_samples)
        y_for_cost_example = y_train_np


    # Ensure A2_example and y_for_cost_example have compatible shapes for the test
    # A2_example from Lesson 6 had shape (1, num_samples_in_X_train_np)
    # y_for_cost_example should match this.
    # If X_train_np was used for A2_example, then y_for_cost_example should use y_train_np.
    if A2_example.shape == y_for_cost_example.shape:
        print("\n--- Example NN Cost Computation ---")
        cost_example_nn = compute_cost_nn(A2_example, y_for_cost_example)
        print(f"Example Cost: {cost_example_nn}")
    else:
        print("\n--- Example NN Cost Computation ---")
        print(f"Shape mismatch for cost computation: A2_example shape {A2_example.shape}, y_for_cost_example shape {y_for_cost_example.shape}")
        print("Cannot compute example cost. Ensure y_for_cost_example corresponds to the same samples as A2_example.")
        # Fallback: Compute cost with dummy data if shapes mismatch, to show function usage
        dummy_A2 = np.random.rand(1, 5) # e.g. 1 output neuron, 5 samples
        dummy_Y = np.random.randint(0, 2, (1, 5))
        dummy_cost = compute_cost_nn(dummy_A2, dummy_Y)
        print(f"Dummy Cost example with shapes A2:{dummy_A2.shape}, Y:{dummy_Y.shape} = {dummy_cost}")

else:
    print("\n--- Example NN Cost Computation ---")
    print("A2_example or y_train_np not available from previous steps. Cannot run cost example.")
    # Fallback: Compute cost with dummy data to show function usage
    dummy_A2 = np.random.rand(1, 5) # e.g. 1 output neuron, 5 samples
    dummy_Y = np.random.randint(0, 2, (1, 5))
    dummy_cost = compute_cost_nn(dummy_A2, dummy_Y)
    print(f"Dummy Cost example = {dummy_cost}")
```

### 2. Backward Propagation: The Intuition (Conceptual - 20 min) ⏪

Backward propagation (or "backprop") is the algorithm used to calculate the gradients of the cost function $J$ with respect to each parameter ($W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$) in the network. These gradients tell us how much a small change in each parameter would affect the cost.

* **Why do we need gradients?** To perform gradient descent. We want to update our parameters in the direction that *minimizes* the cost. Gradients point in the direction of the steepest *ascent*, so we move in the opposite direction.
* **The Chain Rule:** Backprop relies heavily on the chain rule from calculus. Because the cost $J$ is an indirect function of, say, $W^{[1]}$ (i.e., $J$ depends on $A^{[2]}$, which depends on $Z^{[2]}$, which depends on $A^{[1]}$, which depends on $Z^{[1]}$, which finally depends on $W^{[1]}$), we use the chain rule to find $\frac{\partial J}{\partial W^{[1]}}$.

**Flow of Backpropagation:**
It's called "backward" because we start by calculating the gradient of the cost with respect to the output of the network ($A^{[2]}$) and then propagate these gradients backward, layer by layer, until we reach the earliest layers.

**Key steps for our 2-layer network:**

1.  **Calculate $dZ^{[2]}$:** The gradient of the cost $J$ with respect to $Z^{[2]}$ (the linear output of the final layer).
    * $dZ^{[2]} = A^{[2]} - Y$ (This is a remarkably simple and elegant result for sigmoid output and binary cross-entropy loss).
2.  **Calculate $dW^{[2]}$ and $db^{[2]}$:** Gradients for the parameters of the output layer.
    * $dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$
    * $db^{[2]} = \frac{1}{m} \sum_{\text{examples}} dZ^{[2]}$ (summing $dZ^{[2]}$ over all examples, effectively taking the mean across the batch dimension if $dZ^{[2]}$ is a row vector per sample, or summing all elements if it's a column vector with each element corresponding to a sample and then dividing by m. For vector notation $(1,m)$ or $(m,1)$ it's typically `np.sum(dZ2, axis=1, keepdims=True)` or `np.sum(dZ2)` then divide by m). *Correct vector form: `(1/m) * np.sum(dZ2, axis=1, keepdims=True)` if $dZ^{[2]}$ is `(num_output_units, m_samples)` which is `(1, m_samples)` here.*
3.  **Calculate $dA^{[1]}$:** The gradient of the cost $J$ with respect to $A^{[1]}$ (the activation output of the hidden layer).
    * $dA^{[1]} = W^{[2]T} dZ^{[2]}$
4.  **Calculate $dZ^{[1]}$:** The gradient of the cost $J$ with respect to $Z^{[1]}$ (the linear output of the hidden layer). This involves the derivative of the hidden layer's activation function (ReLU in our case).
    * $dZ^{[1]} = dA^{[1]} * g^{[1]'}(Z^{[1]})$ where $g^{[1]'}(Z^{[1]})$ is the derivative of ReLU applied to $Z^{[1]}$.
        * Derivative of ReLU: $g'(z) = 1$ if $z > 0$, and $0$ if $z \le 0$.
5.  **Calculate $dW^{[1]}$ and $db^{[1]}$:** Gradients for the parameters of the hidden layer.
    * $dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$ (where X is the input to the network, shape `(n_x, m_samples)`)
    * $db^{[1]} = \frac{1}{m} \sum_{\text{examples}} dZ^{[1]}$ *(Correct vector form: `(1/m) * np.sum(dZ1, axis=1, keepdims=True)` if $dZ^{[1]}$ is `(num_hidden_units, m_samples)`)*

*(The actual implementation of the `backward_propagation_nn` function will be in the next part of this lesson).*

---
*(Continuing Lesson 7: Neural Network from Scratch - Part 2)*

### 3. Implementing Backward Propagation (Practical - 30 min) ⚙️⏪

Now, let's translate the formulas from the conceptual overview into a Python function. We'll need the `cache` from forward propagation, which contains $Z^{[1]}, A^{[1]}, Z^{[2]}, A^{[2]}$. We also need the input $X$ and true labels $Y$.

Remember our network structure: Input -> [LINEAR -> RELU] (Layer 1) -> [LINEAR -> SIGMOID] (Layer 2) -> Output.
The input $X$ for the NN functions is `(n_x, m_samples)`.
The output $Y$ is `(1, m_samples)`.

```python
# (Ensure numpy is imported: import numpy as np)
# (Activation functions relu, sigmoid are assumed to be defined from Lesson 6)
# (X_train_nn_input_example, y_for_cost_example, params_example, cache_example from L6/L7-Part1 for context)
# import numpy as np

def backward_propagation_nn(parameters, cache, X_input_nn, Y_labels_nn):
    """
    Implements the backward propagation for the 2-layer neural network.

    Arguments:
    parameters -- python dictionary containing our parameters (W1, b1, W2, b2)
    cache -- python dictionary containing Z1, A1, Z2, A2 (from forward_propagation_nn)
    X_input_nn -- input data of shape (n_x, m_samples)
    Y_labels_nn -- "true" labels vector of shape (1, m_samples)

    Returns:
    grads -- python dictionary containing gradients with respect to different parameters:
             dW1, db1, dW2, db2
    """
    m = X_input_nn.shape[1] # Number of examples

    # First, retrieve W1 and W2 from the parameters dictionary
    # W1 = parameters["W1"] # Not actually needed directly in this function, but W2 is.
    W2 = parameters["W2"] # Shape (n_y, n_h)

    # Retrieve A1 and A2 from the cache dictionary
    A1 = cache["A1"] # Shape (n_h, m_samples)
    A2 = cache["A2"] # Shape (n_y, m_samples) = (1, m_samples)
    Z1 = cache["Z1"] # Shape (n_h, m_samples)
    # Z2 = cache["Z2"] # Not directly needed for gradient calculation here if dZ2 is A2-Y

    # Backward propagation: calculate dW1, db1, dW2, db2.
    # Step 1: Output Layer (Sigmoid)
    # dZ2 = A2 - Y (Derivative of cost w.r.t Z2 for sigmoid output and binary cross-entropy loss)
    dZ2 = A2 - Y_labels_nn  # Shape (1, m_samples)

    # Step 2: Gradients for W2 and b2
    # dW2 = (1/m) * dZ2 . A1_transpose
    dW2 = (1/m) * np.dot(dZ2, A1.T) # A1.T has shape (m_samples, n_h) -> dW2 shape (1, n_h) == (n_y, n_h)
    # db2 = (1/m) * sum(dZ2 along axis=1, keepdims=True)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) # Shape (1, 1) == (n_y, 1)

    # Step 3: Gradient w.r.t A1 (dA1)
    # dA1 = W2_transpose . dZ2
    dA1 = np.dot(W2.T, dZ2) # W2.T has shape (n_h, 1), dZ2 has shape (1, m_samples) -> dA1 shape (n_h, m_samples)

    # Step 4: Gradient w.r.t Z1 (dZ1)
    # dZ1 = dA1 * g'(Z1) where g is ReLU
    # Derivative of ReLU: 1 if Z1 > 0, else 0.
    dRelu_Z1 = np.where(Z1 > 0, 1, 0) # Or (Z1 > 0).astype(int)
    dZ1 = dA1 * dRelu_Z1 # Element-wise product, shape (n_h, m_samples)

    # Step 5: Gradients for W1 and b1
    # dW1 = (1/m) * dZ1 . X_input_nn_transpose
    dW1 = (1/m) * np.dot(dZ1, X_input_nn.T) # X_input_nn.T has shape (m_samples, n_x) -> dW1 shape (n_h, n_x)
    # db1 = (1/m) * sum(dZ1 along axis=1, keepdims=True)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True) # Shape (n_h, 1)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

# Example of running backward propagation:
if 'params_example' in locals() and params_example['W1'] is not None and \
   'cache_example' in locals() and \
   'X_train_nn_input_example' in locals() and X_train_nn_input_example.size > 0 and \
   'y_for_cost_example' in locals() and y_for_cost_example.size > 0:

    print("\n--- Example NN Backward Propagation ---")
    # Ensure y_for_cost_example is correctly shaped (1, m_samples) as required by Y_labels_nn
    # y_for_cost_example was shaped for the cost function in the previous example.
    
    # Assuming X_train_nn_input_example and y_for_cost_example are correctly defined and shaped:
    # X_train_nn_input_example should be (n_x, m_samples)
    # y_for_cost_example should be (1, m_samples)
    
    grads_example = backward_propagation_nn(params_example, cache_example, X_train_nn_input_example, y_for_cost_example)

    print(f"dW1 shape: {grads_example['dW1'].shape}")
    print(f"db1 shape: {grads_example['db1'].shape}")
    print(f"dW2 shape: {grads_example['dW2'].shape}")
    print(f"db2 shape: {grads_example['db2'].shape}")
    print(f"First 5 elements of dW1 (flattened): {grads_example['dW1'].flatten()[:5]}")
else:
    print("\nCannot run backward propagation example: Missing params, cache, X input, or Y labels.")
```

4. Updating Parameters (Practical - 10 min) 🛠️
Once we have the gradients, we update the parameters using gradient descent:
$W = W - \alpha \cdot dW$
$b = b - \alpha \cdot db$

```python
def update_parameters_nn(parameters, grads, learning_rate):
    """
    Updates parameters using the gradient descent update rule.

    Arguments:
    parameters -- python dictionary containing your parameters (W1, b1, W2, b2)
    grads -- python dictionary containing gradients (dW1, db1, dW2, db2)
    learning_rate -- the learning rate, alpha.

    Returns:
    parameters -- python dictionary containing updated parameters
    """
    # Retrieve parameters from the dictionary
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve gradients from the grads dictionary
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
```

5. Building the Neural Network Model (Training Loop) (Practical - 25 min) 🔄
Now we integrate all the pieces: initialize, loop for num_iterations (epochs): forward prop, compute cost, backward prop, update parameters.


```python
def nn_model(X_input_nn, Y_labels_nn, n_h, num_iterations=10000, learning_rate=0.0075, print_cost_every=1000):
    """
    Arguments:
    X_input_nn -- input data of shape (n_x, m_samples)
    Y_labels_nn -- true "label" vector of shape (1, m_samples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    learning_rate -- learning rate for the gradient descent update rule
    print_cost_every -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- list of costs recorded during training
    """
    if X_input_nn.size == 0 or Y_labels_nn.size == 0:
        print("Error: Input data or labels are empty. Cannot train model.")
        return None, []

    np.random.seed(3) # for consistency
    n_x = X_input_nn.shape[0] # number of features
    n_y = Y_labels_nn.shape[0]  # should be 1 for binary classification

    # Initialize parameters
    parameters = initialize_parameters_nn(n_x, n_h, n_y)
    costs = []

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: X, parameters. Outputs: A2, cache.
        A2, cache = forward_propagation_nn(X_input_nn, parameters)

        # Cost function. Inputs: A2, Y. Outputs: cost.
        cost = compute_cost_nn(A2, Y_labels_nn)

        # Backpropagation. Inputs: parameters, cache, X, Y. Outputs: grads.
        grads = backward_propagation_nn(parameters, cache, X_input_nn, Y_labels_nn)

        # Gradient descent parameter update. Inputs: parameters, grads, learning_rate. Outputs: parameters.
        parameters = update_parameters_nn(parameters, grads, learning_rate)

        # Print the cost every print_cost_every iterations
        if print_cost_every > 0 and i % print_cost_every == 0:
            costs.append(cost)
            print (f"Cost after iteration {i}: {cost:.6f}")
        if print_cost_every > 0 and i == num_iterations - 1: # Print final cost
             if i % print_cost_every != 0: # Avoid double printing if last iter is a print_cost_every iter
                costs.append(cost)
                print (f"Cost after iteration {i}: {cost:.6f}")


    return parameters, costs

# Prepare data for NN model (transpose X, ensure Y is (1,m))
# Assuming X_train_np and y_train_np are from Lesson 4 (after pd.to_numpy())
final_trained_nn_params = None
costs_history_nn = []

if 'X_train_np' in locals() and X_train_np.size > 0 and \
   'y_train_np' in locals() and y_train_np.size > 0:

    X_train_for_nn = X_train_np.T # Shape (n_x, m_samples)
    
    # Ensure Y_train_for_nn is (1, m_samples)
    if y_train_np.shape[0] > 1 and y_train_np.shape[1] == 1: # if it's (m_samples,1)
        Y_train_for_nn = y_train_np.T
    elif y_train_np.ndim == 1: # if it's a 1D array
        Y_train_for_nn = y_train_np.reshape(1, -1)
    else: # Assuming it's already (1, m_samples)
        Y_train_for_nn = y_train_np

    if X_train_for_nn.shape[1] == Y_train_for_nn.shape[1]: # Check if number of samples match
        print(f"\n--- Training Neural Network Model (1 Hidden Layer) ---")
        print(f"X_train_for_nn shape: {X_train_for_nn.shape}")
        print(f"Y_train_for_nn shape: {Y_train_for_nn.shape}")
        
        n_h_neurons = 4 # Number of neurons in the hidden layer
        final_trained_nn_params, costs_history_nn = nn_model(
            X_train_for_nn, Y_train_for_nn,
            n_h=n_h_neurons,
            num_iterations=10000, # Can be tuned
            learning_rate=0.0075, # Can be tuned
            print_cost_every=1000
        )

        if final_trained_nn_params:
            print("\n--- NN Training Complete ---")
            # Plot the learning curve (cost vs. iterations)
            if costs_history_nn:
                # import matplotlib.pyplot as plt # Should be imported at the top of the notebook/script
                plt.figure(figsize=(10,6))
                plot_iterations_nn = np.arange(0, len(costs_history_nn)) * 1000 # Adjust if print_cost_every changes
                plt.plot(plot_iterations_nn, costs_history_nn)
                plt.ylabel('Cost')
                plt.xlabel('Iterations')
                plt.title('Learning Curve for Neural Network (From Scratch)')
                plt.show() # Ensure plot displays
        else:
            print("NN Training did not complete successfully or returned no parameters.")
    else:
        print(f"Sample number mismatch: X_train_for_nn has {X_train_for_nn.shape[1]} samples, Y_train_for_nn has {Y_train_for_nn.shape[1]} samples.")
else:
    print("\nCannot train NN model: X_train_np or y_train_np is not defined or empty.")

```

6. Predictions with the Neural Network (Practical - 10 min) 🔮
Once the model is trained, we can use the learned parameters to make predictions.

```python
def predict_nn(X_input_nn, parameters, threshold=0.5):
    """
    Using the learned parameters, predicts a class for each example in X.

    Arguments:
    X_input_nn -- input data of size (n_x, m_samples)
    parameters -- python dictionary containing parameters W1, b1, W2, b2
    threshold -- probability threshold to classify as 1

    Returns:
    predictions -- vector of predictions of our model (0 / 1), shape (1, m_samples)
    """
    if parameters is None or parameters.get("W1") is None: # Check if parameters are valid
        print("Error: Model parameters are not available for prediction.")
        return np.array([])


    # Computes probabilities using forward propagation, and classifies to 0/1 using threshold.
    A2, cache = forward_propagation_nn(X_input_nn, parameters)
    predictions = (A2 >= threshold).astype(int)

    return predictions

# Make predictions on training and validation sets
if final_trained_nn_params and \
   'X_train_np' in locals() and X_train_np.size > 0 and \
   'X_val_np' in locals() and X_val_np.size > 0 and \
   'y_train_np' in locals() and y_train_np.size > 0 and \
   'y_val_np' in locals() and y_val_np.size > 0:

    X_train_for_pred = X_train_np.T
    X_val_for_pred = X_val_np.T
    
    # Y labels should be (1, m_samples) for comparison with predictions
    if y_train_np.shape[0] > 1 and y_train_np.shape[1] == 1: Y_train_for_comp = y_train_np.T
    elif y_train_np.ndim == 1: Y_train_for_comp = y_train_np.reshape(1,-1)
    else: Y_train_for_comp = y_train_np
        
    if y_val_np.shape[0] > 1 and y_val_np.shape[1] == 1: Y_val_for_comp = y_val_np.T
    elif y_val_np.ndim == 1: Y_val_for_comp = y_val_np.reshape(1,-1)
    else: Y_val_for_comp = y_val_np

    Y_pred_train_nn = predict_nn(X_train_for_pred, final_trained_nn_params)
    Y_pred_val_nn = predict_nn(X_val_for_pred, final_trained_nn_params)

    if Y_pred_train_nn.size > 0 and Y_pred_val_nn.size > 0 and \
       Y_pred_train_nn.shape == Y_train_for_comp.shape and \
       Y_pred_val_nn.shape == Y_val_for_comp.shape:
        
        train_accuracy_nn = np.mean(Y_pred_train_nn == Y_train_for_comp) * 100
        val_accuracy_nn = np.mean(Y_pred_val_nn == Y_val_for_comp) * 100

        print(f"\n--- Neural Network (1 Hidden Layer) Performance ---")
        print(f"Training Accuracy: {train_accuracy_nn:.2f}%")
        print(f"Validation Accuracy: {val_accuracy_nn:.2f}%")
    else:
        print("Prediction arrays are empty or shape mismatch with true labels. Cannot calculate accuracy.")
        if Y_pred_train_nn.size > 0 : print(f"Shape Y_pred_train_nn: {Y_pred_train_nn.shape}, Shape Y_train_for_comp: {Y_train_for_comp.shape}")
        if Y_pred_val_nn.size > 0 : print(f"Shape Y_pred_val_nn: {Y_pred_val_nn.shape}, Shape Y_val_for_comp: {Y_val_for_comp.shape}")
else:
    print("\nCannot make NN predictions: Model was not trained successfully or necessary data is missing.")
```

---
### 7. Discussion & Next Steps (Conceptual - 10 min)

* **Performance:** How does this simple neural network compare to the Logistic Regression model from scratch?
* **Hyperparameters:**
    * `n_h`: Number of hidden units. Too few might underfit, too many might overfit or be slow to train.
    * `num_iterations`: How many training steps.
    * `learning_rate`: Crucial for convergence.
    Experiment with these!
* **Overfitting/Underfitting:** If training accuracy is high but validation accuracy is much lower, the model might be overfitting. If both are low, it might be underfitting (model is too simple or not trained enough).
* **Improvements (Conceptual):**
    * **Deeper Networks:** Add more hidden layers.
    * **Different Activation Functions:** Experiment.
    * **Regularization:** L2 regularization or Dropout to combat overfitting.
    * **More Advanced Optimizers:** Adam, RMSprop.
    * **Batch Normalization:** Can help stabilize and speed up training.
    * **Better Weight Initialization:** He initialization for ReLU layers.

---

### Wrap-up & Teaser for Lesson 8:

* **Recap:** We've built our first neural network from scratch! This involved initializing parameters, implementing forward propagation, computing the cost, implementing backward propagation to get gradients, and updating parameters using gradient descent. We also trained and made predictions with our simple 1-hidden-layer network.
* **Teaser for Lesson 8:** In the next lesson, "Evaluating and Improving the Neural Network," we'll dive deeper into:
    * More robust **evaluation metrics** (beyond accuracy, like precision, recall, F1-score, confusion matrix).
    * Understanding and diagnosing **overfitting and underfitting** more formally.
    * Introducing techniques like **regularization** (L2, Dropout) conceptually, and perhaps implementing a simple version.
    * Discussing **hyperparameter tuning strategies**.

This sets the stage for building more complex and effective neural networks!
