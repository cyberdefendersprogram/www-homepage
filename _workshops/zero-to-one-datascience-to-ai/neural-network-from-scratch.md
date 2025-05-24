---
layout: workshop
order: 006
slug: neural-network-from-scratch
title: Neural Network From Scratch
module_category: Machine Learning
workshop_name: zero-to-one-datascience-to-machinelearning
date: 2025-05-23
---
## Lesson 6: Neural Network from Scratch - Part 1 (Forward Propagation) 🚀

**Objective:**
* Define a simple neural network architecture (1 hidden layer).
* Implement functions to initialize weights and biases for this architecture.
* Implement ReLU and Sigmoid activation functions (reusing Sigmoid from Logistic Regression).
* Implement the forward propagation steps for the defined network using NumPy.

---

### Recap of Lesson 5:
* Understood core neural network concepts: neurons, layers, weights, biases, activation functions.
* Conceptually understood forward propagation, loss, backpropagation, and optimization.

---

### 1. Defining the Network Architecture (Conceptual - 10 min) 🏗️

For our first neural network from scratch, we'll build a simple 2-layer network (often referred to as a 1-hidden-layer network):

* **Input Layer:** The number of units ($n_x$) will be determined by the number of features in our preprocessed Titanic data (`X_train_np.shape[1]`).
* **Hidden Layer:** We'll choose a number of units for this layer, say $n_h = 4$ (this is a hyperparameter we can tune). It will use the **ReLU** activation function.
* **Output Layer:** Since this is a binary classification problem (Survived or Died), it will have 1 unit ($n_y = 1$) and use the **Sigmoid** activation function to output a probability.

**Diagram:**
```
X (Input Features) --> [LINEAR -> RELU] (Hidden Layer) --> [LINEAR -> SIGMOID] (Output Layer) --> Prediction (A2)
(n_x units)          (n_h units)                       (n_y unit)
```

**Mathematical Representation:**

For one example $x$:
1.  $Z^{[1]} = W^{[1]} x + b^{[1]}$  (Linear step for hidden layer)
2.  $A^{[1]} = \text{ReLU}(Z^{[1]})$ (Activation for hidden layer)
3.  $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ (Linear step for output layer)
4.  $A^{[2]} = \sigma(Z^{[2]})$ (Activation for output layer - this is our prediction $\hat{y}$)

**Dimensions of Parameters:**
* $W^{[1]}$: `(n_h, n_x)`
* $b^{[1]}$: `(n_h, 1)`
* $W^{[2]}$: `(n_y, n_h)`
* $b^{[2]}$: `(n_y, 1)`

---

### 2. Initializing Parameters for a 2-Layer Network (Practical - 15 min) ⚙️

We need to initialize the weights ($W^{[1]}, W^{[2]}$) and biases ($b^{[1]}, b^{[2]}$).
* Weights are typically initialized with small random numbers to break symmetry (e.g., from a normal distribution, scaled down).
* Biases are typically initialized to zeros.

```python
# (Ensure numpy is imported: import numpy as np)
# (Ensure X_train_np, a pandas DataFrame/Series X_train, etc. are defined from previous lessons for context)
# import numpy as np # Already imported in previous lessons, but good to remember
# import pandas as pd # For dummy data if needed

# # Dummy X_train_np for standalone execution if previous lessons weren't run
# if 'X_train_np' not in locals() or X_train_np.size == 0:
#     print("--- X_train_np not found or empty. Using dummy data for parameter initialization example. ---")
#     num_train_samples, num_features = 712, 8 # Example dimensions
#     X_train_np = np.random.rand(num_train_samples, num_features)


def initialize_parameters_nn(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer (number of features)
    n_h -- size of the hidden layer
    n_y -- size of the output layer (1 for binary classification)

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2) # for consistency, so results are reproducible

    # Initialize W1 with small random values, b1 with zeros
    W1 = np.random.randn(n_h, n_x) * 0.01 # Small random values to break symmetry
    b1 = np.zeros((n_h, 1))              # Zeros

    # Initialize W2 with small random values, b2 with zeros
    W2 = np.random.randn(n_y, n_h) * 0.01 # Small random values
    b2 = np.zeros((n_y, 1))              # Zeros

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# Example:
# Assuming X_train_np is defined from Lesson 4 (after to_numpy())
if 'X_train_np' in locals() and X_train_np.size > 0:
    n_x_example = X_train_np.shape[1] # Number of features
    n_h_example = 4                   # Number of hidden units (hyperparameter)
    n_y_example = 1                   # Number of output units (1 for binary classification)

    params_example = initialize_parameters_nn(n_x_example, n_h_example, n_y_example)
    print("\n--- Example NN Parameter Initialization ---")
    print(f"W1 shape: {params_example['W1'].shape}")
    print(f"b1 shape: {params_example['b1'].shape}")
    print(f"W2 shape: {params_example['W2'].shape}")
    print(f"b2 shape: {params_example['b2'].shape}")
else:
    print("X_train_np not defined or is empty. Cannot run parameter initialization example.")
    # Fallback if X_train_np is not available for some reason
    n_x_example, n_h_example, n_y_example = 8, 4, 1
    params_example = initialize_parameters_nn(n_x_example, n_h_example, n_y_example)
    print("\n--- Ran NN Parameter Initialization Example with fallback dimensions ---")
    print(f"W1 shape: {params_example['W1'].shape}")
```

---

### 3. Implementing Activation Functions (Practical - 10 min) 🔥

We need Sigmoid (for the output layer) and ReLU (for the hidden layer). We already have Sigmoid from Lesson 4 (Logistic Regression).

```python
# (Ensure numpy is imported: import numpy as np)

# Sigmoid function (from Lesson 4 or can be redefined here)
def sigmoid(z):
    """
    Compute the sigmoid of z.
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s

# ReLU function
def relu(z):
    """
    Compute the ReLU of z.
    Arguments:
    z -- A scalar or numpy array.
    Return:
    r -- relu(z)
    """
    r = np.maximum(0, z)
    return r

# Test ReLU
print("\nReLU function examples:")
test_z_relu = np.array([-5, -0.5, 0, 0.5, 5])
print(f"relu({test_z_relu}) = {relu(test_z_relu)}")
# Test Sigmoid again for completeness
print("\nSigmoid function examples (re-test):")
test_z_sigmoid = np.array([-10, -1, 0, 1, 10])
print(f"sigmoid({test_z_sigmoid}) = {sigmoid(test_z_sigmoid)}")
```

4. Implementing Forward Propagation (Practical - 25 min) ➡️
Now, let's implement the forward pass step-by-step.
The input X to our forward propagation function will have shape (n_x, m_samples) where m_samples is the number of examples and n_x is the number of features. This is a common convention in many neural network implementations (features as rows, samples as columns). Our X_train_np (from Pandas) is (m_samples, n_x), so we'll need to transpose it before passing it to these NN functions.

```python
def forward_propagation_nn(X_input_nn, parameters):
    """
    Argument:
    X_input_nn -- input data of size (n_x, m_samples).
                  NOTE: X_train_np from pandas is (m_samples, n_x), so it needs to be transposed.
    parameters -- python dictionary containing your parameters (output of initialization function):
                    W1, b1, W2, b2

    Returns:
    A2 -- The sigmoid output of the second activation (final prediction probability), shape (1, m_samples)
    cache -- a dictionary containing "Z1", "A1", "Z2", and "A2" for backpropagation
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"] # Shape (n_h, n_x)
    b1 = parameters["b1"] # Shape (n_h, 1)
    W2 = parameters["W2"] # Shape (n_y, n_h)
    b2 = parameters["b2"] # Shape (n_y, 1)

    # Implement Forward Propagation to calculate A2 (probabilities)
    # Z1 = W1 . X + b1
    Z1 = np.dot(W1, X_input_nn) + b1 # Z1 shape (n_h, m_samples)
    # A1 = relu(Z1)
    A1 = relu(Z1)                    # A1 shape (n_h, m_samples)
    # Z2 = W2 . A1 + b2
    Z2 = np.dot(W2, A1) + b2         # Z2 shape (n_y, m_samples)
    # A2 = sigmoid(Z2)
    A2 = sigmoid(Z2)                 # A2 shape (n_y, m_samples)

    # Ensure A2 shape is (1, m_samples) for binary classification output from a single neuron
    assert(A2.shape == (parameters["W2"].shape[0], X_input_nn.shape[1]))
    assert(A2.shape[0] == 1) # For binary classification, n_y should be 1

    cache = {"Z1": Z1,   # Linear output of layer 1
             "A1": A1,   # Activation output of layer 1 (ReLU)
             "Z2": Z2,   # Linear output of layer 2
             "A2": A2}   # Activation output of layer 2 (Sigmoid - final prediction)

    return A2, cache

# Example of running forward propagation:
# Assuming X_train_np and params_example are defined from previous parts of Lesson 6
if 'X_train_np' in locals() and X_train_np.size > 0 and \
   'params_example' in locals() and params_example['W1'] is not None:

    # Our X_train_np is (m_samples, n_x). We need to transpose it for the NN.
    X_train_nn_input_example = X_train_np.T # Shape (n_x, m_samples)

    print(f"\nShape of original X_train_np: {X_train_np.shape}")
    print(f"Shape of X_train_nn_input_example (transposed for NN input): {X_train_nn_input_example.shape}")
    print(f"W1 shape for dot product: {params_example['W1'].shape}")


    A2_example, cache_example = forward_propagation_nn(X_train_nn_input_example, params_example)

    print("\n--- Example NN Forward Propagation Output ---")
    print(f"Shape of A2 (predictions/probabilities): {A2_example.shape}")
    # Print probabilities for the first 5 samples
    print(f"First 5 prediction probabilities (from A2): \n{A2_example[:, :5]}")
    print(f"\nCache keys: {list(cache_example.keys())}")
    print(f"Shape of Z1 in cache: {cache_example['Z1'].shape}")
    print(f"Shape of A1 in cache: {cache_example['A1'].shape}")
else:
    print("\nCannot run forward propagation example: X_train_np or params_example not defined or parameters are None.")
```

**Important Note on Data Shape:**
Our `X_train_np` (derived from Pandas) has the shape `(m_samples, n_features)`.
For many neural network implementations, including the one we are building here for matrix multiplication convenience (Weight matrix dimensions `(n_units_current_layer, n_units_previous_layer)`), the input `X` is expected to be `(n_features, m_samples)`.
So, when we use our data with these NN functions, we will need to **transpose** `X_train_np` (i.e., `X_train_np.T`).

---

### Wrap-up & Next Steps:

* **Recap:** We've defined a simple 1-hidden-layer neural network architecture. We've implemented functions to initialize its parameters ($W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$) and the necessary activation functions (ReLU and Sigmoid). Most importantly, we've coded the **forward propagation** steps, which take input data and parameters, and compute the network's output (predicted probabilities) along with intermediate values (`cache`) needed for backpropagation.
* **Teaser for Lesson 7:** With forward propagation in place, the next crucial step is to make our network *learn*. This involves:
    * Implementing the **cost function** (Binary Cross-Entropy, similar to Logistic Regression but using $A^{[2]}$).
    * Implementing **backward propagation** to calculate the gradients of the cost with respect to all parameters ($dW^{[1]}, db^{[1]}, dW^{[2]}, db^{[2]}$). This is where the chain rule comes into play!
    * Implementing the **gradient descent update rule** to adjust parameters and minimize the cost.
    * Putting it all together in a main training loop for our neural network!
