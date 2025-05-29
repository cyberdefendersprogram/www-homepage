---
layout: workshop
order: 006
slug: neural-networks-from-scratch
title: Neural Networks From Scratch
module_category: Machine Learning
workshop_name: zero-to-one-datascience-to-ai
date: 2025-05-23
---
## Lesson 6: Neural Networks - The Core Concepts (Conceptual) 🧠💡

> [The Original Neural Network Paper]()
>

**Objective:**
* Understand the fundamental building blocks of a neural network: neurons, layers, weights, biases, and activation functions.
* Grasp the conceptual flow of information (forward propagation) and learning (loss function, backpropagation, optimization) in a neural network.
* Relate these concepts back to what we learned with Logistic Regression.
* (No coding in this lesson, focus is on intuition and concepts).

---

### Recap of Lesson 4:
* Built Logistic Regression from scratch, understanding its components:
    * Linear combination $Z = WX + b$.
    * Sigmoid activation $\sigma(Z)$ to get probabilities.
    * Cost function (Binary Cross-Entropy) to measure error.
    * Gradient Descent to update $W$ and $b$ to minimize cost.

---

### 1. The Analogy: From Biological Neurons to Artificial Neurons (5 min)

* **Biological Neurons:** Briefly, brain cells (neurons) receive signals through dendrites, process them in the cell body (soma), and if a threshold is reached, fire an output signal along an axon to other neurons.
* **Artificial Neurons (Perceptrons/Units):** Inspired by this, an artificial neuron:
    * Receives one or more **inputs** (features from data, or outputs from previous neurons).
    * Each input has an associated **weight** (representing its importance).
    * Computes a **weighted sum** of its inputs.
    * Adds a **bias** term (an independent parameter, like an intercept).
    * Applies an **activation function** to this sum to produce an **output**.

---

### 2. The Single Artificial Neuron (Perceptron / Logistic Regression Unit) (15 min)

Let's look at a single neuron in detail. It's very similar to what we did in Logistic Regression!

* **Inputs ($x_1, x_2, ..., x_n$):** These are the features of a single data sample.
* **Weights ($w_1, w_2, ..., w_n$):** Each input $x_i$ is multiplied by a corresponding weight $w_i$. These weights are *learned* during training and determine the influence of each input on the neuron's output.
* **Bias ($b$):** An additional parameter that is added to the weighted sum. It allows the neuron to be activated even if all inputs are zero, or to shift the activation function.
* **Weighted Sum (Linear Combination, $z$):**
    $$z = (w_1 x_1 + w_2 x_2 + \dots + w_n x_n) + b = W \cdot X + b$$
    *This is exactly the linear part we calculated in Logistic Regression!*
* **Activation Function ($\sigma(z)$, or $f(z)$):**
    The weighted sum $z$ is then passed through an activation function.
    * **Purpose:** To introduce **non-linearity** into the model. Without non-linear activation functions in hidden layers, a multi-layer neural network would behave just like a single linear model.
    * **Examples:**
        * **Sigmoid:** $\sigma(z) = 1 / (1 + e^{-z})$. Outputs between 0 and 1. Used in Logistic Regression and often in the output layer of binary classification neural networks.
        * **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$. Outputs $z$ if $z>0$, and $0$ otherwise. Very popular for hidden layers because it's computationally efficient and helps mitigate the "vanishing gradient" problem.
        * **Tanh (Hyperbolic Tangent):** $f(z) = (e^z - e^{-z}) / (e^z + e^{-z})$. Outputs between -1 and 1. Similar to sigmoid but zero-centered.
        * **Softmax:** Used in the output layer for multi-class classification. Converts a vector of scores into a probability distribution (outputs sum to 1).

    *A single neuron with a sigmoid activation function is essentially a Logistic Regression unit.*

---

### 3. Layers: Stacking Neurons Together (15 min)

Neural networks gain their power by organizing neurons into layers.

* **Input Layer:**
    * Not really a layer of "computing" neurons. It simply represents the input features of our data.
    * The number of "nodes" or "units" in the input layer corresponds to the number of features in our dataset (e.g., for our processed Titanic data, this would be columns like 'Age', 'Fare', 'Sex_male', 'Embarked_Q', etc.).
* **Hidden Layer(s):**
    * Layers of neurons between the input and output layers. These are where most of the "learning" of complex patterns happens.
    * Each neuron in a hidden layer typically receives input from *all* neurons (or inputs) in the previous layer (this is called a "fully connected" or "dense" layer).
    * Each connection has its own weight. Each neuron has its own bias.
    * They apply an activation function (commonly ReLU or Tanh).
    * A network can have one or more hidden layers. Networks with many hidden layers are called "deep" neural networks (Deep Learning).
    * The number of neurons in each hidden layer is a design choice (a hyperparameter).
* **Output Layer:**
    * The final layer of neurons that produces the model's prediction.
    * The number of neurons and the activation function depend on the task:
        * **Binary Classification (like Titanic):** Typically 1 neuron with a sigmoid activation function (to output a probability between 0 and 1).
        * **Multi-Class Classification (e.g., classifying images into 10 categories):** Typically $N$ neurons (where $N$ is the number of classes) with a softmax activation function.
        * **Regression (predicting a continuous value):** Typically 1 neuron with no activation function (or a linear activation function).

---

### 4. How Neural Networks "Learn" - The Big Picture (Conceptual - 20 min) 🔄

The learning process is an iterative optimization procedure:

1.  **Initialization:**
    * Weights are typically initialized with small random numbers (to break symmetry and allow different neurons to learn different things). Biases are often initialized to zero.

2.  **Forward Propagation:**
    * A batch of training data (or a single sample) is fed into the input layer.
    * The data flows through the network, layer by layer:
        * For each neuron in a layer, calculate its weighted sum ($z$) based on outputs from the previous layer and its own weights/bias.
        * Apply the activation function to $z$ to get the neuron's output ($a$).
    * This process continues until the output layer produces a prediction.

3.  **Calculate Loss (Cost Function):**
    * The model's predictions are compared to the true labels from the training data using a **loss function** (also called a cost function or error function).
    * The loss function quantifies how "wrong" the model's predictions are.
    * **Examples:**
        * Binary Classification: **Binary Cross-Entropy** (same as we used for Logistic Regression).
        * Multi-Class Classification: Categorical Cross-Entropy.
        * Regression: Mean Squared Error (MSE) or Mean Absolute Error (MAE).

4.  **Backward Propagation (Backprop):**
    * This is the core of how neural networks learn. It's a clever algorithm for efficiently calculating the **gradients** of the loss function with respect to *every single weight and bias* in the network.
    * It uses the **chain rule** from calculus to propagate the error "backwards" from the output layer through the hidden layers to the input layer.
    * Essentially, it determines how much each weight and bias contributed to the overall loss.

5.  **Optimization (Update Parameters):**
    * Using the calculated gradients, an **optimizer** updates the weights and biases in the network to reduce the loss.
    * The most basic optimizer is **Gradient Descent** (which we implemented for Logistic Regression): $W = W - \alpha \cdot dW$.
    * More advanced optimizers are commonly used in practice:
        * Stochastic Gradient Descent (SGD) with Momentum
        * AdaGrad
        * RMSprop
        * **Adam** (Adaptive Moment Estimation - very popular and often a good default choice).
        These optimizers adapt the learning rate or use momentum to speed up convergence and navigate complex loss landscapes.

6.  **Repeat:**
    * Steps 2-5 are repeated for many **epochs** (an epoch is one full pass through the entire training dataset) or until the model's performance on a validation set stops improving.

---

### Relating to Logistic Regression:

* A Logistic Regression model can be viewed as a very simple neural network:
    * An input layer (your features).
    * No hidden layers.
    * An output layer with a single neuron using a sigmoid activation function.
* The "learning" process we built for Logistic Regression (forward prop, binary cross-entropy cost, backward prop for dW and db, gradient descent update) is a specific instance of the general neural network learning process.

---

### Wrap-up & Next Steps:

* **Recap:** We've conceptually explored the core components of neural networks: neurons, layers (input, hidden, output), weights, biases, and various activation functions (sigmoid, ReLU). We've also outlined the learning process involving forward propagation, loss calculation, backward propagation (to find gradients), and optimization (like gradient descent) to update parameters. We saw that Logistic Regression is a simple form of a neural network.
* **Teaser for Lesson 6:** Next, we'll start **coding our own simple neural network from scratch** using NumPy, focusing first on implementing **Forward Propagation** for a network with one hidden layer. This will solidify our understanding of how data flows through the network and how predictions are generated.
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
