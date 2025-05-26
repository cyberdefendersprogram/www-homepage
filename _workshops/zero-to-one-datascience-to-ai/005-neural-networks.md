---
layout: workshop
order: 005
slug: neural-networks
title: Neural Networks
module_category: Machine Learning
workshop_name: zero-to-one-datascience-to-ai
date: 2025-05-23
---
## Lesson 5: Neural Networks - The Core Concepts (Conceptual) 🧠💡

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
