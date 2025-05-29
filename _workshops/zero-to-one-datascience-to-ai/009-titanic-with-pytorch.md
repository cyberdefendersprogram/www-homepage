---
layout: workshop
order: 9
slug: titanic-with-pytorch
title: Titanic With Pytorch
workshop_name: zero-to-one-datascience-to-ai
module_category: Advanced Machine Learning
date: 2025-05-23
---
## Lesson 9: Solving the Titanic Problem with PyTorch 🔥🤖

**Objective:**

* Understand the fundamental components of PyTorch for building neural networks.
* Learn how to define, train, and evaluate a neural network for the Titanic survival prediction task using PyTorch.
* Appreciate the advantages of using a deep learning framework.

### Recap of Previous Lessons:

* We've preprocessed the Titanic dataset (handling missing values, encoding categoricals, scaling numerical features).
* We've split our data into training (`X_train`, `y_train`) and validation (`X_val`, `y_val`) sets.
* We've built a logistic regression model and a simple neural network *from scratch* to understand the underlying mechanics.

### 1. Why PyTorch? (Conceptual - 10 min)

Building neural networks from scratch is invaluable for learning, but for larger, more complex models, or for leveraging hardware like GPUs, frameworks are essential.

**Advantages of PyTorch (and similar frameworks like TensorFlow/Keras):**

* **Automatic Differentiation (`autograd`):** PyTorch automatically calculates gradients for backpropagation. No need to manually derive and implement gradient formulas!
* **Pre-built Layers & Modules:** Provides optimized implementations of common layers (linear, convolutional, recurrent), activation functions, loss functions, etc.
* **Optimizers:** Includes various optimization algorithms (SGD, Adam, RMSprop, etc.).
* **GPU Support:** Easily run computations on NVIDIA GPUs for significant speedups in training deep models.
* **Dynamic Computation Graphs:** PyTorch uses dynamic graphs (define-by-run), which can be more intuitive for some and flexible for models with varying structures.
* **Large Community & Ecosystem:** Extensive documentation, tutorials, pre-trained models, and supporting libraries.

### 2. Core PyTorch Concepts (Conceptual - 15 min)

* **Tensors:** The fundamental data structure in PyTorch, similar to NumPy arrays. Tensors can be moved to a GPU for accelerated computation.
    ```python
    import torch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(x)
    ```
* **`torch.nn.Module`:** The base class for all neural network modules (layers, or the entire network itself). Your custom models will inherit from this.
* **Layers (e.g., `torch.nn.Linear`):** Pre-defined layers. `nn.Linear(in_features, out_features)` creates a fully connected (dense) layer.
* **Activation Functions (e.g., `torch.nn.ReLU`, `torch.nn.Sigmoid`):** Found in `torch.nn` or `torch.nn.functional`.
* **Loss Functions (e.g., `torch.nn.BCELoss`):** Quantify the difference between predictions and true labels. `BCELoss` (Binary Cross Entropy Loss) is common for binary classification.
* **Optimizers (e.g., `torch.optim.SGD`, `torch.optim.Adam`):** Implement algorithms to update model weights based on gradients.
* **`autograd`:** PyTorch's automatic differentiation engine. When a tensor has `requires_grad=True`, PyTorch tracks all operations on it to compute gradients. `loss.backward()` triggers gradient computation.

### 3. Building a Neural Network for Titanic with PyTorch (Practical)

Let's rebuild our 1-hidden-layer neural network.

#### a) Data Preparation: Pandas to PyTorch Tensors (Practical - 15 min)

We'll assume `X_train`, `y_train`, `X_val`, `y_val` are your preprocessed Pandas DataFrames/Series from Lesson 3. PyTorch models require PyTorch tensors as input.

```python
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Assume 'train_df' is your loaded Titanic training data
# For demonstration, let's quickly create dummy data resembling preprocessed Titanic data
# In a real scenario, this would come from your comprehensive preprocessing steps

# Dummy features and target
data = {
    'Pclass': [1, 2, 3, 1, 2, 3, 1, 2],
    'Sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
    'Age': [22, 38, 26, 35, 35, None, 54, 2],
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 3],
    'Parch': [0, 0, 0, 0, 0, 0, 0, 1],
    'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S'],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 1]
}
train_df = pd.DataFrame(data)

# Separate target variable
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Define numerical and categorical features
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Create preprocessing pipelines for numerical and categorical features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Split data (important to do this *after* preprocessing or fit preprocessor only on train)
# For simplicity here, we'll just use the processed X and original y
# In a real scenario, you'd split X and y first, then fit_transform on X_train and transform on X_test/X_val
X_train_processed, X_val_processed, y_train_series, y_val_series = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Convert to PyTorch Tensors
# Ensure X_train_processed and X_val_processed are numpy arrays if they are sparse matrices
if hasattr(X_train_processed, "toarray"): # Check if it's a sparse matrix
    X_train_np = X_train_processed.toarray()
    X_val_np = X_val_processed.toarray()
else:
    X_train_np = X_train_processed
    X_val_np = X_val_processed

X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_series.values, dtype=torch.float32).unsqueeze(1) # Unsqueeze for (batch_size, 1) shape

X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_series.values, dtype=torch.float32).unsqueeze(1)

print("Shape of X_train_tensor:", X_train_tensor.shape)
print("Shape of y_train_tensor:", y_train_tensor.shape)
print("Shape of X_val_tensor:", X_val_tensor.shape)
print("Shape of y_val_tensor:", y_val_tensor.shape)

# Determine input_size from the processed features
input_size = X_train_tensor.shape[1]
print(f"Number of input features: {input_size}")
```

Explanation:
- We first ensure our data X_train, X_val (features) and y_train, y_val (labels) are in a numerical format (e.g., NumPy arrays after preprocessing).
- torch.tensor(data, dtype=torch.float32) converts NumPy arrays or lists into PyTorch tensors. We use float32 as it's standard for neural network inputs.
- For the target y, unsqueeze(1) is often used to change its shape from (batch_size,) to (batch_size, 1), which is typically expected by loss functions like BCELoss when dealing with batch predictions.

** b) Defining the Neural Network (Practical - 20 min) **
We'll create a class that inherits from torch.nn.Module.

```python
import torch.nn as nn
import torch.nn.functional as F # Often used for activation functions

class TitanicNet(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(TitanicNet, self).__init__() # Call the init of the parent class nn.Module
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size1) # Fully connected layer 1
        self.relu1 = nn.ReLU()                         # ReLU activation
        self.fc2 = nn.Linear(hidden_size1, output_size) # Fully connected layer 2 (output layer)
        self.sigmoid = nn.Sigmoid()                    # Sigmoid activation for binary classification

    def forward(self, x):
        # Define the forward pass: how data flows through the layers
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out) # Output a probability between 0 and 1
        return out

# Hyperparameters for the network structure
# input_size was determined from X_train_tensor.shape[1] in the previous step
hidden_size1 = 32  # Number of neurons in the first hidden layer
output_size = 1    # Output a single value (probability of survival)

# Instantiate the model
model = TitanicNet(input_size, hidden_size1, output_size)
print(model)
```

Explanation:

- __init__(self, ...): The constructor. Here we define all the layers our network will use.
    - super(TitanicNet, self).__init__(): Essential to call the __init__ method of the parent nn.Module class.
    - nn.Linear(in_features, out_features): Creates a dense (fully connected) layer. in_features is the number of input features to this layer, and out_features is the number of output features (neurons).
    - nn.ReLU() and nn.Sigmoid(): Instances of activation function layers.
- forward(self, x): This method defines how the input x passes through the layers defined in __init__.
    - The output of one layer becomes the input to the next.
    - For binary classification, the final layer often uses a Sigmoid activation to squash the output to a probability between 0 and 1.


** c) Defining Loss Function and Optimizer (Practical - 10 min) **

```python 
# Loss Function
criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification

# Optimizer
learning_rate = 0.001 # A common starting point
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# model.parameters() tells the optimizer which tensors (weights and biases) to update.

print(f"Using Loss: {criterion}")
print(f"Using Optimizer: {optimizer}")

```

Explanation:

- nn.BCELoss(): Suitable for binary classification problems where the model outputs a probability (after a Sigmoid activation).
- torch.optim.Adam: A popular and often effective optimization algorithm. It adapts the learning rate for each parameter.
    - model.parameters(): This conveniently provides all learnable parameters (weights and biases of all layers defined in our TitanicNet) to the optimizer.
    - lr: The learning rate, controlling the step size during optimization.

** d) Training Loop (Practical - 25 min) **
This is where the model learns from the training data.

```python
num_epochs = 200  # Number of times to iterate over the entire training dataset
batch_size = 16   # Number of samples to process before updating weights (if using DataLoader)

# For simplicity, we'll do full-batch training here (process all training data at once)
# For larger datasets, you'd use torch.utils.data.DataLoader to create mini-batches

print(f"Starting training for {num_epochs} epochs...")

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train() # Set the model to training mode (important for layers like Dropout, BatchNorm)

    # --- Training Phase ---
    # 1. Forward pass: Compute predicted y by passing X_train to the model
    outputs = model(X_train_tensor)

    # 2. Compute loss
    loss = criterion(outputs, y_train_tensor)
    train_losses.append(loss.item()) # .item() gets the scalar value of the loss

    # 3. Backward pass and optimize
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients for all parameters with requires_grad=True
    optimizer.step()       # Update weights using the computed gradients

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for validation (saves memory and computation)
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())

        # Calculate validation accuracy
        predicted_classes = (val_outputs > 0.5).float() # Convert probabilities to 0 or 1
        correct_predictions = (predicted_classes == y_val_tensor).sum().item()
        total_predictions = y_val_tensor.size(0)
        accuracy = correct_predictions / total_predictions
        val_accuracies.append(accuracy)

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy:.4f}')

print("Training finished.")
```

Explanation of the Training Loop:

1. model.train(): Sets the model to training mode. This is important if your model includes layers like Dropout or BatchNorm, which behave differently during training and evaluation.
2. Forward Pass (outputs = model(X_train_tensor)): The input data is passed through the network, and the model produces predictions.
3. Compute Loss (loss = criterion(outputs, y_train_tensor)): The loss function compares the model's predictions (outputs) with the actual labels (y_train_tensor) to quantify how wrong the predictions are.
4. Backward Pass and Optimization:
    - optimizer.zero_grad(): Gradients accumulate by default. So, before each backward pass, we must explicitly zero out the gradients from the previous step.
    - loss.backward(): This is where PyTorch's autograd shines. It computes the gradients of the loss with respect to all model parameters (those with requires_grad=True).
    - optimizer.step(): The optimizer updates the model's parameters using the gradients computed in the backward() call and the chosen optimization algorithm (e.g., Adam).
5. model.eval() and with torch.no_grad():
    - model.eval() sets the model to evaluation mode.
    - torch.no_grad() disables gradient calculations within its block. This is crucial during validation/testing because it reduces memory consumption and speeds up computations, as we don't need to track gradients for weight updates.
6. Metrics Tracking: We store training loss, validation loss, and validation accuracy to monitor the learning process.

Note on Batching:
For this example, we're performing full-batch gradient descent (processing all training samples at once). For larger datasets, this is not feasible due to memory constraints. You would typically use torch.utils.data.TensorDataset and torch.utils.data.DataLoader to create mini-batches of data, and the training loop would iterate over these mini-batches

**e) Evaluating the Model (Practical - 10 min) **
We already incorporated evaluation within the training loop. Let's visualize the learning process.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Final evaluation on the validation set
model.eval() # Ensure model is in evaluation mode
with torch.no_grad():
    y_pred_val = model(X_val_tensor)
    y_pred_val_classes = (y_pred_val > 0.5).float() # Threshold at 0.5 for binary classification
    accuracy = (y_pred_val_classes == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print(f'Final Validation Accuracy: {accuracy * 100:.2f}%')
```

Explanation:
    - Plotting the training and validation loss helps to identify overfitting (when training loss continues to decrease while validation loss starts to increase).
    - Plotting validation accuracy shows how well the model generalizes to unseen data over epochs.
    - The final accuracy gives a single metric for the model's performance on the validation set.

** f) Making Predictions on New Data (e.g., Test Set) (Conceptual - 5 min) **
Once your model is trained, you'd use it to predict on the actual Kaggle test set.

```python
# Assume 'test_df' is loaded and preprocessed similarly to X_train
# X_test_processed = preprocessor.transform(test_df) # Use the *fitted* preprocessor
# if hasattr(X_test_processed, "toarray"):
#     X_test_np = X_test_processed.toarray()
# else:
#     X_test_np = X_test_processed
# X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

# model.eval() # Set to evaluation mode
# with torch.no_grad(): # No gradients needed for prediction
#     test_predictions_probs = model(X_test_tensor)
#     test_predictions_classes = (test_predictions_probs > 0.5).int() # Convert to 0 or 1

# # Now 'test_predictions_classes' contains your predictions for the test set.
# # You would then format these into a submission file as per Kaggle's requirements.
# print(test_predictions_classes[:10]) # Print first 10 predictions
```

Explanation:

1. Load and preprocess the test data using the same preprocessing steps and fitted objects (like scalers, encoders) from the training data. This is crucial.
2. Convert the preprocessed test data to a PyTorch tensor.
3. Set the model to evaluation mode (model.eval()).
4. Use torch.no_grad() as we are not training.
5. Pass the test tensor through the model to get probability predictions.
5. Convert probabilities to class labels (0 or 1) using a threshold (typically 0.5).
7. Format these predictions into the submission file format specified by the Kaggle competition.

4. Advantages Revisited & Next Steps (Conceptual - 10 min)
Having used PyTorch, you should now appreciate:

* Ease of Building: nn.Module and pre-built layers make defining complex architectures straightforward.
* Automatic Gradients: loss.backward() is powerful. No manual calculus for backpropagation!
* Flexibility: Easy to modify network structure, change optimizers, or loss functions.
* GPU Acceleration (if available): With a GPU, you'd add .to('cuda') to your model and tensors for significant speedups, with minimal code changes.

```python
# Example of GPU usage
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# X_train_tensor = X_train_tensor.to(device)
# y_train_tensor = y_train_tensor.to(device)
# ... and so on for other tensors
```

Possible Next Steps & Improvements:

- Hyperparameter Tuning: Experiment with learning_rate, batch_size, num_epochs, number of hidden layers, number of neurons per layer, different activation functions.
- Regularization: Add nn.Dropout layers or L2 regularization (weight decay in the optimizer) to prevent overfitting.
```
# Example Dropout in __init__
# self.dropout = nn.Dropout(p=0.5) # 50% dropout
# Example usage in forward pass (after activation, before next linear layer)
# out = self.dropout(out)
#
# Example L2 regularization (weight_decay in Adam optimizer)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
```
- Different Optimizers: Try SGD with momentum, RMSprop, etc.
- More Complex Architectures: For image data, Convolutional Neural Networks (CNNs); for sequential data, Recurrent Neural Networks (RNNs). (Not directly applicable to this tabular Titanic data without feature engineering).
- Cross-Validation: For more robust evaluation instead of a single train/val split.
- Using DataLoader: Implement proper mini-batch training for larger datasets
```python
from torch.utils.data import TensorDataset, DataLoader
# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Create DataLoaders
batch_size = 32 # Example batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Training loop would then iterate over train_loader:
# for epoch in range(num_epochs):
#     for inputs, labels in train_loader:
#         # inputs, labels are now a mini-batch
#         # Move to device if using GPU: inputs, labels = inputs.to(device), labels.to(device)
#         # ... rest of training step ...
#     # Validation loop would iterate over val_loader
```
- Saving and Loading Models:
```
# Save model
# torch.save(model.state_dict(), 'titanic_model.pth')

# Load model
# model_loaded = TitanicNet(input_size, hidden_size1, output_size)
# model_loaded.load_state_dict(torch.load('titanic_model.pth'))
# model_loaded.eval() # Set to evaluation mode after loading
```

### Summary
In this lesson, you've successfully built, trained, and evaluated a neural network for the Titanic survival prediction task using PyTorch. You've learned about core PyTorch components like Tensors, `nn.Module`, layers, loss functions, optimizers, and the `autograd` system. This provides a solid foundation for tackling more complex deep learning problems.
