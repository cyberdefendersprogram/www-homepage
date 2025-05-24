---
layout: workshop
order: 001
slug: introduction-with-titanic-challenge
title: Introduction With Titanic Challenge 
module_category: Introduction
workshop_name: zero-to-one-datascience-to-machinelearning
last_modified_at: 2025-05-23T21:55:43Z
date: 2025-05-23
---
# Data Science & AI with the Titanic Dataset : Lesson 1

## Lesson 1: Introduction to Data Science & The Titanic Challenge 🚢

### **Objective:**
*   Understand what Data Science and AI are at a high level.
*   Get familiar with the Titanic dataset and the problem we're trying to solve.
*   Perform initial data loading and exploration using Python and Pandas.


### **1. What are Data Science and AI? (10 min)**
*   **Data Science:** Think of it as using data, scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It's about finding patterns, making predictions, and understanding the world through data. [1]
*   **Artificial Intelligence (AI):** This is a broader field focused on creating systems or machines that can perform tasks that typically require human intelligence. This includes things like learning, problem-solving, decision-making, and understanding language.
*   **Machine Learning (ML):** A subset of AI where we "teach" computers by showing them lots of examples (data) rather than programming explicit rules for every single case. Our Titanic project will be a machine learning task. The goal is to build a model that learns from the historical Titanic passenger data to predict survival.

### **2. The Titanic Dataset: "Machine Learning from Disaster" (15 min)**

*   **Background:** Briefly discuss the historical event of the Titanic sinking in 1912. Many factors influenced a passenger's chance of survival, such as their age, gender, and class.
*   **The Challenge (from Kaggle):** The goal is to build a predictive model that can determine whether a passenger survived the Titanic disaster or not, based on a set of features provided in the dataset. This is a classic beginner problem in machine learning.
*   **Why this dataset?**
    *   It's well-structured and relatively small, making it manageable for learning.
    *   The objective (predicting survival) is clear and easy to understand.
    *   It allows for exploration of various data science concepts like data cleaning, feature engineering, and model building.
*   **Data Files:**
    *   `train.csv`: This file contains passenger details *and* whether they survived (the "Survived" column, which is our target variable). We'll use this dataset to "train" our machine learning model.
    *   `test.csv`: This file contains passenger details for a different set of passengers, but it *does not* include the "Survived" column. We'll use our trained model to predict survival for these passengers and see how well it generalizes to unseen data.

### **3. Setting Up Your Environment (10 min)**

*   **Tool:** We'll be using Python, a versatile and widely-used programming language in the data science community.
*   **Key Libraries:** For this initial lesson, we'll primarily use:
    *   **Pandas:** An essential open-source library providing high-performance, easy-to-use data structures and data analysis tools. [1] We'll use it for loading and manipulating our tabular data (like spreadsheets). [1]
        *   To install Pandas, you can use pip:
            ```bash
            pip install pandas
            ```
            It's recommended to do this within a virtual environment. [3]
    *   **NumPy:** A fundamental package for scientific computing with Python. Pandas is built on top of NumPy. [1] It's often installed as a dependency with Pandas. If not, you can install it via:
        ```bash
        pip install numpy
        ```
*   **Importing Libraries:** In your Python script or Jupyter Notebook, you'll import these libraries like so:
    ```python
    import pandas as pd
    import numpy as np
    ```
    Using `pd` as an alias for Pandas is a widely adopted convention. [1]

### **4. First Look at the Data (20 min)**

We'll use Pandas to load and inspect the `train.csv` file.

*   **Load the training data:**
    The `pd.read_csv()` function is used to read data from a CSV file into a Pandas DataFrame. [6, 7] A DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns), much like a spreadsheet. [2]
    ```python
    import pandas as pd
    import numpy as np # Ensure numpy is imported for np.nan later

    # Assuming 'train.csv' is in the same directory as your script/notebook
    # If it's elsewhere, provide the full path to the file.
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found. Please make sure the file is in the correct directory.")
        # In a real scenario, you might exit or handle this more gracefully.
        # For this lesson, we'll create a placeholder if the file isn't found to allow continuation.
        data = {'PassengerId': [1, 2, 3, 4, 5],
                'Survived': ,
                'Pclass': [3, 1, 3, 1, 3],
                'Name':,
                'Sex': ['male', 'female', 'female', 'female', 'male'],
                'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
                'SibSp': ,
                'Parch': ,
                'Ticket':,
                'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
                'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan], # Use np.nan for missing Cabin values
                'Embarked':}
        train_df = pd.DataFrame(data)
        print("\n--- Created a placeholder DataFrame as 'train.csv' was not found. ---")
    ```

*   **Inspect the data:** Pandas DataFrames have several useful methods and attributes for a quick overview.
    *   `train_df.head()`: Shows the first few rows of the DataFrame (default is 5 rows). [1] This is great for a quick glance at the data's structure and some sample values.
        ```python
        print("First 5 rows of the training data:")
        print(train_df.head())
        ```
    *   `train_df.tail()`: Shows the last few rows of the DataFrame (default is 5 rows). [1]
        ```python
        print("\nLast 5 rows of the training data:")
        print(train_df.tail())
        ```
    *   `train_df.info()`: Provides a concise summary of the DataFrame, including the data type of each column, the number of non-null values, and memory usage. [8, 9] This is very useful for identifying columns with missing data.
        ```python
        print("\nDataFrame information:")
        train_df.info()
        ```
    *   `train_df.describe()`: Generates descriptive statistics for numerical columns (like count, mean, std, min, max, and quartiles). [1] For object/string columns, it provides count, unique, top (most frequent), and freq (frequency of top). [1]
        ```python
        print("\nDescriptive statistics for numerical columns:")
        print(train_df.describe())

        print("\nDescriptive statistics for object columns:")
        print(train_df.describe(include=['object']))
        ```
    *   `train_df.shape`: An attribute that returns a tuple representing the dimensionality of the DataFrame (number_of_rows, number_of_columns). [2, 10]
        ```python
        print("\nShape of the DataFrame (rows, columns):")
        print(train_df.shape)
        ```

*   **Discuss Key Columns (Features):**
    Based on the output of `train_df.head()` and `train_df.info()`, we can identify the columns:
    *   `PassengerId`: Unique ID for each passenger.
    *   `Survived`: Our target variable (0 = No, 1 = Yes). This is what we want to predict.
    *   `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) - often a proxy for socio-economic status.
    *   `Name`: Passenger's name.
    *   `Sex`: Passenger's sex (male or female).
    *   `Age`: Passenger's age in years. (Note: `info()` will show if there are missing values here).
    *   `SibSp`: Number of siblings / spouses aboard the Titanic.
    *   `Parch`: Number of parents / children aboard the Titanic.
    *   `Ticket`: Ticket number.
    *   `Fare`: Passenger fare.
    *   `Cabin`: Cabin number. (Note: `info()` will likely show many missing values here).
    *   `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

### **5. Defining the Problem (5 min)**

*   Clearly state the problem: We want to build a machine learning model that takes various passenger features (like `Pclass`, `Sex`, `Age`, `Fare`, etc.) as input and predicts whether they `Survived` the Titanic disaster.
*   This is a **binary classification** problem because the outcome we are trying to predict (`Survived`) has two possible values: 0 (did not survive) or 1 (survived).


### **Wrap-up & Next Steps for Lesson 1 (5 min)**
*   **Recap:** In this lesson, we've been introduced to the basic concepts of Data Science, AI, and Machine Learning. We've familiarized ourselves with the Titanic dataset, its challenge, and used Pandas to perform initial data loading and inspection.
*   **Teaser for Lesson 2:** Next time (which is now!), we will dive deeper into [**Exploratory Data Analysis (EDA)**](/0-to-1-datascience2ai-2).


