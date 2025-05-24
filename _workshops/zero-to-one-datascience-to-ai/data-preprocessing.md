---
layout: workshop
order: 003
slug: data-preprocessing
title: Data Preprocessing
module_category: Data Science
workshop_name: zero-to-one-datascience-to-machinelearning
date: 2025-05-23
---
## Lesson 3: Data Preprocessing for Machine Learning ⚙️

**Objective:**
* Convert categorical features into numerical representations suitable for machine learning models.
* Understand and apply feature scaling to numerical features.
* Split the data into training and validation sets to properly evaluate model performance.

---

### Recap of Lesson 2:
* Handled missing data in `Age`, `Cabin`, and `Embarked`.
* Visualized relationships between various features and survival.
* Created new features: `FamilySize` and `IsAlone`.
* Our working DataFrame is `df_processed`. *(If running standalone, ensure `df_processed` is created or loaded as in Lesson 2)*

---

### 1. Converting Categorical Features to Numerical (Practical - 25 min)

Machine learning algorithms typically require numerical input. We need to convert categorical features (text-based) into numbers.

* **Identify Categorical Columns for Conversion:**
    In `df_processed`, the main categorical columns we need to convert are `Sex` and `Embarked`. `Pclass` is numerical but represents categories; we'll treat it as numerical for now, but it could also be one-hot encoded if treated as purely nominal. If we had extracted `Title` from the `Name` column, that would also be categorical.

    ```python
    # (Assuming df_processed is available from Lesson 2)
    # import pandas as pd # Ensure pandas is imported
    # import numpy as np # Ensure numpy is imported
    # # Dummy df_processed for standalone execution in Canvas if necessary
    # if 'df_processed' not in locals():
    #     data = {'PassengerId': [1,2,3,4,5], 'Survived': [0,1,1,0,1], 'Pclass': [3,1,3,1,2],
    #             'Name': ['A','B','C','D','E'], 'Sex': ['male','female','female','male','female'],
    #             'Age': [22,38,26,35,28], 'SibSp': [1,1,0,1,0], 'Parch': [0,0,0,1,0],
    #             'Ticket': ['t1','t2','t3','t4','t5'], 'Fare': [7.25,71.2,7.9,53.1,13],
    #             'Embarked': ['S','C','S','S','C'], 'FamilySize': [2,2,1,2,1], 'IsAlone': [0,0,1,0,1]}
    #     df_processed = pd.DataFrame(data)
    #     print("--- Loaded a dummy df_processed for Lesson 3 demonstration. ---")

    print("Data types before categorical conversion:")
    print(df_processed.dtypes)
    ```

* **Techniques for Conversion:**
    * **Label Encoding:** Assigns a unique integer to each category (e.g., Male=0, Female=1).
        * *Pros:* Simple.
        * *Cons:* Can inadvertently introduce an ordinal relationship where none exists (e.g., model might think category 2 is "greater" than category 1). Usually not ideal for nominal categories (like `Sex` or `Embarked`) if using linear models or distance-based models.
    * **One-Hot Encoding (Dummy Variables):** Creates new binary (0 or 1) columns for each category. For example, if `Embarked` has values S, C, Q, one-hot encoding creates three new columns: `Embarked_S`, `Embarked_C`, `Embarked_Q`. A passenger from port S would have 1 in `Embarked_S` and 0 in the others.
        * *Pros:* No ordinal relationship is implied. Generally better for nominal data with most algorithms.
        * *Cons:* Can increase the number of features (dimensionality), especially if a category has many unique values.

* **Applying One-Hot Encoding with Pandas:**
    Pandas has a convenient function `pd.get_dummies()` for this.

    ```python
    # Let's work on a new copy for this stage, or continue with df_processed
    df_model_ready = df_processed.copy()

    # One-Hot Encode 'Sex'
    # drop_first=True is often used to avoid multicollinearity (dummy variable trap).
    # If you have k categories, drop_first=True creates k-1 dummy variables.
    # The dropped category is implicitly represented when all other dummies are 0.
    if 'Sex' in df_model_ready.columns:
        df_model_ready = pd.get_dummies(df_model_ready, columns=['Sex'], prefix='Sex', drop_first=True)
        print("\nDataFrame after One-Hot Encoding 'Sex'.")
    else:
        print("\n'Sex' column not found for One-Hot Encoding.")


    # One-Hot Encode 'Embarked'
    if 'Embarked' in df_model_ready.columns:
        df_model_ready = pd.get_dummies(df_model_ready, columns=['Embarked'], prefix='Embarked', drop_first=True)
        print("DataFrame after One-Hot Encoding 'Embarked'.")
    else:
        print("\n'Embarked' column not found for One-Hot Encoding.")


    print("\nDataFrame after One-Hot Encoding (head):")
    print(df_model_ready.head())
    print("\nNew columns created (if any):")
    print(df_model_ready.columns)
    ```

* **Drop Unnecessary Columns for Modeling:**
    Some columns might not be useful for the model or might have been superseded by engineered features.
    * `Name`: We haven't extracted titles yet, so the raw name is hard for most models to use.
    * `Ticket`: Ticket numbers are often unique or require complex parsing.
    * `PassengerId`: Just an identifier.
    * `SibSp` and `Parch`: We created `FamilySize` and `IsAlone`, which might capture this information more effectively. Let's decide if we want to keep the originals or just the engineered ones. For now, let's keep `FamilySize` and `IsAlone` and drop `SibSp` and `Parch` to avoid redundancy if `FamilySize` captures their essence.

    ```python
    columns_to_drop_for_model = ['Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch']
    # Ensure columns exist before attempting to drop, to avoid errors
    columns_present_to_drop = [col for col in columns_to_drop_for_model if col in df_model_ready.columns]

    if columns_present_to_drop:
        df_model_ready.drop(columns=columns_present_to_drop, axis=1, inplace=True)
        print(f"\nDropped columns for modeling: {columns_present_to_drop}")
    else:
        print(f"\nSome columns intended for dropping were not found or already dropped: {columns_to_drop_for_model}")


    print("\nDataFrame ready for scaling (head):")
    print(df_model_ready.head())
    print("\nFinal columns for the model (before scaling):")
    print(df_model_ready.columns)
    ```
---

### 2. Feature Scaling (Numerical Features) (Practical - 20 min)

Many machine learning algorithms perform better when numerical input features are on a similar scale. This is especially true for algorithms that compute distances (like k-NN) or use gradient descent (like Logistic Regression, SVMs, Neural Networks).

* **Identify Numerical Columns for Scaling:**
    From `df_model_ready.columns`, our numerical features are likely `Age`, `Fare`, `Pclass` (treating as numerical), `FamilySize`. The one-hot encoded columns are already 0/1. `IsAlone` is also 0/1. `Survived` is the target, not a feature to scale.

    ```python
    # (Assuming df_model_ready is available from Part 1 of Lesson 3)
    # numerical_cols_for_scaling = ['Age', 'Fare', 'Pclass', 'FamilySize'] # Example from before
    # For robustness, dynamically identify numerical columns that are not binary (0/1) or the target
    
    if 'df_model_ready' in locals() and not df_model_ready.empty:
        potential_numerical_cols = df_model_ready.select_dtypes(include=np.number).columns.tolist()
        # Exclude target and already binary/dummy columns
        cols_to_exclude_from_scaling = ['Survived'] # Always exclude target
        for col in df_model_ready.columns:
            if df_model_ready[col].nunique() <= 2: # If column is binary (like dummies, IsAlone)
                cols_to_exclude_from_scaling.append(col)
        
        numerical_cols_for_scaling = [col for col in potential_numerical_cols if col not in cols_to_exclude_from_scaling]
        numerical_cols_for_scaling = [col for col in numerical_cols_for_scaling if col in df_model_ready.columns] # Ensure they still exist
    else: # Fallback if df_model_ready is not defined
        numerical_cols_for_scaling = ['Age', 'Fare', 'Pclass', 'FamilySize'] 
        # This fallback list requires manual adjustment if columns were named differently or dropped.
        print("--- Using a predefined list for numerical_cols_for_scaling as df_model_ready was not found. ---")


    print(f"\nNumerical columns identified for scaling: {numerical_cols_for_scaling}")
    if numerical_cols_for_scaling and 'df_model_ready' in locals() and not df_model_ready.empty:
         # Ensure all listed columns are actually in the DataFrame before calling describe
        valid_cols_for_describe = [col for col in numerical_cols_for_scaling if col in df_model_ready.columns]
        if valid_cols_for_describe:
            print(df_model_ready[valid_cols_for_describe].describe())
        else:
            print("None of the identified numerical columns for scaling are present in df_model_ready.")
    elif not numerical_cols_for_scaling:
        print("No numerical columns were identified for scaling.")
    ```
    *Observe the different means and standard deviations if printed.*

* **Techniques for Scaling:**
    * **Standardization (Z-score Normalization):** Transforms data to have a mean of 0 and a standard deviation of 1.
        Formula: $X_{scaled} = (X - \mu) / \sigma$
        Tool: `sklearn.preprocessing.StandardScaler`
    * **Min-Max Scaling (Normalization):** Rescales data to a fixed range, usually 0 to 1.
        Formula: $X_{scaled} = (X - X_{min}) / (X_{max} - X_{min})$
        Tool: `sklearn.preprocessing.MinMaxScaler`

* **Applying Standardization:**
    Standardization is often a good default choice.
    **Important:** We should fit the scaler **only** on the training data and then use that *same* fitted scaler to transform both the training and the validation/test data. This prevents "data leakage" from the validation/test set into the training process. Since we haven't split yet, we'll demonstrate scaling on `df_model_ready` for now, but keep this principle in mind. Ideally, split first, then scale. For simplicity in this lesson flow, we scale before splitting, assuming `df_model_ready` represents our *potential* full training data.

    ```python
    from sklearn.preprocessing import StandardScaler

    if numerical_cols_for_scaling and 'df_model_ready' in locals() and not df_model_ready.empty: # Proceed only if there are columns to scale
        # Ensure all identified columns are actually present
        valid_cols_to_scale = [col for col in numerical_cols_for_scaling if col in df_model_ready.columns]
        if valid_cols_to_scale:
            scaler = StandardScaler()
            # Fit the scaler and transform the numerical columns
            df_model_ready[valid_cols_to_scale] = scaler.fit_transform(df_model_ready[valid_cols_to_scale])

            print("\nDataFrame after Scaling numerical features (head):")
            print(df_model_ready[valid_cols_to_scale].head())
            print("\nDescriptive stats of scaled features (should have mean ~0, std ~1):")
            print(df_model_ready[valid_cols_to_scale].describe())
        else:
            print("\nNone of the identified numerical columns for scaling were found in df_model_ready.")
    else:
        print("\nNo numerical columns were identified or available for scaling, or df_model_ready is not defined.")
    ```

---

### 3. Splitting Data into Training and Validation Sets (Practical - 15 min)

To evaluate our model's performance on unseen data, we need to split our dataset.

* **Why Split?**
    * **Training Set:** Used to train the machine learning model (i.e., learn the parameters).
    * **Validation Set (or Development Set):** Used to tune hyperparameters of the model and make decisions about the model (e.g., feature selection, model architecture). It provides an unbiased estimate of how the model performs on data it wasn't trained on.
    * **Test Set (from `test.csv`):** Used for a final, truly unbiased evaluation of the chosen model *after* all training and tuning are complete. We won't touch `test.csv` for labels until the very end.

* **Define Features (X) and Target (y):**
    * `X`: All columns in `df_model_ready` except `Survived`. These are our input features.
    * `y`: The `Survived` column. This is what we want to predict.

    ```python
    # (Assuming df_model_ready is available)
    if 'df_model_ready' in locals() and not df_model_ready.empty and 'Survived' in df_model_ready.columns:
        X = df_model_ready.drop('Survived', axis=1)
        y = df_model_ready['Survived']

        print("\nFeatures (X) for the model (head):")
        print(X.head())
        print("\nTarget (y) for the model (head):")
        print(y.head())
        print(f"\nShape of X: {X.shape}, Shape of y: {y.shape}")
    else:
        print("\n'Survived' column not found in df_model_ready or df_model_ready is empty. Cannot define X and y.")
        # Create dummy X and y if 'Survived' is missing for notebook continuity
        if 'df_model_ready' in locals() and not df_model_ready.empty:
            # Select all columns except a potential target if 'Survived' is missing
            potential_features = [col for col in df_model_ready.columns if col not in ['Survived']] # Example exclusion
            if potential_features:
                X = df_model_ready[potential_features]
                y = pd.Series(np.random.randint(0,2,size=len(df_model_ready)), name='Survived') # Dummy target
                print("--- Created dummy X and y as 'Survived' was not found for split. ---")
            else:
                X, y = pd.DataFrame(), pd.Series(name='Survived') # Empty placeholders
                print("--- df_model_ready is empty or has no suitable columns for X. ---")
        else: # If df_model_ready itself is not defined
            X, y = pd.DataFrame(), pd.Series(name='Survived') # Empty placeholders
            print("--- df_model_ready is not defined. Cannot define X and y. ---")
    ```

* **Performing the Split using `sklearn.model_selection.train_test_split`:**

    ```python
    from sklearn.model_selection import train_test_split

    if 'X' in locals() and 'y' in locals() and not X.empty and not y.empty: # Proceed only if X and y are defined
        # test_size: proportion of the dataset to include in the validation split (e.g., 0.2 for 20%)
        # random_state: ensures the split is the same every time you run the code (for reproducibility)
        # stratify=y: recommended for classification tasks. It ensures that the proportion of the target
        #             variable's classes is approximately the same in both training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,    # 20% for validation, 80% for training
            random_state=42,  # The answer to life, the universe, and everything - for consistency
            stratify=y        # Important for classification if y has more than 1 class and is not empty
        )

        print("\nShapes of the resulting data splits:")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        if not y.empty and y.nunique() > 1: # Stratification makes sense if there are classes to stratify
            print("\nProportion of target classes in original y:")
            print(y.value_counts(normalize=True))
            print("\nProportion of target classes in y_train:")
            print(y_train.value_counts(normalize=True))
            print("\nProportion of target classes in y_val (should be similar to y_train due to stratify):")
            print(y_val.value_counts(normalize=True))
        elif not y.empty:
             print("\nTarget variable 'y' has only one class or is empty, stratification may not be meaningful or applied.")
             print(y.value_counts(normalize=True))

    else:
        print("\nCannot perform train-test split as X or y is empty/undefined.")
        # Define empty placeholders if split failed, for subsequent lessons to run
        X_train, X_val, y_train, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='int')
    ```

---

### Wrap-up & Next Steps:

* **Recap:** Our data is now thoroughly preprocessed and ready for machine learning! We have:
    * Converted categorical features (`Sex`, `Embarked`) into numerical representations using one-hot encoding.
    * Dropped columns deemed less useful for an initial model (`Name`, `Ticket`, `PassengerId`, `SibSp`, `Parch`).
    * Scaled our numerical features (e.g., `Age`, `Fare`, `Pclass`, `FamilySize`) using Standardization.
    * Split the data into a training set (`X_train`, `y_train`) and a validation set (`X_val`, `y_val`).
* **Teaser for Lesson 4:** With our data fully preprocessed, we are finally ready to build and train our first predictive model! We'll start with a foundational algorithm for binary classification: **Logistic Regression**. We'll aim to build a simple version of this *from scratch* using NumPy to understand its inner workings before using library implementations.
