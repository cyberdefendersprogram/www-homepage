---
layout: workshop
order: 002
slug: exploratory-data-analysis
title: Exploratory Data Analysis
module_category: Introduction
workshop_name: zero-to-one-datascience-to-ai
date: 2025-05-23
---
## Lesson 2: Exploratory Data Analysis (EDA) & First Insights 📊

>
> Notebook on [Github]()
>

**Objective:**
* Learn techniques for handling missing data.
* Use visualizations (with Matplotlib and Seaborn) to find patterns and relationships between features and survival.
* Form initial hypotheses about which features are most important for predicting survival.
* Begin simple feature engineering.

---

### Recap of Lesson 1:
* Understood Data Science, AI, and Machine Learning basics.
* Introduced the Titanic dataset and the survival prediction challenge.
* Loaded the data using Pandas and performed initial inspection (`head()`, `info()`, `describe()`).

---

### 1. Handling Missing Data (Practical - 25 min)

Missing data is a common problem. `train_df.info()` from Lesson 1 showed us which columns have missing values (e.g., `Age`, `Cabin`, `Embarked`).

* **Identify Missing Values (Recap):**
    ```python
    # (Assuming train_df is loaded from Lesson 1)
    # import pandas as pd # Ensure pandas is imported
    # import numpy as np # Ensure numpy is imported
    # # Dummy train_df for standalone execution in Canvas if necessary
    # if 'train_df' not in locals():
    #     data = {'PassengerId': [1,2,3,4,5], 'Survived': [0,1,1,0,1], 'Pclass': [3,1,3,1,2],
    #             'Name': ['A','B','C','D','E'], 'Sex': ['male','female','female','male','female'],
    #             'Age': [22,38,26,35,np.nan], 'SibSp': [1,1,0,1,0], 'Parch': [0,0,0,1,0],
    #             'Ticket': ['t1','t2','t3','t4','t5'], 'Fare': [7.25,71.2,7.9,53.1,13],
    #             'Cabin': [np.nan,'C85',np.nan,'C123',np.nan], 'Embarked': ['S','C','S','S',np.nan]}
    #     train_df = pd.DataFrame(data)
    #     print("--- Loaded a dummy train_df for Lesson 2 demonstration. ---")

    print("Missing values before handling:")
    print(train_df.isnull().sum()) # .isnull() returns a boolean DataFrame, .sum() counts True values per column
    ```
    We typically see missing values in `Age`, `Cabin`, and a few in `Embarked`.

* **Strategies for Missing Data:**
    * **Dropping:**
        * Drop the whole row: If a row has too many missing values or if the missing value is in a crucial column for a specific analysis.
        * Drop the whole column: If a column has too many missing values to be useful (e.g., `Cabin`).
    * **Imputing:** Filling in the missing values.
        * For numerical data: Use mean, median (more robust to outliers), or mode.
        * For categorical data: Use mode (most frequent category).
        * More advanced: Use a model to predict missing values (e.g., k-NN imputation).

* **Applying Strategies:**

    * **`Cabin`:** This column has a lot of missing values. For a first pass, dropping it is often reasonable.
        ```python
        # We'll make a copy to keep the original train_df intact for now,
        # or you can operate directly on train_df if you prefer.
        df_processed = train_df.copy()
        df_processed.drop('Cabin', axis=1, inplace=True) # axis=1 means column, inplace=True modifies df_processed
        print("\n'Cabin' column dropped.")
        print("Missing values after dropping Cabin (if it existed):")
        print(df_processed.isnull().sum())
        ```
        *Alternative for `Cabin`: Later, we could create a feature like `HasCabin` (1 if a cabin is listed, 0 otherwise).*

    * **`Age`:** This numerical column has some missing values. Let's impute with the median age, which is less sensitive to outliers than the mean.
        ```python
        median_age = df_processed['Age'].median()
        df_processed.fillna({'Age': median_age}, inplace=True) #Making pandas 3.0 compatible.
        print(f"\nMissing 'Age' values imputed with median: {median_age}")
        print("Missing values in 'Age' after imputation:")
        print(df_processed['Age'].isnull().sum())
        ```

    * **`Embarked`:** This categorical column has only a couple of missing values. Let's impute with the mode (most frequent port).
        ```python
        mode_embarked = df_processed['Embarked'].mode()[0] # .mode() returns a Series, so we take the first element
        df_processed['Embarked'].fillna(mode_embarked, inplace=True)
        print(f"\nMissing 'Embarked' values imputed with mode: {mode_embarked}")

        print("Missing values in 'Embarked' after imputation:")
        print(df_processed['Embarked'].isnull().sum())

        print("\nFinal check for missing values in df_processed:")
        print(df_processed.isnull().sum())
        ```
        Now, our `df_processed` DataFrame should have no missing values in these key columns.

---

### 2. Visualizing Data to Understand Survival 📈 (Practical - 30 min)

Visualizations are powerful tools for uncovering relationships. We'll use `matplotlib` and `seaborn`.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set a nice style for the plots
sns.set(style="whitegrid")
```

*** Target Variable: Survived ***

Let's see the distribution of our target variable.

```python
# Ensure df_processed is available
if 'df_processed' in locals():
    plt.figure(figsize=(6,4))
    sns.countplot(x='Survived', data=df_processed)
    plt.title('Distribution of Survival (0 = Died, 1 = Survived)')
    plt.show() # Essential for displaying plots in script/Canvas environment
    print(df_processed['Survived'].value_counts(normalize=True)) # Percentage
```

Discussion: Is the dataset balanced or imbalanced regarding survival? (It's somewhat imbalanced, more died than survived)


* **Categorical Features vs. Survival:**
    `sns.barplot` or `sns.catplot` (with `kind="bar"`) are good for comparing a numerical value (like mean survival rate) across categories.

    * **`Sex` vs. `Survived`:**
        ```python
        if 'Sex' in df_processed.columns and 'Survived' in df_processed.columns:
            plt.figure(figsize=(6,4))
            sns.barplot(x='Sex', y='Survived', data=df_processed, errorbar=None) # errorbar=None for cleaner plot
            plt.title('Survival Rate by Sex')
            plt.ylabel('Survival Rate (0 to 1)')
            plt.show()
        else:
            print("'Sex' or 'Survived' column not in df_processed.")
        ```
        *Hypothesis: Females were more likely to survive.*

    * **`Pclass` vs. `Survived`:**
        ```python
        if 'Pclass' in df_processed.columns and 'Survived' in df_processed.columns:
            plt.figure(figsize=(6,4))
            sns.barplot(x='Pclass', y='Survived', data=df_processed, errorbar=None)
            plt.title('Survival Rate by Passenger Class')
            plt.ylabel('Survival Rate (0 to 1)')
            plt.show()
        else:
            print("'Pclass' or 'Survived' column not in df_processed.")
        ```
        *Hypothesis: Higher class passengers (Pclass=1) had a higher survival rate.*

    * **`Embarked` vs. `Survived`:**
        ```python
        if 'Embarked' in df_processed.columns and 'Survived' in df_processed.columns:
            plt.figure(figsize=(7,5))
            sns.barplot(x='Embarked', y='Survived', data=df_processed, errorbar=None)
            plt.title('Survival Rate by Port of Embarkation')
            plt.ylabel('Survival Rate (0 to 1)')
            plt.show()
        else:
            print("'Embarked' or 'Survived' column not in df_processed.")
        ```
        *Hypothesis: Passengers embarking from Cherbourg (C) might have had a slightly higher survival rate.*

* **Numerical Features vs. Survival:**
    * **`Age` vs. `Survived`:**
        `sns.histplot` with `hue='Survived'` can show distributions for different outcomes.
        `sns.kdeplot` can show smoothed distributions.
        `sns.boxplot` or `sns.violinplot` can compare distributions.
        ```python
        if 'Age' in df_processed.columns and 'Survived' in df_processed.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_processed, x='Age', hue='Survived', multiple="stack", kde=True)
            plt.title('Age Distribution by Survival Status')
            plt.xlabel('Age')
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Survived', y='Age', data=df_processed)
            plt.title('Age Distribution by Survival Status')
            plt.xticks([0,1], ['Died', 'Survived']) # Ensure correct labels for x-axis ticks
            plt.show()
        else:
            print("'Age' or 'Survived' column not in df_processed.")
        ```
        *Hypothesis: Younger children had a higher survival rate. Very old passengers might have had lower survival. The median age of survivors might be slightly lower.*

    * **`Fare` vs. `Survived`:**
        Fare is skewed, so a log transformation or careful binning might be useful for some plots, but let's try a boxplot and KDE first.
        ```python
        if 'Fare' in df_processed.columns and 'Survived' in df_processed.columns:
            plt.figure(figsize=(10, 6))
            # Using common_norm=False for KDE to show shapes independently
            sns.kdeplot(data=df_processed, x='Fare', hue='Survived', fill=True, common_norm=False, clip=(0,300)) # clip to see main part
            plt.title('Fare Distribution by Survival Status (KDE)')
            plt.xlabel('Fare')
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Survived', y='Fare', data=df_processed)
            plt.title('Fare Distribution by Survival Status')
            plt.xticks([0,1], ['Died', 'Survived']) # Ensure correct labels for x-axis ticks
            plt.ylim(0, 300) # Zoom in on the majority of fares for better visualization
            plt.show()
        else:
            print("'Fare' or 'Survived' column not in df_processed.")
        ```
        *Hypothesis: Passengers who paid higher fares were more likely to survive.*

---

### 3. Initial Feature Engineering 🛠️ (Practical - 15 min)
Feature engineering is the process of creating new features from existing ones to potentially improve model performance.

* **`FamilySize`**: Combine `SibSp` (siblings/spouses) and `Parch` (parents/children).
    ```python
    if 'SibSp' in df_processed.columns and 'Parch' in df_processed.columns:
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1 # +1 for the passenger themselves

        print("\nFirst 5 rows with 'FamilySize':")
        # Ensure 'Survived' column exists if you want to print it here, or handle its absence
        if 'Survived' in df_processed.columns:
            print(df_processed[['SibSp', 'Parch', 'FamilySize', 'Survived']].head())
        else:
            print(df_processed[['SibSp', 'Parch', 'FamilySize']].head())


        if 'Survived' in df_processed.columns:
            plt.figure(figsize=(10,6))
            sns.barplot(x='FamilySize', y='Survived', data=df_processed, errorbar=None)
            plt.title('Survival Rate by Family Size')
            plt.ylabel('Survival Rate (0 to 1)')
            plt.show()
    else:
        print("'SibSp' or 'Parch' column not found for creating 'FamilySize'.")
    ```
    *Discussion: How does family size relate to survival? (e.g., being alone or in a very large family might be detrimental. Small families might have fared better).*

* **`IsAlone`**: A binary feature indicating if the passenger was traveling alone (FamilySize = 1).
    ```python
    if 'FamilySize' in df_processed.columns:
        df_processed['IsAlone'] = 0 # Initialize with 0 (Not Alone)
        df_processed.loc[df_processed['FamilySize'] == 1, 'IsAlone'] = 1 # Set to 1 where FamilySize is 1

        print("\nFirst 5 rows with 'IsAlone':")
        if 'Survived' in df_processed.columns:
            print(df_processed[['FamilySize', 'IsAlone', 'Survived']].head())
        else:
            print(df_processed[['FamilySize', 'IsAlone']].head())


        if 'Survived' in df_processed.columns:
            plt.figure(figsize=(6,4))
            sns.barplot(x='IsAlone', y='Survived', data=df_processed, errorbar=None)
            plt.title('Survival Rate by IsAlone')
            plt.xticks([0,1], ['Not Alone', 'Alone']) # Ensure correct labels for x-axis ticks
            plt.ylabel('Survival Rate (0 to 1)')
            plt.show()
    else:
        print("'FamilySize' column not found for creating 'IsAlone'.")
    ```
    *Hypothesis: Passengers traveling alone might have had a lower survival rate than those in small families.*

* **Consider `Title` from `Name` (Conceptual for now, implement later if desired):**
    The `Name` column contains titles like "Mr.", "Mrs.", "Miss.", "Master.", etc. These could be powerful.
    * "Master." usually indicates a young male child.
    * "Miss." can indicate an unmarried female (potentially younger).
    * "Mrs." indicates a married female.
    * Other titles ("Dr.", "Rev.", "Col.") might indicate status or profession.
    Extracting these titles could be a good feature. We would use string processing (e.g., regular expressions) for this.

---

### Wrap-up & Next Steps:
* **Recap:** We've cleaned our data by handling missing values for `Age`, `Cabin`, and `Embarked`. We've used visualizations to explore how `Sex`, `Pclass`, `Embarked`, `Age`, and `Fare` relate to `Survived`. We've also started feature engineering by creating `FamilySize` and `IsAlone`.
* **Teaser for Lesson 3:** Next, we'll focus on **Data Preprocessing for Machine Learning**. This critical step involves:
    * Converting categorical features (like `Sex`, `Embarked`, and any new title feature) into numerical representations that machine learning models can understand (e.g., one-hot encoding).
    * Potentially scaling numerical features (like `Age`, `Fare`, `FamilySize`) so they are on a similar range.
    * Splitting our processed data into training and validation sets to prepare for model building and evaluation.
