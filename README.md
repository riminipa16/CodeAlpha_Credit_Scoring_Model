---

# 📊 Credit Scoring Model

Predicting Loan Default Using Machine Learning

This project builds a **Credit Scoring Model** to predict whether an individual is likely to default on a loan based on past financial history. The project includes data preprocessing, feature engineering, exploratory data analysis (EDA), and model building using Logistic Regression, Decision Trees, and Random Forest classifiers.

---

## 🚀 Objective

To develop a machine learning model that predicts an individual's **creditworthiness** using financial data such as income, loan amount, credit history, and loan terms.

---

## 🧠 Approach

Algorithms used:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**

---

## ⭐ Key Features

* Missing value handling
* Data preprocessing & scaling
* Feature engineering
* Data visualization using Seaborn & Matplotlib
* Classification metrics: Precision, Recall, F1-Score, ROC-AUC
* Model comparison

---

# 📂 Dataset Overview
Dataset Link:https://drive.google.com/file/d/1rliGZ-tW5xV0SvFsZ6lyQccwWPnhe2wG/view

The dataset contains **1248 rows** and **5 columns**:

| Column           | Description                                        |
| ---------------- | -------------------------------------------------- |
| `income`         | Annual income of borrower                          |
| `loan_amount`    | Loan amount requested                              |
| `term`           | Loan tenure (36 or 60 months)                      |
| `credit_history` | Indicates good credit history (1 = good)           |
| `defaulted`      | Target variable (1 = defaulted, 0 = not defaulted) |

---

# 🧰 Importing Required Libraries

```python
from matplotlib import axes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

# 📝 Step 1: Load Dataset

```python
df = pd.read_csv('/content/loan_data_1248_with_missing.csv')
df.head(10)
```

---

# 📐 Step 2: Data Understanding

```python
df.shape
df.columns.tolist()
df.dtypes
```

---

# 🧹 Step 3: Data Preprocessing

### Check Missing Values

```python
df.isnull().sum()
```

### Impute Missing Values

```python
df.fillna({'income': df['income'].median()}, inplace=True)
df.fillna({'loan_amount': df['loan_amount'].median()}, inplace=True)
df.fillna({'credit_history': df['credit_history'].mode()[0]}, inplace=True)
```

---

# 📊 Step 4: Data Visualization

### Income & Loan Amount Distribution

```python
sns.histplot(df['income'], kde=True, bins=30)
sns.histplot(df['loan_amount'], kde=True, bins=30)
```

### Loan Term Distribution

```python
sns.countplot(x='term', data=df)
```

### Credit History vs Default Status

```python
sns.countplot(x='credit_history', hue='defaulted', data=df)
```

### Correlation Heatmap

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```

---

# 🏗 Step 5: Feature Engineering

### Binary Term Feature

```python
df['term_binary'] = df['term'].apply(lambda x: 1 if x == 60 else 0)
```

### Log Transformation

```python
df['log_income'] = np.log(df['income'])
df['log_loan_amount'] = np.log(df['loan_amount'])
```

### Select Features

```python
features = ['log_income', 'log_loan_amount', 'credit_history']
target = 'defaulted'
```

---

# 🤖 Step 6: Model Training

### Scaling Numeric Features

```python
scaler = StandardScaler()
scale_features = ['log_income', 'log_loan_amount']
df[scale_features] = scaler.fit_transform(df[scale_features])
```

### Train-Test Split

```python
x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

---

# 🏆 Model Building & Evaluation

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"\nModel: {name}")
    pipeline = Pipeline([('classifier', model)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
```

---

# 📈 Model Performance Summary

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | **0.652** |
| Decision Tree       | **0.540** |
| Random Forest       | **0.608** |

✔ Logistic Regression performed best overall
✔ Random Forest performed better for non-linearity
✔ More data & hyperparameter tuning can improve performance

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---



