
# 🚢 Titanic Dataset – Data Cleaning & Preprocessing

## 📊 Overview
This project demonstrates how to clean and preprocess raw data using the **Titanic dataset**. The goal is to prepare the data for machine learning by handling missing values, encoding categorical variables, scaling numerical features, and removing outliers.

---

## 🎯 Objectives
- Understand and explore the structure of a dataset.
- Clean missing and invalid data.
- Convert categorical features into numerical formats.
- Normalize/standardize numerical features.
- Detect and remove outliers.
- Prepare the dataset for model training.

---

## 🛠️ Tools and Libraries Used
- **Python**
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Preprocessing tools

---

## 📁 Files
- `titanic.csv` – Raw dataset (download from [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data))
- `preprocess.py` or `preprocessing.ipynb` – Python script or notebook containing the data cleaning steps.

---

## 🧪 Steps Performed

### 1. Importing the Dataset
```python
import pandas as pd
df = pd.read_csv("titanic.csv")
```

### 2. Exploring Data
```python
df.head()
df.info()
df.describe()
df.isnull().sum()
```

### 3. Handling Missing Values
- Fill numeric missing values using **median**
- Fill categorical missing values using **mode**
```python
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

### 4. Encoding Categorical Variables
```python
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
```

### 5. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

### 6. Outlier Detection and Removal
```python
import seaborn as sns
sns.boxplot(x=df['Fare'])

# Remove outliers using IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]
```

---

## ✅ Final Outcome
A clean and ready-to-use dataset suitable for training a machine learning model.

---

## 📘 What You'll Learn
- Data exploration
- Null handling
- Feature encoding
- Feature scaling
- Outlier removal

---

## 🚀 Next Steps
- Use this cleaned dataset for training classification models like:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Try pipeline-based preprocessing using `sklearn.pipeline`.

---

## 📩 Contact
For questions or suggestions, feel free to open an issue or reach out.
