
# 📊 Task 2: Exploratory Data Analysis (EDA)

## 🎯 Objective
To explore and understand the structure, patterns, and key relationships in the dataset using **statistics** and **visualizations**.  
This step helps prepare the data for modeling and uncover hidden insights.

---

## 🛠️ Tools & Libraries Used

- **Python 3**
- [Pandas](https://pandas.pydata.org/) – Data manipulation
- [Matplotlib](https://matplotlib.org/) – Static visualizations
- [Seaborn](https://seaborn.pydata.org/) – Statistical plots
- [Plotly (optional)](https://plotly.com/python/) – Interactive charts

---

## 🗂️ Dataset
This task uses the **Titanic Dataset**, which includes passenger details like:
- Age, Gender, Class
- Fare, Embarkation Port
- Survival (target)

📥 [Download Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

---

## 🧭 EDA Process Overview

### 1. Understand the Dataset
- Load and preview the data
- Check shape, column types, and missing values

### 2. Summary Statistics
- Use `.describe()`, `.info()`, and `.isnull().sum()` for:
  - Mean, median, standard deviation
  - Missing data
  - Data types

### 3. Univariate Analysis
- Distribution of numeric features (Age, Fare)
- Count plots of categorical features (Sex, Pclass, Embarked)

### 4. Bivariate Analysis
- Survival rate vs gender
- Survival vs class (Pclass)
- Correlation matrix and heatmap

### 5. Outlier Detection
- Boxplots for numerical data
- Identify unusually high or low values

### 6. Basic Inference
- Who had higher chances of survival?
- Which features might be useful for prediction?

---

## 📷 Example Visuals

- Age distribution using `sns.histplot()`
- Survival count using `sns.countplot()`
- Correlation heatmap using `sns.heatmap()`

---

## 📌 Key Insights (Titanic Sample)
- Women and children had higher survival rates.
- 1st class passengers were more likely to survive.
- High fare may correlate with survival.

---

## 🗃️ Output
- Cleaned and analyzed dataset
- Visualizations
- Observations to guide modeling (in Task 3)

---

## 📁 File Structure Example
```
Task2_EDA/
├── Simple_Titanic_EDA.ipynb
├── titanic.csv
├── README.md
```
