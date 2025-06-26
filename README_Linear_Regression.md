
# 📈 Task 3: Linear Regression

## 🎯 Objective
To implement and understand **Simple** and **Multiple Linear Regression** using Python.

This task involves applying linear regression to a real-world dataset (e.g., House Price Prediction) to predict a target variable based on one or more features.

---

## 🧰 Tools & Libraries
- **Python 3.x**
- **Pandas** – For data loading and preprocessing
- **Scikit-learn** – For building and evaluating models
- **Matplotlib** – For plotting and visualizing results

---

## 📁 Dataset
You can use any regression-friendly dataset.  
Example: **House Price Prediction Dataset**  
📥 [Click here to download dataset](#) <!-- Replace # with actual link if available -->

---

## 🔁 Step-by-Step Process

### 1. Import and Preprocess the Dataset
- Load dataset using Pandas.
- Check for and handle missing/null values.
- Encode categorical variables (if any).
- Feature selection as needed.

### 2. Train-Test Split
- Use `train_test_split()` from `sklearn.model_selection` to split data.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Fit the Linear Regression Model
- Use `LinearRegression()` from `sklearn.linear_model`.
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### 4. Evaluate the Model
- Use these performance metrics:
  - **MAE** (Mean Absolute Error)
  - **MSE** (Mean Squared Error)
  - **R² Score** (Goodness of Fit)
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

### 5. Plot Regression Line (For Simple Regression)
```python
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Line')
plt.show()
```

---

## 📊 Example Output
```text
MAE: 2.31
MSE: 8.45
R² Score: 0.87
```

---

## 📘 What You'll Learn
- Core concepts of **regression modeling**
- How to evaluate a model using metrics like MAE, MSE, and R²
- How to **interpret model coefficients** and visualize predictions

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author
Developed by [Your Name Here] – feel free to contribute or raise issues!
