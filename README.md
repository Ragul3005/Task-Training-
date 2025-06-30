# Task 4: Classification with Logistic Regression

## 🧠 Objective
Build a **binary classifier** using **Logistic Regression** to predict outcomes based on input features.

## 🛠️ Tools Used
- Python
- Scikit-learn
- Pandas
- Matplotlib

---

## 📌 Step-by-Step Process

### 1. Load Dataset
Used the **Breast Cancer Wisconsin Dataset** from Scikit-learn.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
```

---

### 2. Train/Test Split and Feature Standardization

- Data split into training and testing sets.
- Features standardized using `StandardScaler`.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 3. Fit a Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

### 4. Evaluate the Model

Used **Confusion Matrix**, **Classification Report**, and **ROC-AUC Curve**.

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

### 5. Tune Threshold and Explain Sigmoid Function

#### Threshold Adjustment:
```python
threshold = 0.4  # example
y_pred_new = (y_prob >= threshold).astype(int)
print(confusion_matrix(y_test, y_pred_new))
```

#### Sigmoid Function:
Logistic regression uses the sigmoid function to output probabilities:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

This maps any real-valued number into a probability between 0 and 1.

---

## 📚 What You'll Learn
- Binary classification with Logistic Regression
- Evaluation metrics: Confusion Matrix, Precision, Recall, ROC-AUC
- Threshold tuning and Sigmoid curve

---

## 📎 Dataset
Breast Cancer Wisconsin Dataset (available via `sklearn.datasets.load_breast_cancer`)
