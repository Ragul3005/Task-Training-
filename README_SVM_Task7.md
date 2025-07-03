
# Task 7: Support Vector Machines (SVM)

## 🎯 Objective
Use SVMs for linear and non-linear classification.

## 🛠️ Tools Required
- Scikit-learn
- NumPy
- Matplotlib

## 📂 Dataset
You can use any dataset relevant to the task, e.g., Breast Cancer Dataset.
Download: [Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

## 🔧 Step-by-Step Procedure

### ✅ Step 1: Load and Prepare the Dataset
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### ✅ Step 2: Train SVM (Linear and RBF Kernel)
```python
from sklearn.svm import SVC

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
```

### ✅ Step 3: Visualize Decision Boundary (2D Data)
```python
import matplotlib.pyplot as plt

X_2d = X[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)
model_2d = SVC(kernel='linear')
model_2d.fit(X_train_2d, y_train_2d)

def plot_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(model_2d, X_test_2d, y_test_2d)
```

### ✅ Step 4: Tune Hyperparameters (C, gamma)
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
```

### ✅ Step 5: Evaluate Model with Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
print("Cross-validation accuracy scores:", scores)
print("Average CV accuracy:", scores.mean())
```

---

## ✅ Final Notes
- Make sure to scale your features before using SVM.
- Hyperparameter tuning significantly improves model performance.
