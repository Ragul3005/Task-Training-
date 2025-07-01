
# Task 6: K-Nearest Neighbors (KNN) Classification

## 🎯 Objective:
Understand and implement the **KNN algorithm** for classification problems using Python.

## 🛠 Tools Required:
- Scikit-learn
- Pandas
- Matplotlib

## ✅ Step-by-Step Guide:

### Step 1: Import Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

### Step 2: Load and Prepare Dataset
```python
data = load_iris()
X = data.data
y = data.target
```

### Step 3: Normalize Features
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 4: Split Dataset
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### Step 5: Train the KNN Model
```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

### Step 6: Evaluate the Model
```python
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
```

### Step 7: Visualize Confusion Matrix
```python
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

### Step 8: Visualize Decision Boundaries (Optional for 2D Data)
```python
from matplotlib.colors import ListedColormap

# Use only first 2 features for visualization
X_vis = X_scaled[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42
)

knn_vis = KNeighborsClassifier(n_neighbors=3)
knn_vis.fit(X_train_vis, y_train_vis)

h = .02
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), alpha=0.8)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['red', 'green', 'blue']))
plt.title("KNN Decision Boundaries (2 Features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

## 📌 Notes:
- You can try other datasets like wine, digits etc.
- Tune the `K` value for better performance.
- Use cross-validation for optimal results.
