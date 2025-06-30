# Task 5: Decision Trees and Random Forests

## 🧠 Objective
Learn tree-based models for **classification and regression** using Decision Trees and Random Forests.

## 🛠️ Tools Used
- Python
- Scikit-learn
- Graphviz
- Matplotlib
- Seaborn

---

## 📌 Step-by-Step Process

### 1. Load Dataset
Used the **Heart Disease Dataset**. You can replace with any relevant binary classification dataset.

```python
import pandas as pd

data = pd.read_csv("heart.csv")  # Replace with your dataset
X = data.drop("target", axis=1)
y = data["target"]
```

---

### 2. Split Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 3. Train a Decision Tree Classifier and Visualize

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

plt.figure(figsize=(15, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.show()
```

---

### 4. Analyze Overfitting and Control Tree Depth

```python
from sklearn.metrics import accuracy_score

for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Depth {depth}: Train Acc = {train_acc:.2f}, Test Acc = {test_acc:.2f}")
```

---

### 5. Train a Random Forest and Compare Accuracy

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
```

---

### 6. Interpret Feature Importances

```python
import seaborn as sns

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_imp_df.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature")
plt.title("Feature Importance from Random Forest")
plt.show()
```

---

### 7. Evaluate Using Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X, y, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

---

## 📚 What You'll Learn
- Tree-based classification
- Model visualization using `plot_tree`
- Overfitting control via `max_depth`
- Feature importance analysis
- Random Forest vs Decision Tree
- Cross-validation evaluation

---

## 📎 Dataset
Heart Disease Dataset (or any other binary classification dataset)
