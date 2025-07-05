
# Task 8: Clustering with K-Means

## 🎯 Objective
Perform unsupervised learning using **K-Means Clustering**.

## 🛠️ Tools Required
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- NumPy (optional)
- sklearn.metrics

---

## ✅ Step-by-Step Procedure

### 1. Load and Visualize the Dataset
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.info())
```

### 2. Preprocess Data (Select Features)
```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

### 3. Use the Elbow Method to Find Optimal K
```python
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
```

### 4. Fit K-Means and Assign Cluster Labels
```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
```

### 5. Visualize Clusters
```python
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', data=df)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

### 6. Evaluate Clustering (Silhouette Score)
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, df['Cluster'])
print(f'Silhouette Score: {score:.2f}')
```

---

## 📁 Dataset
Use any dataset relevant to the task, such as the **Mall Customer Segmentation Dataset**.

📥 [Download Dataset](#)

---

## 📌 Notes
- Normalize data if feature scales vary greatly.
- Try PCA if working with more than 2 features.
