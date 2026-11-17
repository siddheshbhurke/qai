import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("First five records:")
print(X.head())

# Step 2: Standardize the data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Step 3: Generate dendrogram
linked = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(linked)

plt.title("Agglomerative Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")

plt.show()

# Step 4: Apply Agglomerative Clustering
cluster_range = [2, 3, 4, 5, 6]

silhouette_scores = []

for k in cluster_range:

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print("\nNumber of Clusters:", k)
    print("Silhouette Score:", round(score, 4))

# Step 5: Find optimal number of clusters
optimal_k = cluster_range[
    np.argmax(silhouette_scores)
]

print("\nOptimal Number of Clusters:", optimal_k)

# Step 6: Train final clustering model
final_model = AgglomerativeClustering(
    n_clusters=optimal_k,
    linkage="ward"
)

final_labels = final_model.fit_predict(X_scaled)

# Step 7: Display cluster assignments
X["Cluster"] = final_labels

print("\nCluster Assignments:")
print(X.head(15))

print("\nNumber of Data Points in Each Cluster:")
print(X["Cluster"].value_counts())

# Step 8: Plot silhouette scores
plt.figure(figsize=(8, 5))

plt.bar(
    cluster_range,
    silhouette_scores
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

plt.title("Silhouette Score Comparison")

plt.show()

# Step 9: Visualize final clusters using first two features
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=final_labels,
    cmap="rainbow"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Agglomerative Hierarchical Clustering")

plt.show()