# ML-Assignment-5---Clustering-Algorithm
This  Assignment explains about the Clustering algorithm in unsupervised machine learning 
# Key components to be fulfilled :
# 1. Loading and Preprocessing 
Load the Iris dataset from sklearn.
Drop the species column since this is a clustering problem.

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn import datasets
import pandas as pd
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# Load the Iris dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display the first few rows
data.head()

data.shape
data.info()
null_values=data.isnull().sum()
print("The null values in each column:\n",null_values)


data.duplicated().sum()
data=data.drop_duplicates()
data.duplicated().sum()
There is only one duplicate value, removed using drop function.
# 2.Clustering Algorithm Implementation 
Implement the following two clustering algorithms:
# A) KMeans Clustering 
-- Provide a brief description of how KMeans clustering works.
-- Explain why KMeans clustering might be suitable for the Iris dataset.
-- Apply KMeans clustering to the preprocessed Iris dataset and visualize the clusters.
# KMeans Clustering 
KMeans is an unsupervised machine learning algorithm designed for clustering, which involves partitioning data into distinct groups. 
The process begins with the random initialization of k centroids. Each data point is then assigned to the nearest centroid based on Euclidean distance. 
The centroids are subsequently updated by calculating the mean position of all points within each cluster. 
This iterative process continues until the centroids stabilize, meaning they no longer change significantly, or a predefined stopping condition is met.

The Iris dataset naturally forms three distinct clusters that correspond to the three species: Setosa, Versicolor, and Virginica. 
KMeans clustering is particularly effective for this dataset because it performs well when clusters are well-separated in the feature space. 
With only four numerical features, the dataset provides a suitable structure for KMeans to efficiently identify patterns and group similar data points into meaningful clusters.

scaler=StandardScaler()
scaled_features=scaler.fit_transform(data)

# Elbow Method to Determine Optimal Number of Clusters
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Choose Optimal Number of Clusters 
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

cluster_labels=kmeans.fit_predict(scaled_features)
data["Cluster"]=kmeans.labels_
data

sns.scatterplot(data=data,x="sepal length (cm)",y="petal length (cm)",hue="Cluster")
data["Cluster"].unique()
# Evaluate Model Performance
# 1. Silhouette Score
sil_score = silhouette_score(scaled_features, cluster_labels)
print(f"Silhouette Score: {sil_score:.2f}")

# 2. Davies-Bouldin Index
db_score = davies_bouldin_score(scaled_features, cluster_labels)
print(f"Davies-Bouldin Index: {db_score:.2f}")

from sklearn.metrics import calinski_harabasz_score

# Calculate Calinski-Harabasz Index
ch_score = calinski_harabasz_score(scaled_features, cluster_labels)
print(f"Calinski-Harabasz Index: {ch_score:.2f}")

# B) Hierarchical Clustering 
-- Provide a brief description of how Hierarchical clustering works.
-- Explain why Hierarchical clustering might be suitable for the Iris dataset.
-- Apply Hierarchical clustering to the preprocessed Iris dataset and visualize the clusters.

# Hierarchical clustering
Hierarchical clustering is an unsupervised machine learning technique that groups data points into a hierarchy of clusters. 
It operates in two ways: agglomerative (bottom-up) and divisive (top-down). 
The agglomerative approach, which is more commonly used, begins by treating each data point as its own cluster and then progressively merging the closest clusters based on a similarity measure, such as Euclidean distance. 
This process continues until all points are merged into a single cluster or a stopping criterion is met. 
In contrast, the divisive approach starts with all data points in a single cluster and recursively splits them into smaller groups. 
The results of hierarchical clustering are typically represented using a dendrogram, a tree-like diagram that visually illustrates how clusters are formed and their relationships.

Hierarchical clustering is suitable for the Iris dataset as it can reveal natural relationships between its three species based on feature similarities. 
Unlike K-Means, it does not require a predefined number of clusters and provides an intuitive visualization through a dendrogram. 
While hierarchical clustering is computationally expensive for large datasets, the small size of the Iris dataset makes it efficient and interpretable, aiding in understanding species groupings.

clusters=AgglomerativeClustering(n_clusters=3)
hierarchical_labels=clusters.fit_predict(scaled_features)

# Plotting a Dendrogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage (scaled_features, method='ward'))

clusters=AgglomerativeClustering()
clusters.fit_predict(data)

data['cluster2']=clusters.fit_predict(data)
data
data["cluster2"].unique()
uniquecolor=set(dend['color_list'])
uniquecolor
optimal_number_of_clusters=len(uniquecolor)-1
optimal_number_of_clusters

# Evaluate Model Performance
#1.silhouette_score
sil_score=silhouette_score(scaled_features,hierarchical_labels)
print(f"silhouette_score:{sil_score:.2f}")

#2.Davies-Bouldin Index
db_score=davies_bouldin_score(scaled_features,hierarchical_labels)
print(f"Davies-Bouldin Index:{db_score:.2f}")

#3.Calinski-Harabasz Index
ch_score=calinski_harabasz_score(scaled_features,hierarchical_labels)
print(f"Calinski-Harabasz Index:{ch_score:.2f}")

--The End--
