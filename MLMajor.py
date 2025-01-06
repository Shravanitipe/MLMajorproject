import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset (Ensure the file path is correct)
data = pd.read_csv("D:/mapython/spotify dataset.csv")

# Display the first few rows of the dataset
print("Dataset Overview:")
print(data.head())

# Data Preprocessing
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Handle missing values (drop rows with null values or consider imputation)
data.dropna(inplace=True)  # Alternatively, you could use data.fillna() for imputation

# Feature selection for clustering
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'tempo']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data Analysis and Visualization
# Correlation Matrix
correlation_matrix = data[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Popularity Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['track_popularity'], kde=True, bins=20, color='blue')
plt.title('Track Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

# Energy vs. Danceability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='energy', y='danceability', hue='playlist_genre', palette='Set1', s=100)
plt.title('Energy vs Danceability by Playlist Genre')
plt.xlabel('Energy')
plt.ylabel('Danceability')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Tempo Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['tempo'], kde=True, bins=20, color='green')
plt.title('Tempo Distribution')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Frequency')
plt.show()

# Clustering with KMeans
# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# Perform KMeans clustering with the optimal number of clusters (adjust as needed based on elbow plot)
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters if needed
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

# Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
plt.title('Clusters Visualization using PCA')
plt.show()

# Analyze Clusters by Playlist Genre (Only if 'playlist_genre' exists)
if 'playlist_genre' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='playlist_genre', style='Cluster', palette='Set1', s=100)
    plt.title('Clusters by Playlist Genre')
    plt.show()
else:
    print("\nColumn 'playlist_genre' not found. Skipping genre-based visualization.")

# Average Popularity by Playlist Genre
if 'playlist_genre' in data.columns:
    genre_popularity = data.groupby('playlist_genre')['track_popularity'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    genre_popularity.plot(kind='bar', color='orange')
    plt.title('Average Track Popularity by Playlist Genre')
    plt.xlabel('Playlist Genre')
    plt.ylabel('Average Popularity')
    plt.show()

# Duration vs. Popularity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='duration_ms', y='track_popularity', hue='playlist_genre', palette='viridis', s=100)
plt.title('Track Duration vs Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Build Recommendation System
def recommend_song(song_features):
    # Convert the song features into a DataFrame with feature names
    song_scaled = scaler.transform(pd.DataFrame([song_features], columns=features))
    cluster_label = kmeans.predict(song_scaled)
    recommended_songs = data[data['Cluster'] == cluster_label[0]]['track_name'].head(5)
    return recommended_songs

# Evaluate Clustering Performance
sil_score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score: {sil_score}")

