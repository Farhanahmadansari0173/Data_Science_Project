"""
Script 3: Unsupervised Machine Learning (K-Means & PCA)

Instead of relying on human bias to group customers, I use K-Means Clustering 
to algorithmically discover behavioral segments based on RFM features.
I also use Principal Component Analysis (PCA) to visualize these high-dimensional clusters.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROCESSED_DIR = "data/processed"
FIGURES_DIR = "reports/figures"

def run_clustering():
    print("Running Unsupervised ML (K-Means Clustering)...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    rfm = pd.read_csv(os.path.join(PROCESSED_DIR, "rfm_data.csv"))
    
    # We must scale the data because Monetary is in hundreds but Frequency is single digits
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # 1. Elbow Method to find optimal K
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertia.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, "kmeans_elbow_curve.png"), dpi=300)
    plt.close()
    
    # Based on standard RFM, k=4 usually provides distinct business segments
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 2. Visualize the RFM distributions across clusters
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(x='Cluster', y='Recency', data=rfm, ax=axes[0], palette='Set2')
    sns.boxplot(x='Cluster', y='Frequency', data=rfm, ax=axes[1], palette='Set2')
    sns.boxplot(x='Cluster', y='Monetary', data=rfm, ax=axes[2], palette='Set2')
    plt.suptitle('RFM Distribution by K-Means Cluster')
    plt.savefig(os.path.join(FIGURES_DIR, "cluster_rfm_boxplots.png"), dpi=300)
    plt.close()
    
    # 3. Principal Component Analysis (PCA) for 2D Visualization
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = rfm_pca[:, 0]
    rfm['PCA2'] = rfm_pca[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=rfm, palette='Set2', alpha=0.7)
    plt.title('2D PCA Projection of Customer Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(FIGURES_DIR, "pca_clusters_2d.png"), dpi=300)
    plt.close()
    
    print("Unsupervised ML complete. Saved clustering figures.")

if __name__ == "__main__":
    run_clustering()
