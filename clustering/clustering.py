import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt


def cluster_documents():
    df = pd.read_csv('../data_cleaning/dataset_final.csv')
    n_clusters = len(df) // 15
    print(f"Number of documents: {len(df)}")
    print(f"Using {n_clusters} clusters")

    features = np.vstack([
        np.array(ast.literal_eval(feat))
        for feat in tqdm(df['enhanced_features'])
    ])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(features)

    em = GaussianMixture(
        n_components=n_clusters,
        random_state=42,
        covariance_type='diag',
        max_iter=100,
        init_params='kmeans',
        verbose=1
    )
    em_clusters = em.fit_predict(features)

    output_df = df[['Chunk ID', 'Document ID', 'Page Content', 'Entities', 'High Confidence Entities']].copy()
    output_df['KMeans_Cluster'] = kmeans_clusters
    output_df['EM_Cluster'] = em_clusters
    output_df.to_csv('dataset_clustered.csv', index=False)

    return output_df, kmeans_clusters, em_clusters


if __name__ == "__main__":
    try:
        output_df, kmeans_clusters, em_clusters = cluster_documents()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # K-means cluster sizes
        pd.Series(kmeans_clusters).value_counts().sort_index().plot(
            kind='bar', ax=ax1, title='K-means Cluster Sizes')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Documents')

        # EM cluster sizes
        pd.Series(em_clusters).value_counts().sort_index().plot(
            kind='bar', ax=ax2, title='EM Cluster Sizes')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Documents')

        plt.tight_layout()
        plt.savefig('cluster_distributions.png')
        plt.close()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise