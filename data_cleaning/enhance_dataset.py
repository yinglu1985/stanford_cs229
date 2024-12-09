import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast


def string_to_array(vector_string):
    vector_string = vector_string.replace('\n', ' ').strip()
    return np.array([float(x) for x in vector_string.strip('[]').split()])


def enhance_dataset():
    df = pd.read_csv('dataset.csv')

    print("Processing features...")
    chunk_vectors = []
    for vec in tqdm(df['Chunk Vector'], desc="Converting vectors"):
        arr = string_to_array(vec)
        if arr is not None:
            chunk_vectors.append(arr)
    chunk_vectors = np.vstack(chunk_vectors)

    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_features = tfidf.fit_transform(df['Page Content']).toarray()

    print("Normalizing features...")
    scaler = StandardScaler()
    normalized_vectors = scaler.fit_transform(chunk_vectors)

    print("Applying dimension reduction...")

    pca_analyzer = PCA(n_components=0.99)
    pca_analyzer.fit(normalized_vectors)

    cumsum = np.cumsum(pca_analyzer.explained_variance_ratio_)

    thresholds = [0.5, 0.7, 0.8, 0.9]
    n_components_for_threshold = {
        f"{int(threshold * 100)}%": np.argmax(cumsum >= threshold) + 1
        for threshold in thresholds
    }

    print("\nPCA Analysis:")
    for threshold, n_comp in n_components_for_threshold.items():
        print(f"Components needed for {threshold} variance: {n_comp}")

    optimal_components = n_components_for_threshold["80%"]
    print(f"\nSelected {optimal_components} components for final PCA")


    pca = PCA(n_components=optimal_components)
    pca_features = pca.fit_transform(normalized_vectors)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'b-', label='Cumulative Explained Variance')
    plt.plot(range(1, len(cumsum) + 1), pca_analyzer.explained_variance_ratio_, 'r-',
             label='Individual Explained Variance')

    for threshold in thresholds:
        n_comp = n_components_for_threshold[f"{int(threshold * 100)}%"]
        plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.3)
        plt.axvline(x=n_comp, color='g', linestyle='--', alpha=0.3)
        plt.text(n_comp + 1, threshold, f'{int(threshold * 100)}% ({n_comp} components)',
                 verticalalignment='bottom')

    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_variance_analysis.png')
    plt.close()

    print("\nCreating enhanced feature matrix...")
    enhanced_features = np.hstack([
        chunk_vectors,
        pca_features,
        tfidf_features
    ])

    print("Saving enhanced dataset...")
    df['pca_features'] = [str(feat.tolist()) for feat in pca_features]
    df['tfidf_features'] = [str(feat.tolist()) for feat in tfidf_features]
    df['enhanced_features'] = [str(feat.tolist()) for feat in enhanced_features]

    feature_dims = {
        'original_dim': chunk_vectors.shape[1],
        'pca_dim': pca_features.shape[1],
        'tfidf_dim': tfidf_features.shape[1],
        'total_enhanced_dim': enhanced_features.shape[1]
    }

    print("\nFeature Dimensions:")
    for key, value in feature_dims.items():
        print(f"{key}: {value}")

    df.to_csv('dataset_final.csv', index=False)

    plt.figure(figsize=(12, 6))
    component_sizes = [
        feature_dims['original_dim'],
        feature_dims['pca_dim'],
        feature_dims['tfidf_dim']
    ]
    labels = ['Original Vectors', 'PCA Features', 'TF-IDF Features']
    plt.bar(labels, component_sizes)
    plt.title('Feature Dimensions by Component')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Dimensions')
    plt.tight_layout()
    plt.savefig('feature_dimensions.png')
    plt.close()

    return df, enhanced_features, feature_dims


if __name__ == "__main__":
    try:
        df, enhanced_features, feature_dims = enhance_dataset()

        print("\nDataset Summary:")
        print(f"Number of documents: {len(df)}")
        print(f"Total features in enhanced representation: {feature_dims['total_enhanced_dim']}")
        print("\nFeature composition:")
        print(f"- Original embedding features: {feature_dims['original_dim']}")
        print(f"- PCA features: {feature_dims['pca_dim']}")
        print(f"- TF-IDF features: {feature_dims['tfidf_dim']}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
