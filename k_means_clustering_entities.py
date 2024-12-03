from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Load the dataset
dataset_with_entities_df = pd.read_csv('dataset_with_entities_df_sample.csv')

# Convert the `Entities` column back to lists if stored as strings
dataset_with_entities_df['Entities'] = dataset_with_entities_df['Entities'].apply(eval)

# Join the entities list into a single string for vectorization
dataset_with_entities_df['Entities_String'] = dataset_with_entities_df['Entities'].apply(lambda x: ' '.join([entity['word'] for entity in x]))

# Use TF-IDF to convert text data into numerical feature vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset_with_entities_df['Entities_String'])

# Apply K-means clustering
num_clusters = 5  # Adjust this number based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
dataset_with_entities_df['Cluster'] = kmeans.fit_predict(X)

# Save the dataset with cluster labels
output_file_with_clusters = 'dataset_with_clusters.csv'
dataset_with_entities_df.to_csv(output_file_with_clusters, index=False)

print(f"Clustering complete. Results saved to '{output_file_with_clusters}'")
