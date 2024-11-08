import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Step 1: Load the DataFrame from CSV
df = pd.read_csv("chunk_metadata_embeddings.csv")

# The column 'chunk_embedding_vec' is stored as strings, convert them to numpy arrays
def parse_embedding(embedding_str):
    embedding_str = embedding_str.strip("[]")  # Remove brackets
    return np.array(list(map(float, embedding_str.split())))  # Convert to float array

df['chunk_embedding_vec'] = df['chunk_embedding_vec'].apply(parse_embedding)

# Step 2: Prepare the embeddings matrix for clustering
embedding_matrix = np.vstack(df['chunk_embedding_vec'].values)

# Step 3: Determine an appropriate value for K for K-means
# Since we have around 30K chunks, we can start with k around sqrt(n)
n_chunks = len(df)
k = int(np.sqrt(n_chunks))  # A heuristic value for initial clusters

# Step 4: Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(embedding_matrix)

# Step 5: Sort by cluster number
df['cluster'] = df['cluster'] + 1  # To make cluster numbers start from 1 instead of 0
df_sorted = df.sort_values(by='cluster')

# Step 6: Save the results
df_sorted.to_csv("chunk_metadata_clustered.csv", index=False)

# Optional: Display the first few rows of the DataFrame to verify
print(df_sorted.head())
