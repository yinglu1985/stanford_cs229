import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer("thenlper/gte-small")

# Step 1: Read file and parse based on single quotes
data = []
with open('processed_data.csv', 'r') as file:
    content = file.read()

# Function to parse entries based on single quotes and remove unwanted characters
def extract_entries(content):
    entries = []
    raw_entries = content.split("'")  # Split based on single quotes
    entry = []

    for item in raw_entries:
        item = item.strip().replace("\n", "")  # Remove newlines for consistency
        if item:  # Only process non-empty items
            entry.append(item)
            if len(entry) == 4:  # Ensure each entry has exactly 4 parts
                entries.append(entry)
                entry = []  # Reset for next entry

    return entries

# Extract all entries based on single quotes
entries = extract_entries(content)

# Process each entry into a structured format
for entry in entries:
    try:
        # Convert each field, skip if not convertible
        parsed_entry = {
            'chunk_id': int(float(entry[0].strip())),     # Convert chunk_id
            'doc_id': int(float(entry[1].strip())),       # Convert doc_id
            'chunk_seq_id': int(float(entry[2].strip())), # Convert chunk_seq_id
            'chunk_text': entry[3].strip()                # Keep chunk_text as string
        }
        data.append(parsed_entry)
    except ValueError as e:
        print(f"Skipping entry due to conversion issue: {entry}. Error: {e}")

# Convert to DataFrame for structured manipulation
df = pd.DataFrame(data)

# Step 2: Generate embeddings for each chunk_text
df['chunk_embedding_vec'] = df['chunk_text'].apply(lambda x: embedding_model.encode(x))

# Prepare data for clustering
embedding_matrix = np.vstack(df['chunk_embedding_vec'].values)
chunk_ids = df['chunk_id'].tolist()
doc_ids = df['doc_id'].tolist()

# Step 3: Apply K-means clustering
n_clusters = 5  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(embedding_matrix)
labels = kmeans.labels_

# Organize clustered data by cluster label
clustered_data = {i: {'chunk_id_list': [], 'doc_id_list': []} for i in range(n_clusters)}
for i, label in enumerate(labels):
    clustered_data[label]['chunk_id_list'].append(chunk_ids[i])
    clustered_data[label]['doc_id_list'].append(doc_ids[i])

# Step 4: Save clustered data to 'data_2.txt' without brackets
with open('data_2.txt', 'w') as f:
    for cluster_id, info in clustered_data.items():
        f.write(f"cluster_id: {cluster_id}, chunk_id_list: {', '.join(map(str, info['chunk_id_list']))}, doc_id_list: {', '.join(map(str, info['doc_id_list']))}\n")

print("Clustered data saved to 'data_2.txt'")
