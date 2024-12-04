import pandas as pd
import random

# Load the dataset with clusters
input_file = "dynamic_document_clusters.csv"
clustered_docs_df = pd.read_csv(input_file)

# Select a random cluster
unique_clusters = clustered_docs_df['Cluster'].unique()
random_cluster = random.choice(unique_clusters)
print(f"Randomly selected cluster: {random_cluster}")

# Extract documents in selected cluster
sampled_cluster_df = clustered_docs_df[clustered_docs_df['Cluster'] == random_cluster]
unique_document_numbers = sampled_cluster_df.index.tolist()

# Output the results to a text file
output_file = f"sampled_cluster_{random_cluster}.txt"
with open(output_file, "w") as f:
    # Add unique document numbers
    f.write("Unique Document Numbers: " + ", ".join(map(str, unique_document_numbers)) + "\n\n")

    # Add document content
    for idx, row in sampled_cluster_df.iterrows():
        f.write(f"Document {idx}: {row['Content']}\n\n")

print(f"Sampled cluster saved to '{output_file}'")
