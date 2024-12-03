import pandas as pd
import random

# Load the clustered dataset
input_file = "dynamic_document_clusters.csv"
clustered_docs_df = pd.read_csv(input_file)

# Get the unique clusters
unique_clusters = clustered_docs_df['Cluster'].unique()
print(f"Available clusters: {unique_clusters}")

# Randomly select a cluster
random_cluster = random.choice(unique_clusters)
print(f"Randomly selected cluster: {random_cluster}")

# Filter the DataFrame for the selected cluster
sampled_cluster_df = clustered_docs_df[clustered_docs_df['Cluster'] == random_cluster]

# Extract unique document numbers (Index or row numbers)
unique_document_numbers = sampled_cluster_df.index.tolist()

# Output the results to a text file
output_file = f"sampled_cluster_{random_cluster}.txt"
with open(output_file, "w") as f:
    # Write the unique document numbers as the first line
    f.write("Unique Document Numbers: " + ", ".join(map(str, unique_document_numbers)) + "\n\n")

    # Write each document with its content
    for idx, row in sampled_cluster_df.iterrows():
        f.write(f"Document {idx}: {row['Content']}\n\n")

print(f"Sampled cluster saved to '{output_file}'")
