import pandas as pd
import random

df = pd.read_csv('../clustering/dataset_clustered.csv')
unique_em_clusters = df['EM_Cluster'].unique()

output_lines = []
valid_clusters = []

while len(valid_clusters) < 20:
    cluster = random.choice(unique_em_clusters)

    cluster_df = df[df['EM_Cluster'] == cluster]
    doc_ids = cluster_df['Document ID'].drop_duplicates().tolist()

    while len(doc_ids) < 8 or len(doc_ids) > 15:
        cluster = random.choice(unique_em_clusters)
        cluster_df = df[df['EM_Cluster'] == cluster]
        doc_ids = cluster_df['Document ID'].drop_duplicates().tolist()

    valid_clusters.append((cluster, doc_ids))

for cluster, doc_ids in valid_clusters:
    output_lines.append(f"Cluster {cluster}: {','.join(map(str, doc_ids))}")

for line in output_lines:
    print(line)

output_file_path = 'random_clusters.txt'
with open(output_file_path, 'w') as f:
    f.write("\n".join(output_lines))
