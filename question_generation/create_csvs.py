import pandas as pd
import os

documents_df = pd.read_csv('../data_cleaning/documents.csv')
dataset_clustered_df = pd.read_csv('../clustering/dataset_clustered.csv')


output_dir = 'output_csvs'
file_counter = 1

with open('random_clusters.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    if line.strip():
        cluster_info = line.strip().split(':')
        cluster_id = cluster_info[0].replace('Cluster', '').strip()
        doc_ids = map(float, cluster_info[1].strip().split(','))
        doc_ids = list(doc_ids)

        cluster_data = []
        for doc_id in doc_ids:
            doc_chunks = dataset_clustered_df[dataset_clustered_df['Document ID'] == doc_id]

            for i, chunk in doc_chunks.iterrows():
                cluster_data.append({
                    'Chunk ID': chunk['Chunk ID'],
                    'Document ID': chunk['Document ID'],
                    'Page Content': chunk['Page Content']
                })

        cluster_df = pd.DataFrame(cluster_data)

        out_file = f'{output_dir}/texts_{file_counter}.csv'
        cluster_df.to_csv(out_file, index=False)

        file_counter += 1

