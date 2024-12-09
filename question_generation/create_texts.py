import pandas as pd
import os

docs_df = pd.read_csv('../data_cleaning/documents.csv')

output_dir = 'output_texts'
file_counter = 1

with open('random_clusters.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    if line.strip():
        cluster_info = line.strip().split(':')
        cluster_id = cluster_info[0].replace('Cluster', '').strip()
        doc_ids = map(float, cluster_info[1].strip().split(','))
        doc_ids = list(doc_ids)

        doc_ids_line = f"Cluster {cluster_id}: Documents {','.join(map(str, doc_ids))}\n"

        doc_contents = ""

        for doc_id in doc_ids:
            doc_row = docs_df[docs_df['Document ID'] == doc_id]
            doc_contents += f"Document {int(doc_id)}: {doc_row['Content'].values[0]}\n"

        cluster_content = doc_ids_line + doc_contents

        outfile = f'{output_dir}/texts_{file_counter}.txt'
        with open(outfile, 'w', encoding='utf-8') as file:
            file.write(cluster_content)

        file_counter += 1
