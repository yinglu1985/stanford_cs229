import pandas as pd

file_path = 'questions_and_answers.csv'
df = pd.read_csv(file_path)

cluster = 16
filtered_df = df[df['Cluster'] == cluster][['Question', 'A', 'B', 'C', 'D']]

output_path = f'questions_hard/questions_{cluster}.csv'
filtered_df.to_csv(output_path, index=False)