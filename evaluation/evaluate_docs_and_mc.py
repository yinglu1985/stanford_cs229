import pandas as pd
import ast
import re


multiple_answer_response = pd.read_csv('../RAG/multiple_answer_response_200_300.csv').drop_duplicates(subset=['query'])
llm_rag_responses = pd.read_csv('../RAG/llm_rag_responses.csv').drop_duplicates(subset=['query'])
questions_and_answers = pd.read_csv('../question_generation/questions_and_answers.csv').drop_duplicates(subset=['Question'])
dataset_clustered = pd.read_csv('../clustering/dataset_clustered.csv')

llm_rag_responses = llm_rag_responses.drop(columns=['rag_response'])
chunk_to_doc_map = dict(zip(dataset_clustered['Chunk ID'], dataset_clustered['Document ID']))

def map_chunks_to_documents(ids):
    if not isinstance(ids, str):
        return []
    chunk_ids = [item[0] for item in ast.literal_eval(ids)]
    return list(set(chunk_to_doc_map.get(chunk_id) for chunk_id in chunk_ids if chunk_id in chunk_to_doc_map))

llm_rag_responses['retrieved_document_ids'] = llm_rag_responses['ids'].apply(map_chunks_to_documents)

def extract_ground_truth_ids(documents):
    if isinstance(documents, str):
        return [doc_id.strip() for doc_id in documents.split(",")]
    return []

questions_and_answers['ground_truth_document_ids'] = questions_and_answers['Documents'].apply(extract_ground_truth_ids)

llm_rag_responses['retrieved_document_ids'] = llm_rag_responses['retrieved_document_ids'].apply(
    lambda x: [str(doc_id) for doc_id in x]
)

llm_rag_responses = llm_rag_responses.merge(
    questions_and_answers[['Question', 'ground_truth_document_ids', 'Correct Answer', 'A', 'B', 'C', 'D']],
    left_on='query',
    right_on='Question',
    how='left'
)

llm_rag_responses['ground_truth_document_ids'] = llm_rag_responses['ground_truth_document_ids'].apply(
    lambda x: [] if isinstance(x, float) and pd.isna(x) else x
)

def calculate_doc_accuracy(row):
    retrieved = set(row['retrieved_document_ids'])
    ground_truth = set(row['ground_truth_document_ids'])
    if len(ground_truth) == 0:
        return 0
    return len(retrieved.intersection(ground_truth)) / len(ground_truth)

llm_rag_responses['document_accuracy_rate'] = llm_rag_responses.apply(calculate_doc_accuracy, axis=1)

llm_rag_responses = llm_rag_responses.merge(
    multiple_answer_response[['query', 'rag_response']],
    left_on='query',
    right_on='query',
    how='left'
)

def check_multiple_choice_accuracy(row):
    correct_answer = row['Correct Answer'].strip().upper()
    llm_response = row.get('rag_response', None)

    if not isinstance(llm_response, str):
        invalid_rows.append(row.name)
        return "False"

    alpha_to_numeric = {"A": "1", "B": "2", "C": "3", "D": "4"}
    correct_numeric = alpha_to_numeric.get(correct_answer)
    if not correct_numeric:
        invalid_rows.append(row.name)
        return "False"

    answer_content = row[correct_answer]
    if isinstance(answer_content, str):
        answer_content_snippet = re.escape(answer_content[:20])
    else:
        answer_content_snippet = None

    patterns = [
        fr"Answer {correct_numeric}",
        fr"answer {correct_numeric}",
        fr"Answer{correct_numeric}",
        fr"answer{correct_numeric}",
        fr"{correct_numeric}\.",
        fr"\b{correct_numeric}\b",
    ]

    if answer_content_snippet:
        patterns.append(answer_content_snippet)

    for pattern in patterns:
        match = re.search(pattern, llm_response, re.IGNORECASE)
        if match:
            return "True"

    invalid_rows.append(row.name)
    return "False"

invalid_rows = []
llm_rag_responses['multiple_choice_accuracy'] = llm_rag_responses.apply(check_multiple_choice_accuracy, axis=1)

if invalid_rows:
    print(f"No valid answers for rows: {', '.join(map(str, invalid_rows))}")
else:
    print("All rows have valid answers.")

evaluation_metrics = llm_rag_responses[['query', 'document_accuracy_rate', 'multiple_choice_accuracy']].copy()
evaluation_metrics.rename(columns={'query': 'Question'}, inplace=True)

avg_doc_accuracy_rate = llm_rag_responses['document_accuracy_rate'].mean()
multiple_choice_accuracy_percentage = (
    llm_rag_responses['multiple_choice_accuracy'].value_counts(normalize=True).get("True", 0) * 100
)

evaluation_metrics_path = 'evaluation_metrics.csv'
evaluation_metrics.to_csv(evaluation_metrics_path, index=False)

print(f"Average Document Accuracy Rate: {avg_doc_accuracy_rate:.2f}")
print(f"Multiple Choice Accuracy Percentage: {multiple_choice_accuracy_percentage:.2f}%")
print(f"Evaluation metrics saved to: {evaluation_metrics_path}")
