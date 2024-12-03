from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from datasets import Dataset
import datasets  # To load datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Detect if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load dataset
ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]

# Load pre-trained model and tokenizer
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Create a pipeline for NER, using MPS if available
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 if device.type == "mps" else -1
)

# Text splitting
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

docs_processed = []
doc_list = []
seq = []
i = 0
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])
    seq.append(np.arange(len(text_splitter.split_documents([doc]))))
    doc_list.append(i * np.ones(len(text_splitter.split_documents([doc]))))
    i += 1

index = np.arange(len(docs_processed))
doc_list_flatten = np.array([i for sublist in doc_list for i in sublist])
doc_processed_page_contents = np.array([docs_processed[i].page_content for i in range(len(docs_processed))])

# Build initial dataset
dataset = np.stack([index, doc_list_flatten, doc_processed_page_contents], axis=1)

# Columns to track
text_column = dataset[:, 2]
text_column = [text for text in text_column if text]
entity_column = []
high_score_entity_column = []

# Process rows with progress tracking
print("Processing NER...")
for idx, text in enumerate(tqdm(text_column, desc="Processing rows")):
    results = ner_pipeline(text)
    entity_column.append(results if results else [])
    high_score_entity_column.append([item for item in results if item.get('score', 0) > 0.8] if results else [])

# Convert to numpy arrays and reshape
entity_column = np.array(entity_column, dtype=object).reshape(-1, 1)
high_score_entity_column = np.array(high_score_entity_column, dtype=object).reshape(-1, 1)

# Stack all columns together
dataset_with_entities = np.hstack((dataset, entity_column, high_score_entity_column))

# Save the NumPy array as a CSV file
columns = ['Index', 'Document ID', 'Page Content', 'Entities', 'High Confidence Entities']
dataset_with_entities_df = pd.DataFrame(dataset_with_entities, columns=columns)

# Convert complex columns to strings for better compatibility in CSV
dataset_with_entities_df['Entities'] = dataset_with_entities_df['Entities'].apply(str)
dataset_with_entities_df['High Confidence Entities'] = dataset_with_entities_df['High Confidence Entities'].apply(str)

output_file = 'dataset_with_entities_df.csv'
dataset_with_entities_df.to_csv(output_file, index=False)

print(f"Processing complete. Results saved to '{output_file}'")
