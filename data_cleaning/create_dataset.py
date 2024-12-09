from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch


device = torch.device("mps")
print(f"Using device: {device}")


ds = load_dataset("m-ric/huggingface_doc", split="train")  # Use only 10 samples

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]

MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0  # Assuming MPS usage
)

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
    split_docs = text_splitter.split_documents([doc])
    docs_processed += split_docs
    seq.append(np.arange(len(split_docs)))
    doc_list.append(i * np.ones(len(split_docs)))
    i += 1

index = np.arange(len(docs_processed))
doc_list_flatten = np.array([i for sublist in doc_list for i in sublist])
doc_processed_page_contents = np.array([doc.page_content for doc in docs_processed])

embedding_model = SentenceTransformer("thenlper/gte-small")
chunk_vectors = [embedding_model.encode(doc.page_content) for doc in tqdm(docs_processed)]

assert len(docs_processed) == len(chunk_vectors), "Mismatch in chunk count and vectors!"

entity_column = []
high_score_entity_column = []

print("Processing NER...")
for doc in tqdm(docs_processed, desc="NER Processing"):
    text = doc.page_content
    results = ner_pipeline(text)
    entity_column.append(results if results else [])
    high_score_entity_column.append([item for item in results if item.get('score', 0) > 0.8] if results else [])

assert len(docs_processed) == len(entity_column), "Mismatch in chunk count and entity results!"

data = {
    "Chunk ID": index,
    "Document ID": doc_list_flatten,
    "Page Content": doc_processed_page_contents,
    "Entities": entity_column,
    "High Confidence Entities": high_score_entity_column,
    "Chunk Vector": chunk_vectors
}

final_df = pd.DataFrame(data)

final_df["Entities"] = final_df["Entities"].apply(str)
final_df["High Confidence Entities"] = final_df["High Confidence Entities"].apply(str)

output_file = "dataset.csv"
final_df.to_csv(output_file, index=False)

print(f"Sample processing complete. Results saved to '{output_file}'")
