from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np

import datasets
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import os

from transformers import AutoModelForTokenClassification
from transformers import pipeline


#pd.set_option("display.max_colwidth", None) 


ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
#Copied



RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]


# Load pre-trained model and tokenizer
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"  # A BERT model fine-tuned for NER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Create a pipeline for NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")



# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
    chunk_overlap=100,  # The number of characters to overlap between chunks
    add_start_index=True,  # If `True`, includes chunk's start index in metadata
    strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    separators=MARKDOWN_SEPARATORS,
)

doc_list=[]
seq=[]
docs_processed = []
i = 0 
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])
    seq.append(np.arange(len(text_splitter.split_documents([doc]))))
    doc_list.append(i * np.ones(len(text_splitter.split_documents([doc]))))
    i = i + 1




# To get the value of the max sequence_length, we will query the underlying `SentenceTransformer` object used in the RecursiveCharacterTextSplitter
print(f"Model's maximum sequence length: {SentenceTransformer('thenlper/gte-small').max_seq_length}")



tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]

# Plot the distribution of document lengths, counted as the number of tokens
# fig = pd.Series(lengths).hist()
# plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
# plt.show()



EMBEDDING_MODEL_NAME = "thenlper/gte-small"


# def split_documents(
#     chunk_size: int,
#     knowledge_base: List[LangchainDocument],
#     tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
# ) -> List[LangchainDocument]:
#     """
#     Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
#     """
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         AutoTokenizer.from_pretrained(tokenizer_name),
#         chunk_size=chunk_size,
#         chunk_overlap=int(chunk_size / 10),
#         add_start_index=True,
#         strip_whitespace=True,
#         separators=MARKDOWN_SEPARATORS,
#     )

#     docs_processed = []
#     for doc in knowledge_base:
#         docs_processed += text_splitter.split_documents([doc])

#     # Remove duplicates
#     unique_texts = {}
#     docs_processed_unique = []
#     for doc in docs_processed:
#         if doc.page_content not in unique_texts:
#             unique_texts[doc.page_content] = True
#             docs_processed_unique.append(doc)

#     return docs_processed_unique


# docs_processed = split_documents(
#     512,  # We choose a chunk size adapted to our model
#     RAW_KNOWLEDGE_BASE,
#     tokenizer_name=EMBEDDING_MODEL_NAME,
# )

# Let's visualize the chunk sizes we would have in tokens from a common model

# tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
# lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
# fig = pd.Series(lengths).hist()
# plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
# plt.show()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# embedding_model = HuggingFaceEmbeddings(
#     model_name=EMBEDDING_MODEL_NAME,
#     multi_process=True,
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
# )

# KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#     docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
# )

# KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n().shape

# np.savetxt("knowledge_vector_db.txt", KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n())

# loaded_array = np.loadtxt("knowledge_vector_db.txt")

def flatten_comprehension(matrix):
  return [item for row in matrix for item in row]

index = np.arange(len(docs_processed))
seq_flatten = np.array(flatten_comprehension(seq))
doc_list_flatten = np.array(flatten_comprehension(doc_list))
doc_processed_page_contents = np.array([docs_processed[i].page_content for i in range(len(docs_processed))])
#vector = KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n()

dataset = np.stack([index, doc_list_flatten, seq_flatten, doc_processed_page_contents], axis=1)


#dataset_org = dataset
#dataset = dataset[:100, :]
#print(dataset.shape[0])

# for row in range(dataset.shape[0]):
#     entity_list = ner_pipeline(dataset[row][3])
#     entity_column.append(entity_list)
    
#     high_score_entities = [item for item in entity_list if item['score'] > 0.8]
#     high_score_entity_column.append(high_score_entities)

# entity_column = np.array(entity_column, dtype=object).reshape(-1, 1)
# dataset_with_entities = np.hstack((dataset, entity_column))

# high_score_entity_column = np.array(high_score_entity_column, dtype=object).reshape(-1, 1)
# dataset_with_entities = np.hstack((dataset_with_entities, high_score_entity_column))


text_column = dataset[:, 3]
text_column = [text for text in text_column if text] 
#print(text_column)


# Process all rows in a single batch, if the `ner_pipeline` supports batch processing
# (This assumes `ner_pipeline` can handle multiple inputs at once)
entity_list_batch = ner_pipeline(text_column)

print("-------")

# Initialize the lists for entity columns
entity_column = []
high_score_entity_column = []

# Iterate through the results in a single pass
for entity_list in entity_list_batch:
    entity_column.append(entity_list)
    # Extract entities with a score > 0.8 in one step
    high_score_entity_column.append([item for item in entity_list if item['score'] > 0.8])

# Convert to numpy arrays and reshape them
entity_column = np.array(entity_column, dtype=object).reshape(-1, 1)
high_score_entity_column = np.array(high_score_entity_column, dtype=object).reshape(-1, 1)

# Stack all columns together at once
dataset_with_entities = np.hstack((dataset, entity_column, high_score_entity_column))
    

with open('dataset_with_entities.txt', 'w') as f:
    for line in dataset_with_entities:
        f.write(f"{line}\n")

    

# content_all = []
# with open("data.txt", "r") as f:
#   for line in f:
#     content = f.readline()
#     content_all.append(content)
#     print(content)

# file.close()
