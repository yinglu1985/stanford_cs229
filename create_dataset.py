from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np


pd.set_option("display.max_colwidth", None) 

import datasets

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
Copied
from langchain.docstore.document import Document as LangchainDocument

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]

from langchain.text_splitter import RecursiveCharacterTextSplitter

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


from sentence_transformers import SentenceTransformer

# To get the value of the max sequence_length, we will query the underlying `SentenceTransformer` object used in the RecursiveCharacterTextSplitter
print(f"Model's maximum sequence length: {SentenceTransformer('thenlper/gte-small').max_seq_length}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]

# Plot the distribution of document lengths, counted as the number of tokens
# fig = pd.Series(lengths).hist()
# plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
# plt.show()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

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
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
# lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
# fig = pd.Series(lengths).hist()
# plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
# plt.show()

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import os
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

with open('data.txt', 'w') as f:
    for line in dataset:
        f.write(f"{line}\n")

# content_all = []
# with open("data.txt", "r") as f:
#   for line in f:
#     content = f.readline()
#     content_all.append(content)
#     print(content)

# file.close()
