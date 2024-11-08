from tqdm import tqdm  # Import the standard tqdm for terminal use
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the dataset from Hugging Face
import datasets
ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Set display options for pandas
pd.set_option("display.max_colwidth", None)

# Initialize a LangchainDocument from the dataset
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]

# Define a text splitter
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
    chunk_size=1000,  # Maximum characters in a chunk
    chunk_overlap=100,  # Characters overlap between chunks
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

# Split the documents
docs_processed = []
seq = []
doc_list = []
i = 0 
for doc in RAW_KNOWLEDGE_BASE:
    split_docs = text_splitter.split_documents([doc])
    docs_processed += split_docs
    seq.append(np.arange(len(split_docs)))
    doc_list.append(i * np.ones(len(split_docs)))
    i += 1

# Flatten sequences
def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]

index = np.arange(len(docs_processed))
seq_flatten = np.array(flatten_comprehension(seq))
doc_list_flatten = np.array(flatten_comprehension(doc_list))
chunk_texts = [doc.page_content for doc in docs_processed]

# Initialize SentenceTransformer to generate embeddings
embedding_model = SentenceTransformer("thenlper/gte-small")

# Generate embeddings for each chunk
embeddings = [embedding_model.encode(doc.page_content) for doc in tqdm(docs_processed)]

# Create the dataframe with required fields, including chunk_text
data = {
    'chunk_id': index,
    'chunk_seq_id': seq_flatten,
    'doc_id': doc_list_flatten,
    'chunk_text': chunk_texts,
    'chunk_embedding_vec': embeddings
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Optional: Display the DataFrame
print(df.head())

# Save to CSV or other file if needed
df.to_csv("chunk_metadata_embeddings.csv", index=False)
