from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#import bitsandbytes
from ragatouille import RAGPretrainedModel


# Save to CSV or other file if needed
chunk_metadata_embeddings = pd.read_csv("chunk_metadata_embeddings.csv")
generated_queries = pd.read_csv("more_queries.csv", header=0, index_col=False)

##
# prepare relevant document
num_doc_retrieved = 100
num_docs_final = 10

reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

kbsearch = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

query_list = []
relevant_docs_list=[]
context_list = []
final_prompt_list = []
for i in range(generated_queries.shape[0]):
  print(i)
  query = generated_queries['question'][i]
  query_list.append(query)
  retrieved_docs = kbsearch.similarity_search(query=query, k=num_doc_retrieved)
  retrieved_docs_page_content = [doc.page_content for doc in retrieved_docs]
  relevant_docs = reranker.rerank(query, retrieved_docs_page_content, num_docs_final)
  relevant_docs = [doc["content"] for doc in relevant_docs]
  relevant_docs_list.append(relevant_docs)
  context = "\nExtracted documents:\n"
  context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
  context_list.append(context)

data = {
    'query': query_list,
     'context': context_list
}

df = pd.DataFrame(data)

df.to_csv("rag_qa_pair.csv", index=False)