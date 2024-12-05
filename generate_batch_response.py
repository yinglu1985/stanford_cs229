# query = "How to set up GCP"
# ids, prompt, rag_response = search_and_answer(query, embedding_model, KNOWLEDGE_VECTOR_DATABASE_chunk, KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context, k=20)
# print(rag_response)
# # 'To set up GCP, you need to meet some requirements such as having a project on Google Cloud, enabling the billing, and installing the `gcloud` cli. Here are the steps:\n\n1. Create a project on Google Cloud: Go to the Google Cloud Console and create a new project.\n2. Enable billing: Enable billing for your project by going to the Navigation menu (three horizontal lines in the top left corner) and clicking on "Billing".\n3. Install the `gcloud`'

# query = "How to set up AWS"
# ids, prompt, rag_response = search_and_answer(query, embedding_model, KNOWLEDGE_VECTOR_DATABASE_chunk, KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context, k=20)
# print(rag_response)

from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ragatouille import RAGPretrainedModel
import google.generativeai as genai
import os
import json
#import bitsandbytes
from llamaapi import LlamaAPI


reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

##
KNOWLEDGE_VECTOR_DATABASE_chunk = FAISS.load_local("faiss_index_chunk", embedding_model, allow_dangerous_deserialization=True)

KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context = FAISS.load_local("faiss_index_chunk_plus_context", embedding_model, allow_dangerous_deserialization=True)

llama=LlamaAPI(os.environ['API_KEY'])

def search_and_answer(query, embedding_model, vector_store_chunk, vector_store_chunk_plus_context, k):
    num_retrieval = 100
    query_embedding = embedding_model.embed_documents([query])
    _, indices_chunk = vector_store_chunk.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval)
    _, indices_chunk_plus_context = vector_store_chunk_plus_context.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval)
    # print(indices_chunk[0])
    # print(indices_chunk_plus_context[0])
    indices = list(set(indices_chunk[0])| set(indices_chunk_plus_context[0]))
    ids = []
    doc_context = []
    for j, i in enumerate(indices):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        docstore_id = vector_store_chunk.index_to_docstore_id[i]
        doc = vector_store_chunk.docstore.search(docstore_id).page_content
        ids.append((i, docstore_id))
        doc_context.append(doc)
    relevant_docs = reranker.rerank(query, doc_context, k)
    final_context = ""
    for i in range(len(relevant_docs)):
       final_context += relevant_docs[i]['content'] + '\n'
    prompt = "Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide answer based on all the source document when relevant.If the answer cannot be deduced from the context, do not give an answer. \n " + "Here is the context: {} \n".format(final_context) + "Here is the question: {}".format(query)
    api_request_json = {
    "model": "llama3-70b",
    "messages": [
    {"role": "user", "content": "What is the weather like in Boston?"},
    ]}
    api_request_json['messages'][0]['content'] = prompt
    response = llama.run(api_request_json)
    rag_response = response.json()['choices'][0]['message']['content']
    # print(prompt)
    #model = genai.GenerativeModel("gemini-1.5-flash")
    #rag_response = model.generate_content(prompt).text
    # print(rag_response)
    return ids, prompt, rag_response


def search_and_answer_chunk_only(query, embedding_model, vector_store_chunk, vector_store_chunk_plus_context, k):
    num_retrieval = 100
    query_embedding = embedding_model.embed_documents([query])
    _, indices_chunk = vector_store_chunk.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval)
    _, indices_chunk_plus_context = vector_store_chunk_plus_context.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval)
    # print(indices_chunk[0])
    # print(indices_chunk_plus_context[0])
    num_retrieval_chunk_only = len(set(indices_chunk[0])| set(indices_chunk_plus_context[0]))
    _, indices_chunk = vector_store_chunk.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval_chunk_only)
    indices = list(set(indices_chunk[0]))
    ids = []
    doc_context = []
    for j, i in enumerate(indices):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        docstore_id = vector_store_chunk.index_to_docstore_id[i]
        doc = vector_store_chunk.docstore.search(docstore_id).page_content
        ids.append((i, docstore_id))
        doc_context.append(doc)
    relevant_docs = reranker.rerank(query, doc_context, k)
    final_context = ""
    for i in range(len(relevant_docs)):
       final_context += relevant_docs[i]['content'] + '\n'
    prompt = "Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide answer based on all the source document when relevant.If the answer cannot be deduced from the context, do not give an answer. \n " + "Here is the context: {} \n".format(final_context) + "Here is the question: {}".format(query)
    api_request_json = {
    "model": "llama2-7b	",
    "messages": [
    {"role": "user", "content": "What is the weather like in Boston?"},
    ]}
    api_request_json['messages'][0]['content'] = prompt
    response = llama.run(api_request_json)
    rag_response = response.json()['choices'][0]['message']['content']
    # print(prompt)
    #model = genai.GenerativeModel("gemini-1.5-flash")
    #rag_response = model.generate_content(prompt).text
    # print(rag_response)
    return ids, prompt, rag_response



results = []
question_df = pd.read_csv("questions_125.csv")
# Example queries
queries = question_df['Question']
for i in range(len(queries)):
    try: 
        query = queries[i]
        ids, prompt, rag_response = search_and_answer(query, embedding_model, KNOWLEDGE_VECTOR_DATABASE_chunk, KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context, k=5)
        print(f"Query: {query}")
        print(f"Response: {rag_response}\n")
        results.append({
            "query": query,
            "ids": ids,
            "prompt": prompt,
            "rag_response": rag_response
        })
    except Exception:
        pass

results_df = pd.DataFrame(results)
results_df.to_csv("llm_rag_responses.csv", index=False)
print("Results saved to 'results.csv'")



####

results = []
question_df = pd.read_csv("questions_125.csv")
# Example queries
queries = question_df['Question']
for i in range(len(queries)):
    try: 
        query = queries[i]
        ids, prompt, rag_response = search_and_answer_chunk_only(query, embedding_model, KNOWLEDGE_VECTOR_DATABASE_chunk, KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context, k=5)
        print(f"Query: {query}")
        print(f"Response: {rag_response}\n")
        results.append({
            "query": query,
            "ids": ids,
            "prompt": prompt,
            "rag_response": rag_response
        })
    except Exception:
        pass

results_df = pd.DataFrame(results)
results_df.to_csv("llm_rag_responses_chunk_only.csv", index=False)
print("Results saved to 'results.csv'")

