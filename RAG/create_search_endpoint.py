from flask import Flask, jsonify
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#import bitsandbytes
from ragatouille import RAGPretrainedModel
import google.generativeai as genai
import os
from typing import Union
from fastapi import FastAPI


reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE_chunk = FAISS.load_local("faiss_index_chunk", embedding_model, allow_dangerous_deserialization=True)

KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context = FAISS.load_local("faiss_index_chunk_plus_context", embedding_model, allow_dangerous_deserialization=True)

genai.configure(api_key=os.environ['API_KEY'])

vector_store_chunk = KNOWLEDGE_VECTOR_DATABASE_chunk

vector_store_chunk_plus_context = KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    return {"item_id": item_id}


@app.get("/query/{query}")
async def search_and_answer(query: str):
    num_retrieval = 100
    k = 3
    # k = 20
    query_embedding = embedding_model.embed_documents([query])
    _, indices_chunk = vector_store_chunk.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval)
    _, indices_chunk_plus_context = vector_store_chunk_plus_context.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval)
    indices = list(set(indices_chunk[0])| set(indices_chunk_plus_context[0]))
    ids = []
    id_of_retrieved_chunk = "chunks retrieved"
    doc_context = []
    for j, i in enumerate(indices):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        docstore_id = vector_store_chunk.index_to_docstore_id[i]
        doc = vector_store_chunk.docstore.search(docstore_id).page_content
        ids.append((i, docstore_id))
        id_of_retrieved_chunk +="{}th chunk, ".format(i)
        doc_context.append(doc)
    relevant_docs = reranker.rerank(query, doc_context, k)
    final_context = ""
    for i in range(len(relevant_docs)):
       final_context += relevant_docs[i]['content'] + '\n'
    prompt = "Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide answer based on all the source document when relevant.If the answer cannot be deduced from the context, do not give an answer. \n " + "Here is the context: {} \n".format(final_context) + "Here is the question: {}".format(query)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    rag_response = model.generate_content(prompt).text
    # return {"id_of_retrieved_chunk": id_of_retrieved_chunk, "prompt": prompt, "response": rag_response}
    return {"prompt": prompt, "response": rag_response}

# run uvicorn create_search_endpoint:app --reload on the command line

# http://127.0.0.1:8000/query/how to set up AWS