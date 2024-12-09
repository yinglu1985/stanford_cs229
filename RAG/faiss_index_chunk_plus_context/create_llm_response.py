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
# import bitsandbytes
from llamaapi import LlamaAPI


def search_and_answer(query, embedding_model, vector_store_chunk, vector_store_chunk_plus_context, reranker, llama, k):
    num_retrieval = 100
    query_embedding = embedding_model.embed_documents([query])
    _, indices_chunk = vector_store_chunk.index.search(np.array(query_embedding, dtype=np.float32), k=num_retrieval)
    _, indices_chunk_plus_context = vector_store_chunk_plus_context.index.search(
        np.array(query_embedding, dtype=np.float32), k=num_retrieval)

    indices = list(set(indices_chunk[0]) | set(indices_chunk_plus_context[0]))
    ids = []
    doc_context = []
    for j, i in enumerate(indices):
        if i == -1:
            continue
        docstore_id = vector_store_chunk.index_to_docstore_id[i]
        doc = vector_store_chunk.docstore.search(docstore_id).page_content
        ids.append((i, docstore_id))
        doc_context.append(doc)

    relevant_docs = reranker.rerank(query, doc_context, k)
    final_context = "\n".join(doc['content'] for doc in relevant_docs)

    prompt = (
        "Using the information contained in the context, give a comprehensive answer to the question. "
        "Respond only to the question asked, response should be concise and relevant to the question. "
        "Provide answer based on all the source documents when relevant. If the answer cannot be deduced "
        "from the context, do not give an answer.\n"
        f"Here is the context: {final_context}\nHere is the question: {query}"
    )

    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = llama.run(api_request_json)
    rag_response = response.json()['choices'][0]['message']['content']

    return ids, prompt, rag_response


def main():
    # Configure API and model
    genai.configure(api_key=os.environ['API_KEY'])
    reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    question_df = pd.read_csv("../../question_generation/questions_and_answers.csv")
    # print(question_df['Question'])

    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    llama = LlamaAPI(os.environ['API_KEY'])

    # Load FAISS vector stores
    KNOWLEDGE_VECTOR_DATABASE_chunk = FAISS.load_local(
        "faiss_index_chunk", embedding_model, allow_dangerous_deserialization=True
    )
    KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context = FAISS.load_local(
        "faiss_index_chunk_plus_context", embedding_model, allow_dangerous_deserialization=True
    )

    results = []
    # Example queries
    queries = question_df['Question']
    for query in queries:
        print(query)
        ids, prompt, rag_response = search_and_answer(
            query, embedding_model, KNOWLEDGE_VECTOR_DATABASE_chunk, KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context,
            reranker, llama, k=20
        )
        print(f"Query: {query}")
        print(f"Response: {rag_response}\n")

        results.append({
            "query": query,
            "ids": ids,
            "prompt": prompt,
            "rag_response": rag_response
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("llm_rag_responses.csv", index=False)

    print("Results saved to 'results.csv'")


if __name__ == "__main__":
    main()
