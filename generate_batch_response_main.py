import os
import pandas as pd
import numpy as np
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel
from llamaapi import LlamaAPI

# Initialize global variables
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
llama = LlamaAPI(os.environ['API_KEY'])

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Load FAISS indices
KNOWLEDGE_VECTOR_DATABASE_chunk = FAISS.load_local(
    "faiss_index_chunk", embedding_model, allow_dangerous_deserialization=True
)
KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context = FAISS.load_local(
    "faiss_index_chunk_plus_context", embedding_model, allow_dangerous_deserialization=True
)


def search_and_answer_with_context(query, embedding_model, vector_store_chunk, vector_store_chunk_plus_context, k):
    # Implementation as provided above
    pass


def search_and_answer_chunk_only(query, embedding_model, vector_store_chunk, vector_store_chunk_plus_context, k):
    # Implementation as provided above
    pass


def main():
    # Initialize results list
    results = []

    # Load questions from CSV
    question_df = pd.read_csv("questions_and_answers.csv")
    queries = question_df['Question']

    # Process each query
    for i in range(len(queries)):
        query = queries[i]
        try:
            # Search and answer using context
            ids, prompt, rag_response = search_and_answer_with_context(
                query, embedding_model, KNOWLEDGE_VECTOR_DATABASE_chunk,
                KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context, k=5
            )
            print(f"Query: {query}")
            print(f"Response: {rag_response}\n")
            results.append({"query": query, "ids": ids, "prompt": prompt, "rag_response": rag_response})
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("llm_rag_responses_with_context.csv", index=False)
    print("Results saved to 'llm_rag_responses_with_context.csv'")

    # Reset results for chunk-only processing
    results = []

    # Process each query using chunk-only method
    for i in range(len(queries)):
        query = queries[i]
        try:
            ids, prompt, rag_response = search_and_answer_chunk_only(
                query, embedding_model, KNOWLEDGE_VECTOR_DATABASE_chunk,
                KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context, k=5
            )
            print(f"Query: {query}")
            print(f"Response: {rag_response}\n")
            results.append({"query": query, "ids": ids, "prompt": prompt, "rag_response": rag_response})
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            continue

    # Save chunk-only results
    results_df = pd.DataFrame(results)
    results_df.to_csv("llm_rag_responses_chunk_only.csv", index=False)
    print("Results saved to 'llm_rag_responses_chunk_only.csv'")


if __name__ == "__main__":
    main()
