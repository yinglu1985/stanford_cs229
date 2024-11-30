from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import datasets
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import ast
import typing
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import os
import numpy as np
import pandas as pd


genai.configure(api_key=os.environ['API_KEY'])

pd.set_option("display.max_colwidth", None) 

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

def create_chunk(ds): 
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
    ]
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
        chunk_overlap=200,  # The number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )
    doc_list=[]
    seq=[]
    chunk_set = []
    original_doc = []
    i = 0
    for doc in RAW_KNOWLEDGE_BASE:
        original_doc.append([doc.page_content])
        chunk_set += [text_splitter.split_documents([doc])]
        seq.append(np.arange(len(text_splitter.split_documents([doc]))))
        doc_list.append(i)
        #doc_list.append(i * np.ones(len(text_splitter.split_documents([doc]))))
        i = i + 1
    return original_doc, chunk_set, seq, doc_list

#create contextual text

def generate_prompt(doc, chunk, seq):
    chunk_content = ""
    for i in seq:
        chunk_content += "here is the {seq}th chunk, the content is: {chunk} \n".format(seq=i, chunk = chunk[i].page_content)
    DOCUMENT_CONTEXT_PROMPT = """
    Here is the document
    <document>
    {doc_content}
    </document>
    """.format(doc_content=doc)
    CHUNK_CONTEXT_PROMPT = """
    \n
    Please give a short succinct context to situate each chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    Please return a jason format where each element is the context of the each chunk situated within the document. 
    If the context can not be generated from the chunk, set as empty string.
    Please provide the response in the form of a Python list. It should begin with “[“ and end with “]”.” 
    Please make sure each chunk will generate a context, so chunk and context is 1:1 mapping. Therefore, the length of the list of context is same as the length of the list of chunks. 
    \n
    Here are the list of chunks we want to situate within the whole document
    <chunk>
    {chunk}
    \n
    </chunk>
    {doc}
    """.format(chunk=chunk_content, doc=DOCUMENT_CONTEXT_PROMPT)
    return CHUNK_CONTEXT_PROMPT


def situate_context_gemini(original_doc, chunk_set, seq, doc_id):
    class chunk_context(typing.TypedDict):
        context: list[str]
    doc_content = original_doc[doc_id]
    chunk = chunk_set[doc_id]
    seq_number = seq[doc_id]
    prompt = generate_prompt(doc_content, chunk, seq_number)
    model = genai.GenerativeModel("gemini-1.5-pro-latest") #gemini-1.5-pro-latest, gemini-1.5-flash
    rag_response = model.generate_content(prompt, generation_config=genai.GenerationConfig(
response_mime_type="application/json", response_schema=list[chunk_context]))
    converted_context = ast.literal_eval(rag_response.text)
    final_context = [converted_context[i]['context'][0] if len(converted_context[i]['context']) > 0 else " " for i in range(len(converted_context))]
    return chunk, final_context

def p95_chunk_count_dist(chunk_set):
    len_chunk = [len(seq[doc_id]) for doc_id in doc_list]
    return np.percentile(len_chunk, q=95)

def create_chunk_and_contextual_text(ds):
    original_doc, chunk_set, seq, doc_list = create_chunk(ds)
    cutoff = p95_chunk_count_dist(chunk_set)
    num_docs = np.max(doc_list)
    chunk_plus_context_set =[]
    for doc_id in range(num_docs):
        print(doc_id)
        if len(seq[doc_id]) > cutoff: # too many chunks resulting in too large the context
            chunk_plus_context_set.append(chunk_set[doc_id])
        else:
            _, context= situate_context_gemini(original_doc, chunk_set, seq, doc_id)
            print([len(chunk_set[doc_id]), len(context)])
            if len(context) < len(chunk_set[doc_id]):  # account for LLM giving results that not exactly matches the requirement
                context += [""] * (len(chunk_set[doc_id]) - len(context))
            chunk_plus_context = []
            for i in range(len(chunk_set[doc_id])): 
                chunk_plus_context.append(LangchainDocument(page_content=chunk_set[doc_id][i].page_content + "\n the context is: \n" + context[i]))
            chunk_plus_context_set.append(chunk_plus_context)
    return original_doc, chunk_set, chunk_plus_context_set, seq, doc_list


original_doc, chunk_set, chunk_plus_context_set, seq, doc_list = create_chunk_and_contextual_text(ds)

def flatten_comprehension(matrix):
  return [item for row in matrix for item in row]

chunk_flatten = np.array(flatten_comprehension(chunk_set))
index = np.arange(len(chunk_flatten))
seq_flatten = np.array(flatten_comprehension(seq))
doc_list_flatten = np.array(flatten_comprehension([[doc_list[i]] * len(seq[i]) for i in range(len(seq))]))
chunk_plus_context_flatten = np.array(flatten_comprehension(chunk_plus_context_set))


os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE_chunk = FAISS.from_documents(
    chunk_flatten, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context = FAISS.from_documents(
    chunk_plus_context_flatten, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

KNOWLEDGE_VECTOR_DATABASE_chunk.save_local("faiss_index_chunk")
KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context.save_local("faiss_index_chunk_plus_context")

def convert_ndarray_to_list(matrix):
    list_form = []
    for i in range(matrix.shape[0]):
        list_form.append(matrix[i,])
    return list_form


data_embedding = {
    'index': index,
    'doc_list': doc_list_flatten,
    'seq': seq_flatten, 
    'chunk': chunk_flatten, 
    'chunk_plus_context': chunk_plus_context_flatten,
    'vector_store_chunk': convert_ndarray_to_list(KNOWLEDGE_VECTOR_DATABASE_chunk.index.reconstruct_n()),
    'vector_store_chunk_plus_context':  convert_ndarray_to_list(KNOWLEDGE_VECTOR_DATABASE_chunk_plus_context.index.reconstruct_n())
}

df = pd.DataFrame(data_embedding)

df.to_csv("create_chunk_plus_context_embedding.csv", index=False)

