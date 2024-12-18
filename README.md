# stanford_cs229

Ying: Contribute the overall idea and architecture design for the project. setting up and writing code for RAG pipeline system end-to-end including data cleaning, document chunking, document embedding with FAISS, contextual retrieval implementation with batch prompts, reranking model with thenlper/gte-small, and prompt engineering, LLM answer generation(genai-1.5-pro and Llama-70b) and evaluation.
Hyperparameter tuning for chunk size, retrieval size, and reranking size and LLM models. In addition, creating the search endpoint on local machine for visualizing the answers. 

Binbin: Implement code to configure NER pipeline and enrich the embedding process.\ Implement code to process batch results to get LLM's multiple answers to queries, prompt tuning, and generate batch results for LLM summarized response and multiple choice answers to queries. Set up GCP instance and environment, deploy required codebase and execute workflow from end to end.

Our code should be read in the following order: 

data_cleaning: We cleaned our data here.

clustering: We clustered the data here.

question_generation: Our questions were generated through lots of prompting on ChatGPT manually (took 100+ hours to get this right); we mostly create a csv of the questions here.

RAG: We extract the document IDs and the RAG answer in this step.

Evaluation: We evaluate the results of the RAG answers in this step.

To access our web application, follow the steps:
Part 1: Query generation: 

We would like to generate a set of complex queries to test the limitation of RAG. To do that, we would like to cluster chunks with relevance, and generate queries based on those chunks. 


Generate data entities used for query generation. This step generates the entities for each chunk, that will be later used to cluster chunks, and query generation based on those chunks. 
```
python3 create_dataset_with_entity_v2_batch.py
```


Part 2: RAG data generation:
```
python3 create_chunk_and_contextual_text.py
```

which creates a csv file: create_chunk_plus_context_embedding. 


Perform similarity search, retrieval and reranking to get RAG answer: 
```
python3 generate_batch_response_main.py
```
Perform similarity search, retrieval, and reranking to get multiple choice answer. 

```
python3 create_batch_multiple_answer_response_.py
```

Which creates four datasets: llm_rag_responses_chunk_only_300.csv, llm_rag_responses_with_context_300.csv, llm_rag_responses_multipe_choices.csv and llm_rag_responses_multipe_choices_chunk_only.csv

Part 3: Evaluation: 




Part 4: 
Create search endpoint
```
python3 create_search_endpoint.py
```

We provide an endpoint and a web application for this RAG project.  There are 3647 documents based on which that we can ask questions.

First run
```
python3 create_chunk_and_contextual_text.py
```
We get the chunk embedding along with document index. 

Next, given a query, run the following to do similarity search and perform retrieval and reranking step. Finally, leverage gemini-1.5 to generate answer to the query. 

```
uvicorn create_search_endpoint:app --reload
```
after seeing application startup complete, run the following command: 
```
curl http://127.0.0.1:8000/query/how%20to%20set%20up%20AWS/3/0 -H "Accept: application/json"
```
We get 
```
#{"response":"Initially, AWS is the only available cloud provider, and you can select either `us-east-1` or `eu-west-1` as your region during endpoint creation.\n"}%
```                    

