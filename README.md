# stanford_cs229

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

