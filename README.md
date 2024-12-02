# stanford_cs229

We provide an endpoint and a web application for this RAG project.  There are 3647 documents based on which that we can ask questions.
```
run uvicorn create_search_endpoint:app --reload
```
after seeing application startup complete, run the following command: 
```
curl http://127.0.0.1:8000/query/how%20to%20set%20up%20AWS/3/0 -H "Accept: application/json"
```
We get 
```
#{"response":"Initially, AWS is the only available cloud provider, and you can select either `us-east-1` or `eu-west-1` as your region during endpoint creation.\n"}%
```                    

