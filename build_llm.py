
########

generated_queries = pd.read_csv("more_queries.csv", header=0, index_col=False)

query = generated_queries['question'][i]
query_list.append(query)


context = "\nExtracted documents:\n"
context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(oracle)])


genai.configure(api_key=API_KEY)

rag_response_list = []
prompt_list = []

for i in range(len(query_list)):
  
  prompt = "Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide answer based on all the source document when relevant.If the answer cannot be deduced from the context, do not give an answer." + "here is the context {} and".format(context_list[i]) + "here is the question: {}".format(query_list[i])
  prompt_list.append(prompt)
  model = genai.GenerativeModel("gemini-1.5-flash")
  rag_response = model.generate_content(prompt)
  rag_response_list.append(rag_response.text)

data = {
    'query': query_list,
    'context': context_list,
     'summary': rag_response_list
}

df = pd.DataFrame(data)

df.to_csv("rag_qa_summary_100_20.csv", index=False)
