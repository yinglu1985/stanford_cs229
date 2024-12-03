from tqdm.auto import tqdm
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import math

# Load the dataset
ds = load_dataset("m-ric/huggingface_doc", split="train")

# Convert the dataset into LangChain Documents
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]

# Extract document content and metadata
doc_contents = [doc.page_content for doc in RAW_KNOWLEDGE_BASE]
doc_sources = [doc.metadata["source"] for doc in RAW_KNOWLEDGE_BASE]

# Dynamically calculate the number of clusters as total documents / 10
num_documents = len(doc_contents)
num_clusters = num_documents // 10 # Ensure at least 1 cluster
print(f"Number of clusters: {num_clusters}")

# Convert the document contents to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X = vectorizer.fit_transform(doc_contents)

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
doc_clusters = kmeans.fit_predict(X)

# Create a DataFrame to store the clustering results
clustered_docs_df = pd.DataFrame({
    "Source": doc_sources,
    "Content": doc_contents,
    "Cluster": doc_clusters
})

# Save the clustering results to a CSV file
output_file = "dynamic_document_clusters.csv"
clustered_docs_df.to_csv(output_file, index=False)

print(f"K-means clustering complete. Results saved to '{output_file}'")
