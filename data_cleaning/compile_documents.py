from tqdm.auto import tqdm
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
import pandas as pd


ds = load_dataset("m-ric/huggingface_doc", split="train")


RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]

doc_contents = [doc.page_content for doc in RAW_KNOWLEDGE_BASE]
doc_sources = [doc.metadata["source"] for doc in RAW_KNOWLEDGE_BASE]

clustered_docs_df = pd.DataFrame({
    "Document ID": range(len(doc_contents)),
    "Source": doc_sources,
    "Content": doc_contents
})


output_file = "documents.csv"
clustered_docs_df.to_csv(output_file, index=False)

print(f"Document data saved to '{output_file}'")
