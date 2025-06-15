# rag/retriever.py
import chromadb
from typing import List
CHROMA_PATH = "./chroma_db" 

def retrieve_context(query: str, top_k: int = 3) -> List[str]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("empathetic_data")
    results = collection.query(query_texts=[query], n_results=top_k)
    return [doc for doc in results['documents'][0]]
