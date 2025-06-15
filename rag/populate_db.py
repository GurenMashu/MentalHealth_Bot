# rag/populate_chroma.py
import chromadb
import pandas as pd
from data.save_load_data import load_data
import logging
from typing import Optional

CHROMA_PATH = "./chroma_db"

def populate_chromadb(csv_path: str, collection_name: str = "empathetic_data", batch_size: int = 1000):
    """
    Populate ChromaDB with empathetic conversation data.
    
    Args:
        csv_path: Path to the CSV file containing the data
        collection_name: Name of the ChromaDB collection
        batch_size: Number of documents to add in each batch
    """
    try:
        df = load_data(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        
        required_columns = ['contexts', 'responses', 'emotions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df = df.dropna(subset=required_columns)
        print(f"After removing rows with missing data: {len(df)} rows")
        
        df['combined'] = df.apply(
            lambda row: f"Context: {row['contexts']} | Response: {row['responses']} | Emotion: {row['emotions']}", 
            axis=1
        )
        
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Delete existing collection if it exists (optional - remove if you want to append)
        # try:
        #    client.delete_collection(name=collection_name)
        #    print(f"Deleted existing collection: {collection_name}")
        # except ValueError:
        #    pass  # Collection doesn't exist
        
        collection = client.get_or_create_collection(name=collection_name)
        
        # Add documents in batches for better performance
        total_rows = len(df)
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Prepare batch data
            ids = [str(i) for i in range(start_idx, end_idx)]
            documents = batch_df['combined'].tolist()
            metadatas = [
                {
                    "context": row['contexts'],
                    "response": row['responses'], 
                    "emotion": row['emotions'],
                    "original_index": idx
                }
                for idx, row in batch_df.iterrows()
            ]
            
            # Add batch to collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"Added batch {start_idx//batch_size + 1}/{(total_rows-1)//batch_size + 1} "
                  f"({end_idx}/{total_rows} documents)")
        
        print(f"ChromaDB population complete. Added {total_rows} documents to collection '{collection_name}'")
        
        # Verify the collection
        collection_count = collection.count()
        print(f"Collection now contains {collection_count} documents")
        #return collection
        
    except Exception as e:
        print(f"Error populating ChromaDB: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    csv_path = "path/to/your/data.csv"  # Update this path
    populate_chromadb(csv_path)