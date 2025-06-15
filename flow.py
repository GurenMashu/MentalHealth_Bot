from data.save_load_data import save_data
from rag.populate_db import populate_chromadb
import chromadb

def main() -> None:
    path=save_data()
    print("Data downloaded and saved")
    populate_chromadb(path)
    print("Database creation complete")

if __name__=="__main__":
    main()
