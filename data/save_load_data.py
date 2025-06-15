import pandas as pd
import os
output_dir = os.path.join("data")
os.makedirs(output_dir, exist_ok=True)
path = os.path.join(output_dir, "dataset.csv")  #path="data\dataset.csv"

def save_data() -> str:
    splits = {'train': 'data/train-00000-of-00001-166ca6808f80f10a.parquet', 'test': 'data/test-00000-of-00001-f955869ac3dc9e18.parquet', 'val': 'data/val-00000-of-00001-0fc035ad83961da7.parquet'}
    df = pd.read_parquet("hf://datasets/ashiyakatuka11/empathetic_dialogues_context/" + splits["train"])
    df.to_csv(path, index=False)
    return path

def load_data(csv_path):
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    save_data()