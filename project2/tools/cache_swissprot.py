"""Build the plain-text SwissProt cache from the HF parquet (run once)."""
import os
import pandas as pd

BASE = ("/home/pany3/.cache/huggingface/hub/"
        "datasets--DanielHesslow--SwissProt-Pfam/snapshots/"
        "0db4e0ea3153c4caa5c49e7387f65b78a8996148/data/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split, f in [("train", "train-00000-of-00001.parquet"),
                     ("dev", "dev-00000-of-00001.parquet")]:
        seqs = pd.read_parquet(BASE + f, columns=["seq"])["seq"].tolist()
        with open(os.path.join(OUT, f"swissprot_{split}.txt"), "w") as fh:
            fh.write("\n".join(seqs))
        print(split, len(seqs))
