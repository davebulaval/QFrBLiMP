import os.path

from datasets import load_dataset

data = load_dataset(
    "json",
    data_dir=os.path.join("datastore", "QFrBLiMP", "release"),
    data_files=["qfrblimp.jsonl"],
)

print("a")

data["train"].to_csv("men_colise.csv", index=False)

data_split = load_dataset(
    "json",
    data_dir=os.path.join("datastore", "QFrBLiMP", "release"),
    data_files={
        "train": "qfrblimp_train.jsonl",
        "dev": "qfrblimp_dev.jsonl",
        "test": "qfrblimp_test.jsonl",
    },
)
