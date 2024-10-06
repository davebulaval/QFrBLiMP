# We use load_dataset from HG datasets library since it handle well JSONL
import os

from datasets import load_dataset

dir_path = os.path.join("datastore", "FrBLiMP", "annotated")

files = os.listdir(dir_path)

annotated_data = load_dataset(
    dir_path,
    data_files=files,
)

annotated_data["train"].to_json(
    os.path.join(dir_path, "merge_fr_blimp_annotated_50.jsonl"),
    force_ascii=False,
)
