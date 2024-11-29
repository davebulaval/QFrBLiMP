# We use load_dataset from HG datasets library since it handle well JSONL
import os

from datasets import load_dataset

dir_path = os.path.join("../datastore", "QFrBLiMP")
annotated_data_dir = os.path.join(dir_path, "annotated")

files = os.listdir(annotated_data_dir)

annotated_data = load_dataset(
    annotated_data_dir,
    data_files=files,
)

# We clean that the second sentence is in a list format
annotated_data = annotated_data["train"].map(
    lambda x: {"sentence_2": x["sentence_2"][0]}
)

# We fix the error that the grammatical and ungrammatical column are mixed up
annotated_data = annotated_data.rename_column("grammatical", "temp")
annotated_data = annotated_data.rename_column("ungrammatical", "grammatical")
annotated_data = annotated_data.rename_column("temp", "ungrammatical")

annotated_data.to_json(
    os.path.join(dir_path, "merge_annotated", "fr_blimp_annotated.jsonl"),
    force_ascii=False,
)
