import os

from datasets import load_dataset

columns_to_keep = ["almanach_camembert-base"]

annotations = load_dataset(
    "./datastore/QFrBLiMP", data_files=["annotations.tsv"], sep="\t"
)["train"]

dir_path = os.path.join("predictions", "fr")
all_predictions_file = os.listdir(dir_path)

llm_annotations = load_dataset(dir_path, data_files=all_predictions_file, sep="\t")[
    "train"
]

for column in columns_to_keep:
    annotations = annotations.add_column(column, llm_annotations[column])
