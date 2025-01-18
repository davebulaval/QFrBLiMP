from datasets import load_dataset

columns_to_keep = ["almanach_camembert-base"]

annotations = load_dataset(
    "./datastore/QFrBLiMP", data_files=["annotations.tsv"], sep="\t"
)["train"]

llm_annotations = load_dataset(
    "./predictions/fr", data_files=["fr_all_predictions.tsv"], sep="\t"
)["train"]

for column in columns_to_keep:
    annotations = annotations.add_column(column, llm_annotations[column])
