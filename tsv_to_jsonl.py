import os

from datasets import load_dataset

dataset = load_dataset(os.path.join("datastore", "FrBLiMP"), data_files="complete.tsv")

dataset = dataset["train"].to_pandas()

dataset.columns = (
    ", ".join(dataset.columns.str.lower().to_list())
    .replace("sentence_good", "grammatical")
    .replace("sentence_bad", "ungrammatical")
    .split(", ")
)

with open("datastore/FrBLiMP/fr_blimp_sentences.jsonl", "w") as f:
    f.write(dataset.to_json(orient="records", lines=True, force_ascii=False))
