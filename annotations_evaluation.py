import os

import pandas as pd
from datasets import load_dataset

better_than_human = pd.read_csv(
    os.path.join("results", f"better_than_human_fr.tsv"), sep="\t"
)

columns_to_keep = list(
    set(
        [
            "almanach/camembert-base",
            "almanach/camembert-large",
            "dbmdz/bert-base-french-europeana-cased",
            # "FacebookAI/xlm-roberta-base",
            # "FacebookAI/xlm-roberta-large",
            "jpacifico/Chocolatine-14B-Instruct-DPO-v1.2",
            "jpacifico/Chocolatine-14B-Instruct-DPO-v1.2",
            "jpacifico/French-Alpaca-Llama3-8B-Instruct-v1.0",
            "OpenLLM-France/Lucie-7B",
            "OpenLLM-France/Lucie-7B-Instruct",
            "OpenLLM-France/Lucie-7B-Instruct-human-data",
            "OpenLLM-France/Claire-7B-FR-Instruct-0.1",
        ]
        + better_than_human["model_name"].to_list()
    )
)

annotations = load_dataset(
    "./datastore/QFrBLiMP", data_files=["annotations.tsv"], sep="\t"
)["train"]

dir_path = os.path.join("predictions", "fr")
all_predictions_file = os.listdir(dir_path)


for column in columns_to_keep:
    llm_annotations = load_dataset(
        dir_path, data_files=[f"{column.replace('/', '_')}_predictions.tsv"], sep="\t"
    )["train"]
    annotations = annotations.add_column(
        column.split("/")[-1], llm_annotations["minimal_pair_comparison"]
    )

annotations.to_csv(
    os.path.join("datastore", "QFrBLiMP", "human_llm_annotations.tsv"),
    sep="\t",
    index=False,
)
