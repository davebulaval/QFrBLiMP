import json
import os.path
from functools import partial

import torch
from datasets import load_dataset
from dotenv import dotenv_values
from transformers import logging

from evaluation_tools import evaluation
from factory import model_tokenizer_factory

logging.set_verbosity_warning()
device = torch.device("cuda")

secrets = dotenv_values(".env")

token = secrets["huggingface_token"]

model_names = [
    "almanach/camembert-base",
    "almanach/camembert-large",
    "dbmdz/bert-base-french-europeana-cased",
    "meta-llama/Llama-2-7b-hf",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
]

all_results = {}
for model_name in model_names:
    model, tokenizer = model_tokenizer_factory(
        model_name=model_name, device=device, token=token
    )

    evaluation_fn = partial(evaluation, tokenizer=tokenizer, model=model, device=device)

    dataset = load_dataset(
        os.path.join("datastore", "FrBLiMP"), data_files="dataset.tsv"
    )

    process_dataset = dataset.map(evaluation_fn)

    accuracy = (
        process_dataset["train"].to_pandas()["minimal_pair_comparison"].mean() * 100
    )

    accuracies = (
        process_dataset["train"]
        .to_pandas()
        .groupby("subcat")["minimal_pair_comparison"]
        .mean()
    )

    model_results = {"accuracy": accuracy}
    model_results_per_subcat = {
        "accuracies": {key: value for key, value in accuracies.items()}
    }

    os.makedirs("results", exist_ok=True)
    model_name = model_name.replace("/", "_")
    with open(
        os.path.join("results", f"frblimp_results_{model_name}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(model_results, f, ensure_ascii=False)

    with open(
        os.path.join("results", f"frblimp_results_per_subcat_{model_name}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(model_results_per_subcat, f, ensure_ascii=False)
