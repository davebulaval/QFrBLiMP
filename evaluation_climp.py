import json
from functools import partial

import torch
from datasets import load_dataset
from dotenv import dotenv_values

from evaluation_tools import evaluation
from factory import model_tokenizer_factory

secrets = dotenv_values(".env")

token = secrets["huggingface_token"]

model_names = [
    "google-bert/bert-base-chinese",
    "hfl/chinese-macbert-base",
    "meta-llama/Llama-2-7b-hf",
    "zhichen/Llama3-Chinese",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
]
device = torch.device("cuda")

all_results = {}
for model_name in model_names:
    model, tokenizer = model_tokenizer_factory(
        model_name=model_name, device=device, token=token
    )

    evaluation_fn = partial(evaluation, tokenizer=tokenizer, model=model, device=device)

    dataset = load_dataset("datastore/CLiMP", data_files=["climp.jsonl"])

    process_dataset = dataset.map(evaluation_fn)

    minimal_pair_comparison = process_dataset["train"]["minimal_pair_comparison"]
    accuracy = round(
        sum(minimal_pair_comparison) / len(minimal_pair_comparison) * 100, 2
    )

    model_results = {"accuracy": accuracy}

    model_name = model_name.replace("/", "_")
    with open(f"climp_results_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False)
