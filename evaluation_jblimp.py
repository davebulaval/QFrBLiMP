import json
import os
from functools import partial

import torch
from datasets import load_dataset
from dotenv import dotenv_values
from transformers import logging

from evaluation_tools import evaluation
from factory import model_tokenizer_factory

logging.set_verbosity_warning()
secrets = dotenv_values(".env")

token = secrets["huggingface_token"]

model_names = [
    "tohoku-nlp/bert-base-japanese-v3",
    "meta-llama/Llama-2-7b-hf",
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

    dataset = load_dataset("polm-stability/jblimp")
    dataset = dataset.rename_column("good_sentence", "sentence_good")
    dataset = dataset.rename_column("bad_sentence", "sentence_bad")

    process_dataset = dataset.map(evaluation_fn)

    minimal_pair_comparison = process_dataset["train"]["minimal_pair_comparison"]
    accuracy = round(
        sum(minimal_pair_comparison) / len(minimal_pair_comparison) * 100, 2
    )

    model_results = {"accuracy": accuracy}

    os.makedirs("results", exist_ok=True)
    model_name = model_name.replace("/", "_")
    with open(
        os.path.join("results", f"jblimp_results_{model_name}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(model_results, f, ensure_ascii=False)
