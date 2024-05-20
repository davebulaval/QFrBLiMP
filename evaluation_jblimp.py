import json
from functools import partial

import torch
from datasets import load_dataset
from dotenv import dotenv_values
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from evaluation_tools import evaluation

secrets = dotenv_values(".env")

token = secrets["huggingface_token"]

model_names = [
    "tohoku-nlp/bert-base-japanese-v3",
    "meta-llama/Llama-2-7b-hf",
    "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
]
device = torch.device("cuda")

all_results = {}
for model_name in model_names:
    if "llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, load_in_8bit=True
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluation_fn = partial(evaluation, tokenizer=tokenizer, model=model, device=device)

    dataset = load_dataset("polm-stability/jblimp")

    process_dataset = dataset.map(evaluation_fn)

    minimal_pair_comparison = process_dataset["train"]["minimal_pair_comparison"]
    accuracy = round(
        sum(minimal_pair_comparison) / len(minimal_pair_comparison) * 100, 2
    )

    model_results = {"accuracy": accuracy}

    model_name = model_name.replace("/", "_")
    with open(f"jblimp_results_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False)
