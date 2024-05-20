import json
from functools import partial

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from evaluation_tools import evaluation

model_names = [
    "google-bert/bert-base-chinese",
    "hfl/chinese-macbert-base",
    "zhichen/Llama3-Chinese",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
]
device = torch.device("cuda")

all_results = {}
for model_name in model_names:
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluation_fn = partial(evaluation, tokenizer=tokenizer, model=model, device=device)

    dataset = load_dataset("suchirsalhan/CLiMP")

    process_dataset = dataset.map(evaluation_fn)

    accuracy = round(sum(process_dataset) / len(process_dataset) * 100, 2)

    model_results = {"accuracy": accuracy}

    model_name = model_name.replace("/", "_")
    with open(f"climp_results{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False)
