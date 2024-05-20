import json
from functools import partial

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from evaluation_tools import evaluation

model_names = [
    "almanach/camembert-base",
    "almanach/camembert-large",
    "dbmdz/bert-base-french-europeana-cased",
    "ClassCat/gpt2-base-french",
    "LiteLLMs/French-Alpaca-Llama3-8B-Instruct-v1.0-GGUF",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
]
# device = torch.device("cuda")

all_results = {}
for model_name in model_names:
    if "gpt" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluation_fn = partial(evaluation, tokenizer=tokenizer, model=model, device=device)

    dataset = load_dataset("datastore/dataset")

    process_dataset = dataset.map(evaluation_fn)

    accuracy = round(sum(process_dataset) / len(process_dataset) * 100, 2)

    model_results = {"accuracy": accuracy}

    model_name = model_name.replace("/", "_")
    with open(f"frblimp_results{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False)
