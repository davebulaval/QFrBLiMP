import json

import torch
from dotenv import dotenv_values
from tqdm import tqdm

from factory import model_tokenizer_factory
from tools import LLMs, BASELINES_FR

model_names = (
    [
        "google-bert/bert-base-uncased",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "GroNLP/bert-base-dutch-cased",
        "google-bert/bert-base-chinese",
        "almanach/camembert-base",
        "almanach/camembert-large",
        "dbmdz/bert-base-french-europeana-cased",
        "tohoku-nlp/bert-base-japanese-v3",
    ]
    + LLMs
    + BASELINES_FR
)

secrets = dotenv_values(".env")

huggingface_token = secrets["huggingface_token"]

device = torch.device("cuda")

models_size = {}
for model_name in tqdm(model_names):
    model, _ = model_tokenizer_factory(
        # To clean model name when we have applied a '_prompting' to it.
        model_name=(
            model_name
            if "_prompting" not in model_name
            else model_name.replace("_prompting", "")
        ),
        device=device,
        token=huggingface_token,
    )

    num_params = model.num_parameters()

    models_size[model_name] = num_params

with open("models_size.json", "w", encoding="utf-8") as file:
    json.dump(models_size, file, ensure_ascii=False)
