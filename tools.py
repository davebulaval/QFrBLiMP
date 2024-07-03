import json
import os
from functools import partial
from typing import List, Union

import torch
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values

from evaluation_tools import evaluation_llm, evaluation
from factory import model_tokenizer_factory

BASELINES = ["Aléatoire"]


def filename_to_model_name(filename):
    model_name = None
    if "xlm-roberta-base" in filename:
        model_name = "RoBERTa-base"
    elif "xlm-roberta-large" in filename:
        model_name = "RoBERTa-large"
    elif "bert-base" in filename:
        model_name = "BERT"
    elif "Llama" in filename:
        model_name = "Llama"
    elif "camembert-base" in filename:
        model_name = "CamemBERT-base"
    elif "camembert-large" in filename:
        model_name = "CamemBERT"
    return model_name


secrets = dotenv_values(".env")

huggingface_token = secrets["huggingface_token"]

device = torch.device("cuda")


def evaluation_loop(
    model_names: List,
    dataset: Union[Dataset, DatasetDict],
    output_file_name_format: str,
    compute_subcat: bool = False,
    seed: int = 42,
    class_to_predict: int = 0,
):
    for model_name in model_names:
        model, tokenizer = model_tokenizer_factory(
            model_name=model_name,
            device=device,
            token=huggingface_token,
            seed=seed,
            class_to_predict=class_to_predict,
        )

        if model_name != "Aléatoire":
            evaluation_fn = partial(
                evaluation_llm, tokenizer=tokenizer, model=model, device=device
            )
        else:
            evaluation_fn = partial(evaluation, model=model)

        process_dataset = dataset.map(evaluation_fn)

        minimal_pair_comparison = process_dataset["train"]["minimal_pair_comparison"]
        accuracy = round(
            sum(minimal_pair_comparison) / len(minimal_pair_comparison) * 100, 2
        )

        model_results = {"accuracy": accuracy}

        os.makedirs("results", exist_ok=True)
        model_name = model_name.replace("/", "_")
        with open(
            os.path.join("results", output_file_name_format.format(model_name)),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(model_results, f, ensure_ascii=False)

        if compute_subcat:
            accuracies = (
                process_dataset["train"]
                .to_pandas()
                .groupby("subcat")["minimal_pair_comparison"]
                .mean()
                * 100
            )

            model_results_per_subcat = {
                "accuracies": {key: value for key, value in accuracies.items()}
            }

            output_file_name_format = output_file_name_format.replace(
                "_results", "_results_per_subcat"
            )
            with open(
                os.path.join("results", output_file_name_format.format(model_name)),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(model_results_per_subcat, f, ensure_ascii=False)
