import json
import os
from functools import partial
from typing import List, Union

import torch
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values

from evaluation_tools import (
    evaluation_llm_instruct,
    evaluation,
    evaluation_llm,
)
from factory import model_tokenizer_factory

BASELINES = ["Aléatoire", "Annotateurs"]

LLMs = [
    "gpt2",
    "unsloth/Meta-Llama-3-8B-bnb-4bit",
    "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3-70B-bnb-4bit",
    "unsloth/Meta-Llama-3-70B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "bigscience/bloom",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-7b1",
    "bigscience/bloomz",
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "unsloth/Qwen2.5-0.5B-bnb-4bit",
    "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-1.5B-bnb-4bit",
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-3B-bnb-4bit",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-7B-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-14B-bnb-4bit",
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-32B-bnb-4bit",
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-72B-bnb-4bit",
    "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
    "nvidia/Hymba-1.5B-Base",
    "nvidia/Hymba-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "microsoft/Phi-3.5-MoE-instruct",
    "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    "unsloth/Phi-3-small-4k-instruct-bnb-4bit",
    "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "unsloth/gemma-2-2b-bnb-4bit",
    "unsloth/gemma-2-2b-it-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
    "unsloth/gemma-2-27b-it-bnb-4bit",
    "xai-org/grok-1",
    "unsloth/Pixtral-12B-2409-bnb-4bit",
    "mistralai/Ministral-8B-Instruct-2410",
    "unsloth/Mistral-7B-v0.3-bnb-4bit",
    "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
    "CohereForAI/aya-expanse-8b",
    # "CohereForAI/aya-expanse-32b",
    "CohereForAI/aya-23-8b",
    # "CohereForAI/aya-23-35b",
    # "CohereForAI/c4ai-command-r-plus",
    # "CohereForAI/c4ai-command-r-v01",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


def filename_to_model_name(filename):
    return filename.split("/")[-1]
    # model_name = None
    # if "xlm-roberta-base" in filename:
    #     model_name = "RoBERTa-base"
    # elif "xlm-roberta-large" in filename:
    #     model_name = "RoBERTa-large"
    # elif "bert-base" in filename:
    #     model_name = "BERT"
    # elif "Llama" in filename:
    #     model_name = "Llama"
    # elif "camembert-base" in filename:
    #     model_name = "CamemBERT-base"
    # elif "camembert-large" in filename:
    #     model_name = "CamemBERT"
    #
    # if "instruct" in filename.lower() or "-it" in filename.lower():
    #     model_name += "-Instruct"
    #
    # return model_name


secrets = dotenv_values(".env")

huggingface_token = secrets["huggingface_token"]

device = torch.device("cuda")


def evaluation_loop(
    model_names: List,
    dataset: Union[Dataset, DatasetDict],
    output_file_name: str,
    compute_subcat: bool = False,
    seed: int = 42,
    class_to_predict: int = 0,
):
    model_results = {}
    for model_name in model_names:
        model, tokenizer = model_tokenizer_factory(
            model_name=model_name,
            device=device,
            token=huggingface_token,
            seed=seed,
            class_to_predict=class_to_predict,
        )

        if "instruct" in model_name.lower() or "-it" in model_name.lower():
            evaluation_fn = partial(
                evaluation_llm_instruct, tokenizer=tokenizer, model=model, device=device
            )
        elif model_name != "Aléatoire":
            # Meaning a LLM or BERT model (not necessary fine-tuned)
            evaluation_fn = partial(
                evaluation_llm, tokenizer=tokenizer, model=model, device=device
            )
        elif model_name == "Annotateurs":
            raise NotImplemented
        else:
            # Meaning the "Aléatoire" model
            evaluation_fn = partial(evaluation, model=model)

        process_dataset = dataset.map(
            evaluation_fn, desc=f"----Doing model {model_name} -----"
        )

        minimal_pair_comparison = process_dataset["train"]["minimal_pair_comparison"]
        accuracy = round(
            sum(minimal_pair_comparison) / len(minimal_pair_comparison) * 100, 2
        )

        payload = {"accuracy": accuracy}

        if compute_subcat:
            accuracies = (
                process_dataset["train"]
                .to_pandas()
                .groupby("type")["minimal_pair_comparison"]
                .mean()
                * 100
            )

            model_results_per_subcat = {
                "accuracies": {key: value for key, value in accuracies.items()}
            }

            payload.update({"accuracy_per_subcat": model_results_per_subcat})

        model_results.update({model_name: payload})

    results_path = os.path.join("./", "results")
    os.makedirs(results_path, exist_ok=True)

    with open(
        os.path.join(results_path, output_file_name),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(model_results, f, ensure_ascii=False)
