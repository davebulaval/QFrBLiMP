import os
from functools import partial
from typing import List, Union

import torch
import wandb
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values

from evaluation_tools import (
    evaluation_llm_prompting,
    evaluation,
    evaluation_llm,
)
from evaluations.memory_cleanup import cleanup_memory
from factory import model_tokenizer_factory

BASELINES = ["Aléatoire", "Annotateurs"]

# All LLM we want to evaluate
llms = [
    "gpt2",
    "unsloth/llama-3-8B-bnb-4bit",
    "unsloth/llama-3-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-7b1",
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
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "microsoft/Phi-3.5-MoE-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "unsloth/gemma-2-2b-bnb-4bit",
    "unsloth/gemma-2-2b-it-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
    "unsloth/gemma-2-27b-it-bnb-4bit",
    "mistralai/Ministral-8B-Instruct-2410",
    "unsloth/Mistral-7B-v0.3-bnb-4bit",
    "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
    "CohereForAI/aya-expanse-8b",
    "CohereForAI/aya-23-8b",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]

# We evaluate all LLm in a prompting setup
llms_prompting = [llm + "_prompting" for llm in llms]

LLMs = llms + llms_prompting

secrets = dotenv_values(".env")

huggingface_token = secrets["huggingface_token"]

device = torch.device("cuda")

# To make Wandb silent
os.environ["WANDB_SILENT"] = "true"


def filename_to_model_name(filename):
    return filename.split("/")[-1]


def evaluation_loop(
    model_names: List,
    dataset: Union[Dataset, DatasetDict],
    dataset_name: str,
    lang: str,
    compute_subcat: bool = False,
    seed: int = 42,
    batch_size: int = 128,
):
    config_default_payload = {
        "dataset_name": dataset_name,
        "compute_subcat_bool": compute_subcat,
        "seed": seed,
    }

    model_results = {}
    # The class to predict, i.e. the label.
    class_to_predict = 0
    for model_name in model_names:

        wandb.init(
            project=f"minimal_pair_analysis_{lang}",
            config={"model_name": model_name, **config_default_payload},
        )
        clean_model_name = model_name.split("/")[-1]
        wandb.run.name = f"{clean_model_name}"

        model, tokenizer = model_tokenizer_factory(
            # To clean model name when we have applied a '_prompting' to it.
            model_name=(
                model_name
                if "_prompting" not in model_name
                else model_name.replace("_prompting", "")
            ),
            device=device,
            token=huggingface_token,
            seed=seed,
            class_to_predict=class_to_predict,
        )

        if "_prompting" in model_name:
            # For LLM, we also evaluate them using prompt engineering
            # Thus, we exclude model BERT LM.
            evaluation_fn = partial(
                evaluation_llm_prompting,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            map_params = {"batched": True, "batch_size": batch_size}
        elif model_name != "Aléatoire":
            # Meaning a LLM or BERT model (not necessary fine-tuned)
            # For all language model, we evaluate them using their probability
            evaluation_fn = partial(
                evaluation_llm, tokenizer=tokenizer, model=model, device=device
            )
            map_params = {"batched": False}
        elif model_name == "Annotateurs":
            raise NotImplemented
        else:
            # Meaning the "Aléatoire" model
            evaluation_fn = partial(evaluation, model=model)
            map_params = {"batched": False}

        process_dataset = dataset.map(
            evaluation_fn,
            desc=f"----Doing model {model_name} for {lang}-----",
            **map_params,
        )

        minimal_pair_comparison = process_dataset["train"]["minimal_pair_comparison"]
        accuracy = round(
            sum(minimal_pair_comparison) / len(minimal_pair_comparison) * 100, 2
        )

        payload = {"accuracy": accuracy}

        if compute_subcat:
            accuracies = round(
                process_dataset["train"]
                .to_pandas()
                .groupby("type")["minimal_pair_comparison"]
                .mean()
                * 100,
                2,
            )

            model_results_per_subcat = {key: value for key, value in accuracies.items()}

            payload.update({"accuracy_per_subcat": model_results_per_subcat})

        model_results.update({"test": payload})

        wandb.log(model_results)
        # We close the run since we will start a new one in the for loop for the next model.
        wandb.finish(exit_code=0)

        cleanup_memory(model=model, tokenizer=tokenizer)
