import os
from functools import partial
from typing import Union

import torch
import wandb
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values

from evaluations.evaluation_tools import (
    evaluation_llm,
    evaluation_annotators,
    evaluation_random,
)
from model_tokenizer_factory import model_tokenizer_factory
from models import BASELINES_FR

# To make Wandb silent
os.environ["WANDB_SILENT"] = "true"


def evaluation_loop(
    model_name: str,
    dataset: Union[Dataset, DatasetDict],
    dataset_name: str,
    lang: str,
    compute_subcat: bool = False,
    device_id: str = "0",
):
    seed = 42

    secrets = dotenv_values(".env")

    huggingface_token = secrets["huggingface_token"]

    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device("cuda")

    config_default_payload = {
        "dataset_name": dataset_name,
        "compute_subcat_bool": compute_subcat,
        "seed": seed,
    }

    model_results = {}
    with torch.no_grad():
        print(f"------Evaluating {model_name}------")
        clean_model_name = model_name.split("/")[-1]

        wandb.init(
            project=f"minimal_pair_analysis_{lang}",
            config={"model_name": model_name, **config_default_payload},
            name=clean_model_name,
        )

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
        )

        if model_name not in BASELINES_FR:
            # Meaning a LLM or BERT model (not necessary fine-tuned)
            # For all language model, we evaluate them using their probability
            evaluation_fn = partial(
                evaluation_llm, tokenizer=tokenizer, model=model, device=device
            )
        elif model_name == "Annotateurs":
            evaluation_fn = partial(evaluation_annotators, model=model)
        else:
            # Meaning the "Aléatoire" model
            evaluation_fn = partial(evaluation_random, model=model)

        process_dataset = dataset.map(
            evaluation_fn,
            desc=f"----Doing model {model_name} for {lang}-----",
            batched=False,
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

        os.makedirs("predictions", exist_ok=True)
        output_dir = os.path.join("predictions", lang)
        os.makedirs(output_dir, exist_ok=True)
        process_dataset["train"].to_csv(
            os.path.join(output_dir, f"{model_name.replace('/', '_')}_predictions.tsv"),
            index=False,
            sep="\t",
        )

        model_results.update({"test": payload})

        wandb.log(model_results)
        # We close the run since we will start a new one in the for loop for the next model.
        wandb.finish(exit_code=0)
