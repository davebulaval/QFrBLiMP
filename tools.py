import json
import os
from functools import partial
from typing import List, Union

import torch
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values

from evaluation_tools import evaluation_llm, evaluation, evaluation_llm_instruct
from factory import model_tokenizer_factory

BASELINES = ["Aléatoire", "Annotateurs"]

LLMs = [
    "gpt2",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B",
    "meta-llama/Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "bigscience/bloom",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-7b1",
    "bigscience/bloomz",
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",
    "nvidia/Hymba-1.5B-Base",
    "nvidia/Hymba-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3.5-MoE-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128K-instruct",
    "microsoft/Phi-3-small-4k-instruct",
    "microsoft/Phi-3-small-128K-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3-medium-128K-instruct",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",
    "xai-org/grok-1",
    "mistralai/Pixtral-12B-2409",
    "mistralai/Pixtral-Large-Instruct-2411",
    "mistralai/Mistral-Large-Instruct-2411",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "CohereForAI/aya-expanse-8b",
    "CohereForAI/aya-expanse-32b",
    "CohereForAI/aya-23-8b",
    "CohereForAI/aya-23-35b",
    "CohereForAI/c4ai-command-r-plus",
    "CohereForAI/c4ai-command-r-v01",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


def convert_name_to_unique_id(annotator):
    if "Ayman" in annotator:
        unique_id = 1
    elif "Lag" in annotator:
        unique_id = 2
    elif "Anna" in annotator:
        unique_id = 3
    elif "Abddou" in annotator:
        unique_id = 4
    elif "Emmanuelle" in annotator:
        unique_id = 5
    elif "Hili" in annotator:
        unique_id = 6
    elif "Juliette" in annotator:
        unique_id = 7
    elif "Mahamadou" in annotator:
        unique_id = 8
    elif "Folagnimi" in annotator:
        unique_id = 9
    elif "Jules" in annotator:
        unique_id = 10
    elif "Elvino" in annotator:
        unique_id = 11
    elif "Chaima" in annotator:
        unique_id = 12
    elif "Marc" in annotator:
        unique_id = 13
    elif "Jaouad" in annotator:
        unique_id = 14
    elif "ground_truth" in annotator:
        unique_id = 15
    else:
        raise Exception("Unknown annotator")
    return unique_id


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
                .groupby("type")["minimal_pair_comparison"]
                .mean()
                * 100
            )

            model_results_per_subcat = {
                "accuracies": {key: value for key, value in accuracies.items()}
            }

            output_file_name_format = output_file_name_format.replace(
                "_results", "_results_per_type"
            )
            with open(
                os.path.join("results", output_file_name_format.format(model_name)),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(model_results_per_subcat, f, ensure_ascii=False)
