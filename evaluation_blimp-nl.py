import os.path

from datasets import load_dataset
from transformers import logging

from tools import evaluation_loop, BASELINES

logging.set_verbosity_warning()
model_names = [
    "GroNLP/bert-base-dutch-cased",
    "meta-llama/Llama-2-7b-hf",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
] + BASELINES
dataset = load_dataset(
    os.path.join("datastore", "BLiMP-NL"),
    data_files="blimp-nl.jsonl",
)

output_file_name_format = "blimp-nl_results_{}.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    output_file_name_format=output_file_name_format,
)
