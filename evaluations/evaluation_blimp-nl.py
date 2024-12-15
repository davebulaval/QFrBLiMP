import os.path

from datasets import load_dataset
from transformers import logging

from tools import evaluation_loop, BASELINES, LLMs

logging.set_verbosity_warning()
model_names = (
    [
        "GroNLP/bert-base-dutch-cased",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
    ]
    + LLMs
    + BASELINES
)
dataset = load_dataset(
    os.path.join("../datastore", "BLiMP-NL"),
    data_files="blimp-nl.jsonl",
)

output_file_name = "blimp-nl_results.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    dataset_name="blimp-nl",
    lang="nl",
)
