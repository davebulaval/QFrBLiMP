from datasets import load_dataset
from transformers import logging

from tools import evaluation_loop, BASELINES, LLMs

logging.set_verbosity_warning()

model_names = (
    [
        "google-bert/bert-base-chinese",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
    ]
    + LLMs
    + BASELINES
)

dataset = load_dataset("../datastore/CLiMP", data_files=["climp.jsonl"])

output_file_name = "climp_results.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    output_file_name=output_file_name,
    dataset_name="cblimp",
)
