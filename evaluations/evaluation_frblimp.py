import os.path

from datasets import load_dataset
from transformers import logging

from tools import evaluation_loop, BASELINES, LLMs

logging.set_verbosity_warning()

model_names = (
    [
        "almanach/camembert-base",
        "almanach/camembert-large",
        "dbmdz/bert-base-french-europeana-cased",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
    ]
    + LLMs
    + BASELINES
)

dataset = load_dataset(
    os.path.join("../datastore", "QFrBLiMP"), data_files="dataset.tsv", sep="\t"
)

output_file_name = "QFrBLiMP_results.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    compute_subcat=True,
    dataset_name="frblimp",
)
