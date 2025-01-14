import os.path

from datasets import load_dataset
from transformers import logging

from tools import evaluation_loop, LLMs, BASELINES_FR

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.set_verbosity_warning()

model_names = (
    [
        "almanach/camembert-base",
        "almanach/camembert-large",
        "dbmdz/bert-base-french-europeana-cased",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "jpacifico/Chocolatine-14B-Instruct-DPO-v1.2",
        "jpacifico/Chocolatine-14B-Instruct-DPO-v1.2",
        "jpacifico/French-Alpaca-Llama3-8B-Instruct-v1.0",
        "OpenLLM-France/Lucie-7B-Instruct-human-data",
    ]
    + LLMs
    + BASELINES_FR
)

dataset = load_dataset(
    os.path.join("../datastore", "QFrBLiMP"), data_files="annotations.tsv", sep="\t"
)

output_file_name = "QFrBLiMP_results.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    compute_subcat=True,
    dataset_name="frblimp",
    lang="fr",
)
