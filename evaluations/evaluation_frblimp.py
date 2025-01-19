import os

from datasets import load_dataset
from transformers import logging

from evaluations.models import LLMs, BASELINES_FR
from tools import evaluation_loop

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


for model_name in model_names:
    evaluation_loop(
        model_name=model_name,
        dataset=dataset,
        compute_subcat=True,
        dataset_name="frblimp",
        lang="fr",
    )
