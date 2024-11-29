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
    os.path.join("../datastore", "QFrBLiMP"), data_files="complete.tsv"
)

output_file_name_format = "QFrBLiMP_results_{}.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    output_file_name_format=output_file_name_format,
    compute_subcat=True,
)
