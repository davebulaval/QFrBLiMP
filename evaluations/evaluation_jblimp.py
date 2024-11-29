from datasets import load_dataset
from transformers import logging

from tools import evaluation_loop, BASELINES, LLMs

logging.set_verbosity_warning()

model_names = (
    [
        "tohoku-nlp/bert-base-japanese-v3",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
    ]
    + LLMs
    + BASELINES
)

dataset = load_dataset("polm-stability/jblimp")
dataset = dataset.rename_column("good_sentence", "sentence_good")
dataset = dataset.rename_column("bad_sentence", "sentence_bad")

output_file_name = "jblimp_results.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    output_file_name=output_file_name,
)
