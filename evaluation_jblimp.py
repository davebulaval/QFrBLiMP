from datasets import load_dataset
from transformers import logging

from tools import evaluation_loop, BASELINES

logging.set_verbosity_warning()

model_names = [
    "tohoku-nlp/bert-base-japanese-v3",
    "meta-llama/Llama-2-7b-hf",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
] + BASELINES

dataset = load_dataset("polm-stability/jblimp")
dataset = dataset.rename_column("good_sentence", "sentence_good")
dataset = dataset.rename_column("bad_sentence", "sentence_bad")

output_file_name_format = "jblimp_results_{}.json"

evaluation_loop(
    model_names=model_names,
    dataset=dataset,
    output_file_name_format=output_file_name_format,
)
