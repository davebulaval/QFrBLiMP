import os
import subprocess

from transformers import logging

from models import LLMs, BASELINES

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.set_verbosity_warning()

model_names = (
    [
        "tohoku-nlp/bert-base-japanese-v3",
        "tohoku-nlp/bert-large-japanese-v2",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
    ]
    + LLMs
    + BASELINES
)

for model_name in model_names:
    lang = "ja"
    compute_subcat = False

    subprocess.run(
        f"python3 evaluate.py {model_name} {lang} --compute_subcat {compute_subcat}",
        shell=True,
    )
