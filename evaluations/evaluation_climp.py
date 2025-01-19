import os
import subprocess

from transformers import logging

from models import LLMs, BASELINES

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

lang = "zh"
compute_subcat = False
device_id = "0"
for model_name in model_names:

    subprocess.run(
        f"python3 evaluate.py {model_name} {lang} --compute_subcat {compute_subcat} --device_id {device_id}",
        shell=True,
    )
