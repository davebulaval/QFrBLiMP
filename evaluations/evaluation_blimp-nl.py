import subprocess

from transformers import logging

from models import LLMs, BASELINES

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

lang = "nl"
compute_subcat = False
device_id = "0"
for model_name in model_names:
    subprocess.run(
        f"python3 evaluate.py {model_name} {lang} --compute_subcat {compute_subcat} --device_id {device_id}",
        shell=True,
    )
