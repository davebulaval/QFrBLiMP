import subprocess

from transformers import logging

from models import LLMs, BASELINES_FR

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
lang = "fr"
compute_subcat = True
device_id = "0"
for model_name in model_names:
    subprocess.run(
        f"python3 evaluate.py {model_name} {lang} --compute_subcat {compute_subcat} --device_id {device_id}",
        shell=True,
    )
