import json
import subprocess

from tqdm import tqdm

from models import LLMs, BASELINES_FR

model_names = (
    [
        "google-bert/bert-base-uncased",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "almanach/camembert-base",
        "almanach/camembert-large",
        "dbmdz/bert-base-french-europeana-cased",
        "jpacifico/Chocolatine-2-14B-Instruct-v2.0.3",
        "jpacifico/Chocolatine-14B-Instruct-DPO-v1.2",
        "jpacifico/French-Alpaca-Llama3-8B-Instruct-v1.0",
        "OpenLLM-France/Lucie-7B",
        "OpenLLM-France/Lucie-7B-Instruct",
        "OpenLLM-France/Lucie-7B-Instruct-human-data",
        "OpenLLM-France/Claire-7B-FR-Instruct-0.1",
    ]
    + LLMs
    + BASELINES_FR
)

# We create a new model JSON file to write the number of params.
with open("models_size.json", "w", encoding="utf-8") as file:
    json.dump({}, file, ensure_ascii=False)

for model_name in tqdm(model_names):
    subprocess.run(
        f"python3 get_model_size.py {model_name}",
        shell=True,
    )
