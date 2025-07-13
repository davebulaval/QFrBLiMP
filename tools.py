import json
import os
from functools import partial
from typing import List, Union

import torch
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values

BASELINES = ["Aléatoire", "Annotateurs"]


def convert_name_to_unique_id(annotator):
    if "Ayman" in annotator:
        unique_id = 1
    elif "Lag" in annotator:
        unique_id = 2
    elif "Anna" in annotator:
        unique_id = 3
    elif "Abddou" in annotator:
        unique_id = 4
    elif "Emmanuelle" in annotator:
        unique_id = 5
    elif "Hili" in annotator:
        unique_id = 6
    elif "Juliette" in annotator:
        unique_id = 7
    elif "Mahamadou" in annotator:
        unique_id = 8
    elif "Folagnimi" in annotator:
        unique_id = 9
    elif "Jules" in annotator:
        unique_id = 10
    elif "Elvino" in annotator:
        unique_id = 11
    elif "Chaima" in annotator:
        unique_id = 12
    elif "Marc" in annotator:
        unique_id = 13
    elif "Jaouad" in annotator:
        unique_id = 14
    elif "ground_truth" in annotator:
        unique_id = 15
    else:
        raise Exception("Unknown annotator")
    return unique_id


def filename_to_model_name(filename):
    model_name = None
    if "xlm-roberta-base" in filename:
        model_name = "RoBERTa-base"
    elif "xlm-roberta-large" in filename:
        model_name = "RoBERTa-large"
    elif "bert-base" in filename:
        model_name = "BERT"
    elif "Llama" in filename:
        model_name = "Llama"
    elif "camembert-base" in filename:
        model_name = "CamemBERT-base"
    elif "camembert-large" in filename:
        model_name = "CamemBERT"
    return model_name