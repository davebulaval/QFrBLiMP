import json
import os
from collections import OrderedDict
from statistics import mean

from python2latex import Document, Table

from tools import filename_to_model_name

root = "results"
saving_dir = "figs"
listdir = os.listdir(root)

nl_data = {}
en_data = {}
zh_data = {}
fr_data = {}
fr_data_per_subcat = {}
ja_data = {}

for filename in listdir:
    model_name = filename_to_model_name(filename).lower()
    with open(os.path.join(root, filename), "r") as file:
        data = json.load(file)
        values = data.values()

        if "blimp-nl" in filename:
            nl_data.update({model_name: list(values)[0]})
        elif "frblimp" in filename:
            if "per_subcat" in filename:
                fr_data_per_subcat.update({model_name: list(values)[0]})
            else:
                if "camembert" not in model_name:
                    fr_data.update({model_name: list(values)[0]})
        elif "jblimp" in filename:
            ja_data.update({model_name: list(values)[0]})
        elif "blimp" in filename:
            en_data.update({model_name: mean(list(data.get("accuracies").values()))})
        elif "climp" in filename:
            zh_data.update({model_name: list(values)[0]})

doc = Document(filename="res", filepath=saving_dir, doc_type="article", border="10pt")

# Create the data
col, row = 6, 5

table = doc.new(
    Table(
        shape=(row, col),
        as_float_env=True,
        alignment=["l"] + ["c"] * (col - 1),
        caption=r"Results",
        caption_pos="bottom",
    )
)

table[0, :].add_rule()
table[0, 0:] = ["", "FR", "EN", "ZH", "JA", "NL"]

table[0:, 0] = ["", "bert-base-lang", "Llama", "XLM-Roberta-base", "XLM-roberta-large"]

fr_data_sorted = OrderedDict(
    {
        "bert-base-lang": fr_data.get("bert-base-lang"),
        "llama": fr_data.get("llama"),
        "xlm-roberta-base": fr_data.get("xlm-roberta-base"),
        "xlm-roberta-large": fr_data.get("xlm-roberta-large"),
    }
)
en_data_sorted = OrderedDict(
    {
        "bert-base-lang": en_data.get("bert-base-lang"),
        "llama": en_data.get("llama"),
        "xlm-roberta-base": en_data.get("xlm-roberta-base"),
        "xlm-roberta-large": en_data.get("xlm-roberta-large"),
    }
)
zh_data_sorted = OrderedDict(
    {
        "bert-base-lang": zh_data.get("bert-base-lang"),
        "llama": zh_data.get("llama"),
        "xlm-roberta-base": zh_data.get("xlm-roberta-base"),
        "xlm-roberta-large": zh_data.get("xlm-roberta-large"),
    }
)
ja_data_sorted = OrderedDict(
    {
        "bert-base-lang": ja_data.get("bert-base-lang"),
        "llama": ja_data.get("llama"),
        "xlm-roberta-base": ja_data.get("xlm-roberta-base"),
        "xlm-roberta-large": ja_data.get("xlm-roberta-large"),
    }
)
nl_data_sorted = OrderedDict(
    {
        "bert-base-lang": nl_data.get("bert-base-lang"),
        "llama": nl_data.get("llama"),
        "xlm-roberta-base": nl_data.get("xlm-roberta-base"),
        "xlm-roberta-large": nl_data.get("xlm-roberta-large"),
    }
)

table[1:, 1] = list(fr_data_sorted.values())
table[1:, 2] = list(en_data_sorted.values())
table[1:, 3] = list(zh_data_sorted.values())
table[1:, 4] = list(ja_data_sorted.values())
table[1:, 5] = list(nl_data_sorted.values())

text = doc.build()

## Fr per categopry
doc = Document(
    filename="res_fr_per_subcat", filepath=saving_dir, doc_type="article", border="10pt"
)

# Create the data
col, row = len(fr_data_per_subcat.get("llama")) + 2, len(fr_data_per_subcat) + 1

table = doc.new(
    Table(
        shape=(row, col),
        as_float_env=True,
        alignment=["l"] + ["c"] * (col - 2) + ["|c"],
        caption=r"Results",
        caption_pos="bottom",
    )
)

table[0, :].add_rule()
table[0, 0:] = [""] + list(fr_data_per_subcat.get("llama").keys()) + ["Average"]

models = [
    "bert-base-lang",
    "CamemBERT-large",
    "Llama",
    "XLM-Roberta-base",
    "XLM-roberta-large",
]
table[0:, 0] = [""] + models

for idx, model in enumerate(models):
    table[idx + 1, 1:] = list(fr_data_per_subcat.get(model.lower()).values()) + [
        mean(list(fr_data_per_subcat.get(model.lower()).values()))
    ]

text = doc.build()
