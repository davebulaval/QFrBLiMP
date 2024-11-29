import json
import os
from collections import OrderedDict
from statistics import mean

from python2latex import Document, Table

from evaluations.tools import filename_to_model_name

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
    model_name = filename_to_model_name(filename)
    with open(os.path.join(root, filename), "r") as file:
        data = json.load(file)
        values = data.values()

        if "blimp-nl" in filename:
            nl_data.update({model_name: list(values)[0]})
        elif "QFrBLiMP" in filename:
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

table[0:, 0] = ["", "BERT", "Llama", "RoBERTa-base", "RoBERTa-large"]

fr_data_sorted = OrderedDict(
    {
        "BERT": fr_data.get("BERT"),
        "Llama": fr_data.get("Llama"),
        "RoBERTa-base": fr_data.get("RoBERTa-base"),
        "RoBERTa-large": fr_data.get("RoBERTa-large"),
    }
)
en_data_sorted = OrderedDict(
    {
        "BERT": en_data.get("BERT"),
        "Llama": en_data.get("Llama"),
        "RoBERTa-base": en_data.get("RoBERTa-base"),
        "RoBERTa-large": en_data.get("RoBERTa-large"),
    }
)
zh_data_sorted = OrderedDict(
    {
        "BERT": zh_data.get("BERT"),
        "Llama": zh_data.get("Llama"),
        "RoBERTa-base": zh_data.get("RoBERTa-base"),
        "RoBERTa-large": zh_data.get("RoBERTa-large"),
    }
)
ja_data_sorted = OrderedDict(
    {
        "BERT": ja_data.get("BERT"),
        "Llama": ja_data.get("Llama"),
        "RoBERTa-base": ja_data.get("RoBERTa-base"),
        "RoBERTa-large": ja_data.get("RoBERTa-large"),
    }
)
nl_data_sorted = OrderedDict(
    {
        "BERT": nl_data.get("BERT"),
        "Llama": nl_data.get("Llama"),
        "RoBERTa-base": nl_data.get("RoBERTa-base"),
        "RoBERTa-large": nl_data.get("RoBERTa-large"),
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
col, row = len(fr_data_per_subcat) + 1, len(fr_data_per_subcat.get("Llama")) + 2

table = doc.new(
    Table(
        shape=(row, col),
        as_float_env=True,
        alignment=["l"] + ["c"] * (col - 1),
        caption=r"Results",
        caption_pos="bottom",
        float_format=".1f",
    )
)

table[0, :].add_rule()

models = [
    "BERT",
    "CamemBERT",
    "Llama",
    "RoBERTa-base",
    "RoBERTa-large",
]
table[0, 0:] = [""] + models

table[0:, 0] = [""] + list(fr_data_per_subcat.get("Llama").keys()) + ["Average"]

for idx, model in enumerate(models):
    table[1:, idx + 1] = list(fr_data_per_subcat.get(model).values()) + [
        mean(list(fr_data_per_subcat.get(model).values()))
    ]

table[-2, :].add_rule()
text = doc.build()
