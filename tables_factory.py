import json
import os
from statistics import mean

from python2latex import Document, Table

from tools import filename_to_model_name

root = "results"
saving_dir = "fig.tex"
listdir = os.listdir(root)

nl_data = {}
en_data = {}
zh_data = {}
fr_data = {}
ja_data = {}

for filename in listdir:
    model_name = filename_to_model_name(filename)
    with open(os.path.join(root, filename), "r") as file:
        data = json.load(file)
        values = data.values()

        if "blimp-nl" in filename:
            nl_data.update({model_name: list(values)[0]})
        elif "frblimp" in filename:
            if "camembert" not in model_name:
                fr_data.update({model_name: list(values)[0]})
        elif "jblimp" in filename:
            ja_data.update({model_name: list(values)[0]})
        elif "blimp" in filename:
            en_data.update({model_name: mean(list(data.get("accuracies").values()))})
        elif "climp" in filename:
            zh_data.update({model_name: list(values)[0]})

doc = Document(
    filename="res_1115", filepath=saving_dir, doc_type="article", border="10pt"
)

# Create the data
col, row = 6, 5

table = doc.new(
    Table(
        shape=(row, col),
        as_float_env=True,
        alignment=["l"] + ["c"] * (col - 1),
        caption=r"...",
        caption_pos="bottom",
    )
)

table[0, :].add_rule()
table[0, 0:] = ["", "FR", "EN", "ZH", "JA", "NL"]

table[0:, 0] = ["", "bert-base-lang", "Llama", "XLM-Roberta-base", "XLM-roberta-large"]

table[1:, 1] = list(fr_data.values())
table[1:, 2] = list(en_data.values())
table[1:, 3] = list(zh_data.values())
table[1:, 4] = list(ja_data.values())
table[1:, 5] = list(nl_data.values())

text = doc.build()
