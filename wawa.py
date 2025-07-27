import os.path
from collections import OrderedDict

import numpy as np
import pandas as pd
from datasets import load_dataset

data = load_dataset(
    "json",
    data_dir=os.path.join("datastore", "QFrBLiMP", "release"),
    data_files=["qfrblimp.jsonl"],
)

data = data["train"].to_pandas()
data = data.drop(columns=["sentence_a", "sentence_b", "category", "subcat"])

group_by_data = data.groupby("type")

rows = OrderedDict(
    {
        "phenomenon_1": [],
        "wawa_1": [],
        "phenomenon_2": [],
        "wawa_2": [],
        "phenomenon_3": [],
        "wawa_3": [],
        "phenomenon_4": [],
        "wawa_4": [],
    }
)
for group_name, df_group in group_by_data:
    wawa_score = round(
        df_group.apply(lambda x: np.where(x == df_group["label"], 1, 0))
        .drop(columns=["label", "type"])
        .sum(axis=1)
        .apply(lambda x: x / 12 * 100)
        .mean(),
        2,
    )
    if group_name <= 5:
        rows.get("phenomenon_1").append(str(group_name))
        rows.get("wawa_1").append(wawa_score)
    elif group_name <= 10:
        rows.get("phenomenon_2").append(str(group_name))
        rows.get("wawa_2").append(wawa_score)
    elif group_name <= 15:
        rows.get("phenomenon_3").append(str(group_name))
        rows.get("wawa_3").append(wawa_score)
    elif group_name <= 20:
        rows.get("phenomenon_4").append(str(group_name))
        rows.get("wawa_4").append(wawa_score)

print(pd.DataFrame(rows).to_latex(index=False, float_format="%.2f"))
print(
    pd.DataFrame(rows)
    .drop(columns=["phenomenon_1", "phenomenon_2", "phenomenon_3", "phenomenon_4"])
    .mean()
    .mean()
)
