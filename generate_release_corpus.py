import os
import random

import pandas as pd
from tqdm import tqdm

dataset_path = os.path.join("datastore", "QFrBLiMP", "annotations.tsv")

qfrblimp = pd.read_csv(dataset_path, sep="\t")

qfrblimp = qfrblimp.drop(columns=["id", "annotator_15"])

annotators_columns = [f"annotator_{i}" for i in range(1, 15) if i not in [2, 4]]

labels = round(qfrblimp.loc[:, annotators_columns].sum(axis=1) / 12)

qfrblimp.insert(len(qfrblimp.columns), "label", labels)

seed = 42
random.seed(seed)
for row in tqdm(qfrblimp.iterrows(), total=len(qfrblimp)):
    idx_of_good_sentence = random.randint(a=0, b=1)
    if idx_of_good_sentence == 0:
        # We keep the actual setup
        continue
    elif idx_of_good_sentence == 1:
        # We invert the position
        sentence_b = row[1]["sentence_a"]
        sentence_a = row[1]["sentence_b"]

        annotations = row[1][annotators_columns]
        valid_annotations = annotations.sum()
        transfert_value = len(annotations) - valid_annotations
        inverse_annotations = [
            0 if annotation == 1 else 1 for annotation in annotations
        ]
        assert sum(inverse_annotations) == transfert_value

        label = row[1]["label"]
        inverse_label = 0 if label == 1 else 1

        row[1]["sentence_a"] = sentence_a
        row[1]["sentence_b"] = sentence_b

        row[1]["label"] = inverse_label

        row[1][annotators_columns] = inverse_annotations

        qfrblimp.loc[row[0]] = row[1]


with open("datastore/QFrBLiMP/qfrblimp.jsonl", "w") as f:
    f.write(qfrblimp.to_json(orient="records", lines=True, force_ascii=False))
