import os
from csv import reader

import pandas as pd

root = os.path.join("../datastore", "CLiMP")
listfiles = os.listdir(root)

sentence_good = []
sentence_bad = []
categories = []
for file in listfiles:
    category = file.strip(".csv")
    with open(os.path.join(root, file), "r") as read_obj:
        csv_reader = reader(read_obj)
        for idx, row in enumerate(csv_reader):
            if idx > 0:  # We skip the column header
                if len(row) > 1:
                    # Means we have validation
                    label = int(row[-1])
                    sentence = row[-2]
                else:
                    # Means we only have the sentence
                    sentence = row[0]
                    label = int(
                        idx % 2 != 0
                    )  # Means if it is not equal to 0, it is a unpair number, thus the good sentence

                if bool(label):
                    # It is a good sentence
                    sentence_good.append(sentence)
                    # We add it there since we only need as many as the number of pair
                    categories.append(category)
                else:
                    # It is a bad sentence
                    sentence_bad.append(sentence)

        assert len(sentence_good) == len(sentence_bad)
        assert len(categories) == len(sentence_good)

paired_data = {
    "sentence_good": sentence_good,
    "sentence_bad": sentence_bad,
    "category": categories,
}

data = pd.DataFrame(paired_data)

with open(os.path.join(root, "climp.jsonl"), "w") as f:
    f.write(data.to_json(orient="records", lines=True, force_ascii=False))
