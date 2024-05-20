import os
from csv import reader

import pandas as pd

root = os.path.join("datastore", "BLiMP-NL")
listfiles = os.listdir(root)

sentence_good = []
sentence_bad = []
categories = []
for file in listfiles:
    print(file)
    with open(os.path.join(root, file), "r") as read_obj:
        csv_reader = reader(read_obj)
        for idx, row in enumerate(csv_reader):

            if idx > 0:  # We skip the column header
                category = row[1]
                label = row[5]
                sentence = row[6]

                if label.lower() == "grammatical":
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

with open(os.path.join(root, "blimp-nl.jsonl"), "w") as f:
    f.write(data.to_json(orient="records", lines=True, force_ascii=False))
