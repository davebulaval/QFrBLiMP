# We use load_dataset from HG datasets library since it handle well JSONL
import os
from collections import Counter

from datasets import load_dataset

names = [
    "Ayman",
    # "Lag", # Doublon de Marc
    "Anna",
    # "Abdou", # Doublon de Mahamadou
    "Emmanuelle",
    "Hili",
    "Juliette",
    "Mahamadou",
    "Folagnimi",
    "Jules",
    "Elvino",
    "Chaima",
    "Marc",
    "Jaouad",
]

truth = load_dataset(
    os.path.join("datastore", "QFrBLiMP", "annotated", "Part 2"),
    data_files=["ground_truth.jsonl"],
)
truth_all_ids = truth["train"]["id"]

missing_annotations = {}
for name in names:
    dir_path = os.path.join("datastore", "QFrBLiMP")
    annotated_data_dir = os.path.join(dir_path, "annotated", "Part 2")

    file_name = f"fr_blimp_1711_{name}.jsonl"

    file_exist = os.path.exists(os.path.join(annotated_data_dir, file_name))

    if file_exist:
        all_ids = []
        annotator_missing = []
        annotated_data = load_dataset(
            annotated_data_dir,
            data_files=[file_name],
        )

        for data in annotated_data["train"]:
            try:
                data["sentence_2"][0]
            except:
                sentence_id = data["id"]
                annotator_missing.append(sentence_id)
                # print(data)
            all_ids.append(data["id"])
        missing_annotations.update({name: annotator_missing})
        duplicates = [k for k, v in Counter(all_ids).items() if v > 1]

        print(name)
        print("Duplicate:", duplicates)
        print(
            "Duplicate in sentence 2?", set(annotator_missing).intersection(duplicates)
        )
        print(
            "Missing:",
            set(truth_all_ids) - set(all_ids),
            len(list(set(truth_all_ids) - set(all_ids))),
        )
        missing_id = list(set(truth_all_ids) - set(all_ids))
        if len(missing_id) > 0:
            print(missing_id)
        print("Missing sentence 2:", annotator_missing)

for sentence_truth, sentence_annotators in zip(
    truth["train"][annotator_missing]["sentence_1"],
    annotated_data["train"][annotator_missing]["sentence_1"],
):
    if sentence_truth == sentence_annotators:
        print("oui")
    else:
        print("non")
# print("a")
