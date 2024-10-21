# We use load_dataset from HG datasets library since it handle well JSONL
import os

from datasets import load_dataset, Dataset

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
    os.path.join("datastore", "FrBLiMP", "annotated", "Part 2"),
    data_files=["ground_truth.jsonl"],
)
truth_all_ids = truth["train"]["id"]

mapping = {
    idx: sentence_2
    for idx, sentence_2 in zip(truth["train"]["id"], truth["train"]["sentence_2"])
}

missing_annotations = {}
for name in names:
    dir_path = os.path.join("datastore", "FrBLiMP")
    annotated_data_dir = os.path.join(dir_path, "annotated", "Part 2")

    file_name = f"fr_blimp_1711_{name}.jsonl"

    file_exist = os.path.exists(os.path.join(annotated_data_dir, file_name))

    if file_exist:
        annotated_data = load_dataset(
            annotated_data_dir,
            data_files=[file_name],
        )

        def fix_sentence_2(row):
            try:
                row["sentence_2"][0]
                payload = {"sentence_2": row["sentence_2"]}
            except:
                sentence_id = row["id"]
                payload = {"sentence_2": mapping.get(sentence_id)}
            return payload

        annotated_data = annotated_data["train"].map(fix_sentence_2)

        drop_duplicated = annotated_data.to_pandas().drop_duplicates(["id"])
        Dataset.from_pandas(drop_duplicated, preserve_index=False).to_json(
            os.path.join(annotated_data_dir, file_name), force_ascii=False
        )
