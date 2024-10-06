import json
import os
import string

import pandas as pd
from datasets import load_dataset
from prodigy_iaa.measures import calculate_agreement

dir_path = os.path.join("datastore", "FrBLiMP", "annotated")

# We use load_dataset from HG datasets library since it handle well JSONL
annotated_data = load_dataset(
    dir_path,
    data_files=["fr_blimp_annotated.jsonl"],
)

# Each annotation per annotator is a separated row
# "train" since by default HG set it as train
extracted_annotations = {}


def df_to_reliability(df):
    # Taken from https://github.com/pmbaumgartner/prodigy-iaa/
    # blob/a802f493e09473e6725367340dd316f44cd956d9/tests/conftest.py#L7C1-L12C28
    """Converts a DataFrame to a list of lists, where missing values
    are replaced with `None`."""
    df = df.where(pd.notnull(df), None)
    reliability_data = df.values.tolist()
    return reliability_data


names = [
    "Ayman",
    "Lag",
    "Anna",
    "Abdou",
    "Emmanuelle",
    "Hili",
    "Juliette",
    "Mahamadou",
    "Folagnimi",
    "Jules",
    "Elvino",
    "Chaima",
    "Marc",
]


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
    else:
        raise Exception("Unknown annotator")
    return unique_id


sentence_id_mapping = []
for row in annotated_data["train"]:
    # 2 equal the sentence 2
    # 1 equal the sentence 1
    acceptability_judgment_annotation = int(row["accept"][0])

    annotator = row["_session_id"]
    annotator_numerical_id = convert_name_to_unique_id(annotator)

    row_data = {
        "acceptability_judgment": acceptability_judgment_annotation,
    }

    annotation_hash = row["_input_hash"]
    if extracted_annotations.get(annotation_hash) is None:
        extracted_annotations.update(
            {annotation_hash: {str(annotator_numerical_id): row_data}}
        )
    else:
        annotation_datas = extracted_annotations.get(annotation_hash)
        annotation_datas.update({str(annotator_numerical_id): row_data})
        extracted_annotations.update({annotation_hash: annotation_datas})

    sentence_1 = row["sentence_1"]
    sentence_2 = row["sentence_2"][0]
    sentence_id_mapping.append(
        {
            "sentence_id": annotation_hash,
            "sentence_1": sentence_1,
            "sentence_2": sentence_2,
        }
    )

task_annotations = []
for sentence_idx, sentence in extracted_annotations.items():
    annotations = []
    for annotator_annotations in sentence.values():
        annotations.append(annotator_annotations.get("acceptability_judgment", None))

    task_annotations.append([sentence_idx] + annotations)

# We use the len of annotations to have the proper number of annotators
task_df = pd.DataFrame(
    task_annotations,
    columns=["sentence_id"]
    + [f"Annotator {letter}" for letter in string.ascii_lowercase[: len(annotations)]],
)

# Since it will consider the sentence id as an evaluator
agreement_df = task_df.drop("sentence_id", axis=1)

agreement = calculate_agreement(df_to_reliability(agreement_df))
print(agreement)
with open(os.path.join("results", f"agreement.json"), "w") as f:
    json.dump(agreement, f)

task_df.to_json(os.path.join("results", f"agreement.json"), force_ascii=False)
