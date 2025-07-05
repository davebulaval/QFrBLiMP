import os

import pandas as pd
from datasets import load_dataset, tqdm
from sqlalchemy.util import OrderedDict

from annotation.annotation_tools import convert_name_to_unique_id

root_dir = os.path.join("./datastore", "QFrBLiMP")
dir_path = os.path.join(root_dir, "merge_annotated")

# We use load_dataset from HG datasets library since it handle well JSONL
annotated_data = load_dataset(
    dir_path,
    data_files=["fr_blimp_annotated.jsonl"],
)

annotated_data = annotated_data["train"]

annotated_data = annotated_data.to_pandas()

annotated_data.drop(
    [
        "label",
        "source",
        "commentaire",
        "_input_hash",
        "_task_hash",
        "options",
        "_view_id",
        "config",
        "answer",
        "_timestamp",
        "_session_id",
    ],
    axis=1,
    inplace=True,
)

keys = ["id", "grammatical", "ungrammatical", "category", "type", "subcat"]

annotators_list = [
    f"annotator_{convert_name_to_unique_id(instance_annotation['_annotator_id'].split('-')[-1])}"
    for _, instance_annotation in list(annotated_data.groupby("id"))[1][1].iterrows()
]

instances_dataset = OrderedDict({key: [] for key in keys + annotators_list})
for instance_annotations in tqdm(annotated_data.groupby("id")):
    # We prefill the generic content of the instance data
    instance_annotation = instance_annotations[1].iloc[0]

    for key in keys:
        instances_dataset.get(key).append(instance_annotation[key])

    for idx, instance_annotation in instance_annotations[1].iterrows():
        # If the grammatical sentence equals the sentence_1, it means the grammatical sentence is the first sentence,
        # thus the response is the "1" (the first sentence), and if it is not similar, then the second sentence is
        # the grammatical one, thus the response is the "2".
        ground_truth = (
            "1"
            if instance_annotation["grammatical"] == instance_annotation["sentence_1"]
            else "2"
        )

        annotation = instance_annotation["accept"][0]

        # We evaluate if the annotators gave the same response as the ground truth
        response = 1 if annotation == ground_truth else 0

        annotator_id = convert_name_to_unique_id(
            instance_annotation["_annotator_id"].split("-")[-1]
        )

        instances_dataset.get(f"annotator_{annotator_id}").append(int(response))

merge_dataset = pd.DataFrame(instances_dataset)
new_columns_order = list(merge_dataset.columns[:6]) + sorted(
    merge_dataset.columns[6:], key=lambda x: int(x.split("_")[1])
)
merge_dataset = merge_dataset[new_columns_order]
merge_dataset.to_csv(os.path.join(root_dir, "annotations.tsv"), sep="\t", index=False)
