import os

from datasets import load_dataset, Dataset

# Since pour annotations include the ground truth, we can take any annotators split to extract the truth
dir_path = os.path.join("datastore", "QFrBLiMP", "annotated")

anna = load_dataset(
    dir_path,
    data_files=["fr_blimp_50_Anna.jsonl"],
)["train"]

anna_df = anna.to_pandas()

# We extract the row where the sentence_1 equals the grammatical one, meaning the
# sentence 1 is the normative grammar response
anna_df["accept"].iloc[anna_df.query("grammatical == sentence_1").index] = ["1"]
anna_df["accept"].iloc[anna_df.query("grammatical != sentence_1").index] = ["2"]
anna_df["_annotator_id"] = "ground_truth"
anna_df["_session_id"] = "ground_truth"

for idx, row in enumerate(anna_df["sentence_2"]):
    # to handle empty sentence_2 cas in some annotations for some reasons.
    if len(row) == 0:
        anna_df.loc[idx, "sentence_2"] = [anna_df.loc[idx]["ungrammatical"]]

Dataset.from_pandas(anna_df).to_json(
    os.path.join(dir_path, "ground_truth_50.jsonl"), force_ascii=False
)


anna = load_dataset(
    dir_path,
    data_files=["fr_blimp_1711_Anna.jsonl"],
)["train"]

anna_df = anna.to_pandas()

# We extract the row where the sentence_1 equals the grammatical one, meaning the
# sentence 1 is the normative grammar response
anna_df["accept"].iloc[anna_df.query("grammatical == sentence_1").index] = ["1"]
anna_df["accept"].iloc[anna_df.query("grammatical != sentence_1").index] = ["2"]
anna_df["_annotator_id"] = "ground_truth"
anna_df["_session_id"] = "ground_truth"

for idx, row in enumerate(anna_df["sentence_2"]):
    # to handle empty sentence_2 cas in some annotations for some reasons.
    if len(row) == 0:
        anna_df.loc[idx, "sentence_2"] = [anna_df.loc[idx]["ungrammatical"]]

Dataset.from_pandas(anna_df).to_json(
    os.path.join(dir_path, "ground_truth_1711.jsonl"), force_ascii=False
)
