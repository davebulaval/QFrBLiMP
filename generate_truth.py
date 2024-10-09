import os

from datasets import load_dataset, Dataset
import pandas as pd

# Since pour annotations include the ground truth, we can take any annotators split to extract the truth
dir_path = os.path.join("datastore", "FrBLiMP", "annotated")

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

Dataset.from_pandas(anna_df).to_json(
    os.path.join(dir_path, "ground_truth.jsonl"), force_ascii=False
)
