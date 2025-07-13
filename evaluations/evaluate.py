import argparse
import os

from datasets import load_dataset

from evaluation_loop import evaluation_loop
from tools import bool_parse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to process the evaluation loop on.",
    )

    parser.add_argument(
        "--compute_subcat",
        type=bool_parse,
        default=True,
        help="Whether to compute the subcategory during the evaluation.",
    )

    parser.add_argument(
        "--device_id",
        type=str,
        default="0",
        help="Device ID tu use to be visible by CUDA.",
    )

    args = parser.parse_args()

    model_name = args.model_name
    compute_subcat = args.compute_subcat
    device_id = args.device_id

    dataset = load_dataset(
        "json",
        data_dir=os.path.join("..", "datastore", "QFrBLiMP"),
        data_files=["qfrblimp.jsonl"],
    )

    evaluation_loop(
        model_name=model_name,
        dataset=dataset,
        dataset_name="qfrcola",
        lang="fr",
        compute_subcat=compute_subcat,
        device_id=device_id,
    )
