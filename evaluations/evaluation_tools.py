# REF: https://huggingface.co/spaces/zoebat20/BLiMP/blob/main/app.py
import os

import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import logging

logging.set_verbosity_warning()

max_length = 2056


def evaluation_llm(row, tokenizer, model, device):
    with torch.no_grad():
        # Correct sentence processing
        correct = row["sentence_good"]
        correct_tokenized = tokenizer(
            correct,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)
        out_correct = model(correct_tokenized, labels=correct_tokenized.clone())
        score_correct = out_correct["loss"]
        perplexity_correct = torch.exp(score_correct).detach().item()

        # Incorrect sentence processing
        incorrect = row["sentence_bad"]
        incorrect_tokenized = tokenizer(
            incorrect,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)
        out_incorrect = model(incorrect_tokenized, labels=incorrect_tokenized.clone())
        score_incorrect = out_incorrect["loss"]
        perplexity_incorrect = torch.exp(score_incorrect).detach().item()

        # The smallest perplexity = the lowest probability
        # (True/False, True if perplexity_correct is lower than perplexity_incorrect)
        # We use 1-0 to replace bool to simplify analysis later.
        return {
            "minimal_pair_comparison": (
                1 if perplexity_correct < perplexity_incorrect else 0
            )
        }


def evaluation_random(row, model):
    correct = row["sentence_good"]
    label = 1  # The good sentence is the unitary label (i.e. 1) sentence.

    prediction = model(correct, labels=label)

    # We use 1-0 to replace bool to simplify analysis later.
    return {"minimal_pair_comparison": 1 if prediction == label else 0}


def evaluation_annotators(row, model):
    votes = [value for key, value in row.items() if "annotator" in key]

    # Since the last annotator is a "ground truth",
    # we remove this annotation and use it as the label.
    votes = votes[:-1]
    label = votes[-1]

    prediction = model(votes=votes, labels=label)

    # We use 1-0 to replace bool to simplify analysis later.
    return {"minimal_pair_comparison": 1 if prediction == label else 0}


def filename_to_model_name(filename):
    return filename.split("/")[-1]


def dataset_factory(lang):
    if lang == "en":
        dataset = DatasetDict(
            {
                "train": concatenate_datasets(
                    [load_dataset("nyu-mll/blimp", name)["train"] for name in en_names]
                )
            }
        )
        dataset_name = "blimp"
    elif lang == "nl":
        dataset = load_dataset(
            os.path.join("../datastore", "BLiMP-NL"),
            data_files="blimp-nl.jsonl",
        )
        dataset_name = "blimp-nl"
    elif lang == "zh":
        dataset = load_dataset("../datastore/CLiMP", data_files=["climp.jsonl"])
        dataset_name = "climp"
    elif lang == "fr":
        dataset = load_dataset(
            os.path.join("../datastore", "QFrBLiMP"),
            data_files="annotations.tsv",
            sep="\t",
        )
        dataset_name = "frblimp"
    elif lang == "ja":
        dataset = load_dataset("polm-stability/jblimp")
        dataset = dataset.rename_column("good_sentence", "sentence_good")
        dataset = dataset.rename_column("bad_sentence", "sentence_bad")
        dataset_name = "jblimp"
    else:
        raise ValueError(f"{lang} is not supported.")

    return dataset, dataset_name
