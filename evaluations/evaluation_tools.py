# REF: https://huggingface.co/spaces/zoebat20/BLiMP/blob/main/app.py
import os
import random
import re

import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import logging

en_names = [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
    "drop_argument",
    "ellipsis_n_bar_1",
    "ellipsis_n_bar_2",
    "existential_there_object_raising",
    "existential_there_quantifiers_1",
    "existential_there_quantifiers_2",
    "existential_there_subject_raising",
    "expletive_it_object_raising",
    "inchoative",
    "intransitive",
    "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
    "npi_present_2",
    "only_npi_licensor_present",
    "only_npi_scope",
    "passive_1",
    "passive_2",
    "principle_A_c_command",
    "principle_A_case_1",
    "principle_A_case_2",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1",
    "superlative_quantifiers_2",
    "tough_vs_raising_1",
    "tough_vs_raising_2",
    "transitive",
    "wh_island",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
]

logging.set_verbosity_warning()

max_length = 2056

prompt = (
    "Here are two English sentences: 1) {} 2) {} Which sentence is a better English sentence? "
    "ONLY respond in ENGLISH with EITHER the number '1' or the number '2' as your answer. "
    "DO NOT RESPONSE ANY OTHER NUMBER THAN '1' or '2'."
    "NO YAPPING. Answer: "
)

regex = r"Answer: ([1|2])"


def evaluation_llm_prompting(rows, tokenizer, model, device):
    with torch.no_grad():
        # Correct sentence processing
        grammatical = rows["sentence_good"]
        ungrammatical = rows["sentence_bad"]

        # We sample the sentence to limit the risk of logical pattern inference by the LLM.
        choices = random.choices([1, 2], k=len(grammatical))
        sentences_1 = [
            grammatical if choice == "1" else ungrammatical
            for grammatical, ungrammatical, choice in zip(
                grammatical, ungrammatical, choices
            )
        ]
        sentences_2 = [
            ungrammatical if choice == "2" else grammatical
            for grammatical, ungrammatical, choice in zip(
                grammatical, ungrammatical, choices
            )
        ]

        formated_prompts = [
            prompt.format(sentence_1, sentence_2)
            for sentence_1, sentence_2 in zip(sentences_1, sentences_2)
        ]

        input_ids = tokenizer(
            formated_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).input_ids.to(device)

        outputs = model.generate(input_ids, max_new_tokens=1)
        decoded_batch_response = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        # Extract the answers
        answers = []
        for sentence in decoded_batch_response:
            try:
                answer = re.findall(regex, sentence)[0]
                answers.append(answer)
            except IndexError:
                # Case for direct response is a direct "1" or "2" (e.g. flan).
                try:
                    # We test if it is a possible int
                    answer = int(sentence)
                    answers.append(answer)
                except TypeError:
                    # Case the response is a float
                    answer = float(sentence)
                    answers.append(answer)
                except ValueError:
                    # Case where the model did not respond anything relevant.
                    answers.append("-1")

        return {
            # We use 1-0 to replace bool to simplify analysis later.
            "minimal_pair_comparison": [
                1 if int(answer) == truth else 0
                for answer, truth in zip(answers, choices)
            ]
        }


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
