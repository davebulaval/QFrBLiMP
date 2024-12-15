# REF: https://huggingface.co/spaces/zoebat20/BLiMP/blob/main/app.py
import random
import re

import torch
from transformers import logging

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
            "minimal_pair_comparison": [
                int(answer) == truth for answer, truth in zip(answers, choices)
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
        perplexity_correct = torch.exp(score_correct).item()

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
        perplexity_incorrect = torch.exp(score_incorrect).item()

        # The smallest perplexity = the lowest probability
        # (True/False, True if perplexity_correct is lower than perplexity_incorrect)
        return {"minimal_pair_comparison": perplexity_correct < perplexity_incorrect}


def evaluation_random(row, model):
    correct = row["sentence_good"]
    label = 1  # The good sentence is the unitary label (i.e. 1) sentence.

    prediction = model(correct, labels=label)

    return {"minimal_pair_comparison": prediction == label}


def evaluation_annotators(row, model):
    votes = [value for key, value in row.items() if "annotator" in key]

    # Since the last annotator is a "ground truth",
    # we remove this annotation and use it as the label.
    votes = votes[:-1]
    label = votes[-1]

    prediction = model(votes=votes, labels=label)

    return {"minimal_pair_comparison": prediction == label}
