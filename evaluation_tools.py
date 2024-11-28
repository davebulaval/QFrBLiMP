# REF: https://huggingface.co/spaces/zoebat20/BLiMP/blob/main/app.py
import random

import torch
from transformers import logging

logging.set_verbosity_warning()

max_length = 2056

prompt = """Here are two English sentences: 1) {} 2) {}. Which sentence is a better English sentence? Respond with either 1 or 2 as your answer. 
            Answer:
        """


def evaluation_llm_instruct(row, tokenizer, model, device):
    with torch.no_grad():
        # Correct sentence processing
        grammatical = row["sentence_good"]
        ungrammatical = row["sentence_bad"]

        # We sample the sentence to limit the risk of logical pattern inference by the LLM.
        choices = [grammatical, ungrammatical]
        sentence_1 = random.choice(choices)
        sentence_2 = [choice for choice in choices if choice != sentence_1]

        prompt.format(sentence_1, sentence_2)

        prompt_tokenized = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )["input_ids"].to(device)
        out_prompt = model(prompt_tokenized, labels=prompt_tokenized.clone())

        # extract the answers
        answer = out_prompt[0].argmax().item()
        return {"minimal_pair_comparison": answer}


def evaluation_llm(row, tokenizer, model, device):
    with torch.no_grad():
        # Correct sentence processing
        correct = row["sentence_good"]
        correct_tokenized = tokenizer(
            correct, return_tensors="pt", truncation=True, max_length=max_length
        )["input_ids"].to(device)
        out_correct = model(correct_tokenized, labels=correct_tokenized.clone())
        score_correct = out_correct["loss"]
        perplexity_correct = torch.exp(score_correct).item()

        # Incorrect sentence processing
        incorrect = row["sentence_bad"]
        incorrect_tokenized = tokenizer(
            incorrect, return_tensors="pt", truncation=True, max_length=max_length
        )["input_ids"].to(device)
        out_incorrect = model(incorrect_tokenized, labels=incorrect_tokenized.clone())
        score_incorrect = out_incorrect["loss"]
        perplexity_incorrect = torch.exp(score_incorrect).item()

        # The smallest perplexity = the lowest probability
        # (True/False, True if perplexity_correct is lower than perplexity_incorrect)
        return {"minimal_pair_comparison": perplexity_correct < perplexity_incorrect}


def evaluation(row, model):
    correct = row["sentence_good"]
    label = 0  # The good sentence is the first sentence. Thus, if the prediction is 0, it mean it has predicted the
    # "good" sentence as the well written sentence.

    prediction = model(correct, labels=label)

    return {"minimal_pair_comparison": prediction == label}
