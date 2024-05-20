# REF: https://huggingface.co/spaces/zoebat20/BLiMP/blob/main/app.py

import torch


def evaluation(row, tokenizer, model, device):
    with torch.no_grad():
        # Correct sentence processing
        correct = row["sentence_good"]
        correct_tokenized = tokenizer(correct, return_tensors="pt")["input_ids"].to(
            device
        )
        out_correct = model(correct_tokenized, labels=correct_tokenized.clone())
        score_correct = out_correct["loss"]
        perplexity_correct = torch.exp(score_correct).item()

        # Incorrect sentence processing
        incorrect = row["sentence_bad"]
        incorrect_tokenized = tokenizer(incorrect, return_tensors="pt")["input_ids"].to(
            device
        )
        out_incorrect = model(incorrect_tokenized, labels=incorrect_tokenized.clone())
        score_incorrect = out_incorrect["loss"]
        perplexity_incorrect = torch.exp(score_incorrect).item()

        # The smallest perplexity = the lowest probability
        return {"minimal_pair_comparison": perplexity_correct < perplexity_incorrect}
