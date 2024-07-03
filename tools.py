def filename_to_model_name(filename):
    model_name = None
    if "xlm-roberta-base" in filename:
        model_name = "RoBERTa-base"
    elif "xlm-roberta-large" in filename:
        model_name = "RoBERTa-large"
    elif "bert-base" in filename:
        model_name = "BERT"
    elif "Llama" in filename:
        model_name = "Llama"
    elif "camembert-base" in filename:
        model_name = "CamemBERT-base"
    elif "camembert-large" in filename:
        model_name = "CamemBERT"
    return model_name
