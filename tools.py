def filename_to_model_name(filename):
    model_name = None
    if "xlm-roberta-base" in filename:
        model_name = "xlm-roberta-base"
    elif "xlm-roberta-large" in filename:
        model_name = "xlm-roberta-large"
    elif "bert-base" in filename:
        model_name = "bert-base-lang"
    elif "Llama" in filename:
        model_name = "Llama"
    elif "camembert-base" in filename:
        model_name = "camembert-base"
    elif "camembert-large" in filename:
        model_name = "camembert-large"
    else:
        print("a")
    return model_name
