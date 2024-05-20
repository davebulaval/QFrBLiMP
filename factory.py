from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def model_tokenizer_factory(model_name, device, token):
    if "gpt" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    elif "llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
