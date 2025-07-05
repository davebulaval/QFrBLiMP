from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers import logging

from baseline_models import RandomClassBaselineModel

logging.set_verbosity_warning()


def model_tokenizer_factory(
    model_name, device, token, seed: int = 42, class_to_predict: int = 0
):
    if "gpt" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    elif "llama" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    elif "aléatoire" in model_name.lower():
        model = RandomClassBaselineModel(seed=seed)
        tokenizer = None
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
